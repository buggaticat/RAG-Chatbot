"""Create text and image embeddings for the ingested corpus."""

import base64
import hashlib
import json
import re
from binascii import Error as BinasciiError
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Iterator, List, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import BlipForConditionalGeneration, BlipProcessor

from .build_documents import iter_all_documents
from .config import (
    BATCH_SIZE,
    BLIP_MODEL_NAME,
    EMBEDDING_CHECKPOINT_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    NORMALIZE_EMBEDDINGS,
    PREPROCESSING_HASH_SEED,
)
from .progress import ProgressBar


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding pipeline."""

    model_name: str = EMBEDDING_MODEL_NAME
    embedding_version: str = EMBEDDING_VERSION
    normalize: bool = NORMALIZE_EMBEDDINGS


embedding_config = EmbeddingConfig()
try:
    embedding_model = OpenAIEmbedding(model=embedding_config.model_name)
except TypeError:
    embedding_model = OpenAIEmbedding()
_PREPROCESSING_HASH = hashlib.sha256(
    PREPROCESSING_HASH_SEED.encode("utf-8")
).hexdigest()

blip_processor = None
blip_model = None
blip_device = None
TASK_BATCH_SIZE = BATCH_SIZE
EmbeddingRecord = Tuple[str, List[float], dict]


def _load_checkpoint() -> dict:
    """Load embedding checkpoint state from disk."""

    if not EMBEDDING_CHECKPOINT_PATH.exists():
        return {"next_document_index": 0, "embeddings": []}

    checkpoint_text = EMBEDDING_CHECKPOINT_PATH.read_text(encoding="utf-8").strip()
    if not checkpoint_text:
        return {"next_document_index": 0, "embeddings": []}
    try:
        checkpoint = json.loads(checkpoint_text)
    except json.JSONDecodeError:
        return {"next_document_index": 0, "embeddings": []}

    if "next_task_index" not in checkpoint and "next_document_index" in checkpoint:
        checkpoint["next_task_index"] = checkpoint["next_document_index"]
    return checkpoint


def _save_checkpoint(state: dict) -> None:
    """Persist embedding checkpoint state to disk."""

    EMBEDDING_CHECKPOINT_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_progress_checkpoint(next_task_index: int) -> None:
    """Persist only the embedding progress cursor without embedding payloads."""

    _save_checkpoint(
        {
            "next_task_index": next_task_index,
            "embeddings": [],
        }
    )


def _clear_checkpoint() -> None:
    """Remove the embedding checkpoint after a successful full sync."""

    if EMBEDDING_CHECKPOINT_PATH.exists():
        try:
            EMBEDDING_CHECKPOINT_PATH.unlink()
        except PermissionError:
            # Windows can briefly lock the file during concurrent antivirus/indexer access.
            # Leaving the checkpoint behind is safer than crashing a successful sync.
            pass


def _normalize_vector(vector: List[float]) -> List[float]:
    """Optionally L2-normalize an embedding vector."""

    if not embedding_config.normalize:
        return vector

    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array.tolist()
    return (array / norm).tolist()


def _get_blip_components():
    """Lazily load the BLIP captioning model and processor."""

    global blip_processor, blip_model, blip_device

    if blip_processor is None or blip_model is None or blip_device is None:
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
        blip_device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(blip_device)

    return blip_model, blip_processor, blip_device


def _decode_base64_image(image_data: str) -> Image.Image:
    """Decode a base64 image string into an RGB PIL image."""

    image_data = image_data.strip()
    if image_data.startswith("data:") and "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_data = re.sub(r"\s+", "", image_data)
    image_data = image_data.replace("-", "+").replace("_", "/")
    padding = len(image_data) % 4
    if padding:
        image_data += "=" * (4 - padding)
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _warn(message: str) -> None:
    """Emit a lightweight ingestion warning."""

    print(f"[embedding-corpus] {message}")


def _image_data_preview(image_data: object, limit: int = 80) -> str:
    """Return a short preview for logging skipped image payloads safely."""

    if image_data is None:
        return "<none>"
    if not isinstance(image_data, str):
        return f"<non-string {type(image_data).__name__}>"
    if not image_data:
        return "<empty>"
    preview = image_data[:limit]
    if len(image_data) > limit:
        preview += "..."
    return preview


def _generate_caption(blip_model, processor, device, image):
    """Generate a short caption for an image using BLIP."""

    inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def _embedding_metadata(document_metadata: dict, source_field: str) -> dict:
    """Attach embedding provenance metadata to a record."""

    return {
        **document_metadata,
        "source_field": source_field,
        "embedding_model": embedding_config.model_name,
        "embedding_version": embedding_config.embedding_version,
        "preprocessing_hash": _PREPROCESSING_HASH,
        "embedded_at": datetime.now(timezone.utc).isoformat(),
    }


def _serialize_embedding_record(record: EmbeddingRecord) -> dict:
    """Convert an embedding tuple into checkpoint-friendly JSON."""

    content, vector, metadata = record
    return {"content": content, "vector": vector, "metadata": metadata}


def _deserialize_embedding_record(record: dict) -> EmbeddingRecord:
    """Convert a checkpoint record back into an embedding tuple."""

    return (
        record["content"],
        list(record["vector"]),
        dict(record["metadata"]),
    )


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, falling back to single-item calls when needed."""

    if not texts:
        return []

    batch_embed = getattr(embedding_model, "get_text_embedding_batch", None)
    if callable(batch_embed):
        try:
            vectors = batch_embed(texts)
            if len(vectors) == len(texts):
                return list(vectors)
        except TypeError:
            # Some embedding backends expose the batch method with a different signature.
            pass

    return [embedding_model.get_text_embedding(text) for text in texts]


def _iter_embedding_tasks(documents) -> Iterator[dict]:
    """Stream document tasks from a document iterator."""

    for document in documents:
        node_content = document.get_content()
        node_metadata = document.metadata

        if node_content:
            yield {
                "kind": "text",
                "content": node_content,
                "metadata": node_metadata,
            }

        if node_metadata.get("source_field") == "sections.text":
            images = node_metadata.get("images", {}) or {}
            for image_id, image_data in images.items():
                yield {
                    "kind": "image",
                    "image_id": image_id,
                    "image_data": image_data,
                    "metadata": node_metadata,
                }


def get_checkpoint_embeddings(checkpoint: dict | None = None) -> List[EmbeddingRecord]:
    """Return any embeddings already stored in the local checkpoint."""

    if checkpoint is None:
        checkpoint = _load_checkpoint()
    return [
        _deserialize_embedding_record(record)
        for record in checkpoint.get("embeddings", [])
    ]


def iter_document_embedding_batches(
    checkpoint: dict | None = None,
    documents=None,
) -> Iterator[tuple[int, List[EmbeddingRecord]]]:
    """Yield embedded document batches without accumulating the full corpus."""

    if checkpoint is None:
        checkpoint = _load_checkpoint()
    start_index = int(checkpoint.get("next_task_index", 0) or 0)
    if documents is None:
        documents = iter_all_documents()

    progress = ProgressBar(
        total=None,
        label="embedding corpus",
    )
    seen_image_keys: set[tuple[str, str, str]] = set()
    task_index = 0
    pending_embeddings: List[tuple[str, dict, str]] = []

    def flush_pending() -> List[EmbeddingRecord]:
        batch_records: List[EmbeddingRecord] = []
        if not pending_embeddings:
            return batch_records

        vectors = _embed_texts([content for content, _, _ in pending_embeddings])
        for (content, node_metadata, source_field), vector in zip(pending_embeddings, vectors):
            batch_records.append(
                (
                    content,
                    _normalize_vector(vector),
                    _embedding_metadata(node_metadata, source_field),
                )
            )
        pending_embeddings.clear()
        return batch_records

    for task in _iter_embedding_tasks(documents):
        if task_index < start_index:
            task_index += 1
            continue

        if task["kind"] == "text":
            pending_embeddings.append(
                (
                    task["content"],
                    task["metadata"],
                    task["metadata"].get("source_field", "text"),
                )
            )
            progress.update()
        else:
            image_id = task["image_id"]
            image_data = task["image_data"]
            node_metadata = task["metadata"]
            if not image_data:
                _warn(
                    f"Skipping empty image payload paper_id={node_metadata.get('paper_id')} "
                    f"section_id={node_metadata.get('section_id')} image_id={image_id} "
                    f"image_data={_image_data_preview(image_data)}"
                )
                progress.update()
                task_index += 1
                continue

            image_key = (
                str(node_metadata.get("paper_id", "")),
                str(node_metadata.get("section_id", "")),
                str(image_id),
            )
            if image_key in seen_image_keys:
                progress.update()
                task_index += 1
                continue
            seen_image_keys.add(image_key)
            try:
                image = _decode_base64_image(image_data)
                current_model, current_processor, current_device = _get_blip_components()
                image_caption = _generate_caption(
                    current_model, current_processor, current_device, image
                )
                pending_embeddings.append(
                    (
                        image_caption,
                        {**node_metadata, "image_id": image_id},
                        "sections.images",
                    )
                )
            except (BinasciiError, UnidentifiedImageError, OSError, ValueError) as exc:
                _warn(
                    f"Skipping unreadable image paper_id={node_metadata.get('paper_id')} "
                    f"section_id={node_metadata.get('section_id')} image_id={image_id} "
                    f"image_data={_image_data_preview(image_data)}: {exc}"
                )
            finally:
                progress.update()

        task_index += 1

        if len(pending_embeddings) < TASK_BATCH_SIZE:
            continue

        yield task_index, flush_pending()

    if pending_embeddings:
        yield task_index, flush_pending()

    progress.finish("Embedding corpus complete")


def documents_to_embeddings() -> List[EmbeddingRecord]:
    """Convert all documents, tables, and image captions into embeddings."""

    checkpoint = _load_checkpoint()
    all_embedding = get_checkpoint_embeddings(checkpoint)
    for next_task_index, batch_records in iter_document_embedding_batches(checkpoint, iter_all_documents()):
        all_embedding.extend(batch_records)
        _save_progress_checkpoint(next_task_index)
    return all_embedding


if __name__ == "__main__":
    documents_to_embeddings()
