"""Create text and image embeddings for the ingested corpus."""

import base64
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import BlipForConditionalGeneration, BlipProcessor

from .build_documents import build_all_documents
from .config import (
    BLIP_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    NORMALIZE_EMBEDDINGS,
    PREPROCESSING_HASH_SEED,
)

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

    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


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

def documents_to_embeddings() -> List[Tuple[str, List[float], dict]]:
    """Convert all documents, tables, and image captions into embeddings."""

    all_embedding: List[Tuple[str, List[float], dict]] = []
    seen_image_keys: set[tuple[str, str, str]] = set()

    for document in build_all_documents():
        node_content = document.get_content()
        node_metadata = document.metadata

        tables = node_metadata.get("tables", {}) or {}
        images = node_metadata.get("images", {}) or {}

        if node_content:
            text_embedding = _normalize_vector(
                embedding_model.get_text_embedding(node_content)
            )
            all_embedding.append(
                (
                    node_content,
                    text_embedding,
                    _embedding_metadata(node_metadata, node_metadata.get("source_field", "text")),
                )
            )

        if node_metadata.get("source_field") == "sections.text" and images:
            for image_id, image_data in images.items():
                if not image_data:
                    continue
                image_key = (
                    str(node_metadata.get("paper_id", "")),
                    str(node_metadata.get("section_id", "")),
                    str(image_id),
                )
                if image_key in seen_image_keys:
                    continue
                seen_image_keys.add(image_key)
                image = _decode_base64_image(image_data)
                current_model, current_processor, current_device = _get_blip_components()
                image_caption = _generate_caption(
                    current_model, current_processor, current_device, image
                )
                image_embedding = _normalize_vector(
                    embedding_model.get_text_embedding(image_caption)
                )
                all_embedding.append(
                    (
                        image_caption,
                        image_embedding,
                        _embedding_metadata(
                            {**document.metadata, "image_id": image_id},
                            "sections.images",
                        ),
                    )
                )

    return all_embedding


if __name__ == "__main__":
    documents_to_embeddings()
