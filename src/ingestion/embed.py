import torch
import base64


from io import BytesIO
from typing import List, Tuple

from PIL import Image
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import BlipProcessor, BlipForConditionalGeneration

from .build_documents import build_all_documents

embedding_model = OpenAIEmbedding()
_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

documents = build_all_documents()

def _get_blip_components():
    global blip_processor, blip_model, blip_device

    if blip_processor is None or blip_model is None or blip_device is None:
        blip_processor = BlipProcessor.from_pretrained(_BLIP_MODEL_NAME)
        blip_model = BlipForConditionalGeneration.from_pretrained(_BLIP_MODEL_NAME)
        blip_device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(blip_device)

    return blip_model, blip_processor, blip_device

def _decode_base64_image(image_data: str) -> Image.Image:
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def _generate_caption(blip_model, processor, device, image):
    inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def documents_to_embeddings() -> List[Tuple[str, List[float], dict]]:
    all_embedding: List[Tuple[str, List[float], dict]] = []

    for document in documents:
        node_content = document.get_content()
        node_metadata = document.metadata

        tables = node_metadata.get("tables", {})
        images = node_metadata.get("images", {})

        if node_content:
            text_embedding = embedding_model.get_text_embedding(node_content)
            all_embedding.append((node_content, text_embedding, document.metadata))

        if tables:
            for table_id, table_markdown in tables.items():
                if not table_markdown:
                    continue
                tables_embedding = embedding_model.get_text_embedding(table_markdown)
                all_embedding.append((table_markdown, tables_embedding, {**document.metadata, "table_id": table_id}))

        if images:
            for image_id, image_data in images.items():
                if not image_data:
                    continue
                image = _decode_base64_image(image_data)
                current_model, current_processor, current_device = _get_blip_components()
                image_caption = _generate_caption(current_model, current_processor, current_device, image)
                image_embedding = embedding_model.get_text_embedding(image_caption)
                all_embedding.append(
                    (
                        image_caption,
                        image_embedding,
                        {**document.metadata, "image_id": image_id},
                    )
                )

    return all_embedding

if __name__ == "__main__":
    documents_to_embeddings()
