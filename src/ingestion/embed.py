from typing import List, Tuple

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings


from .build_documents import build_all_documents

embedding_model = OpenAIEmbedding()

Settings.embed_model = embedding_model

def documents_to_embeddings() -> List[Tuple[str, List[float], dict]]:
    documents = build_all_documents()

    all_embedding: List[Tuple[str, List[float], dict]] = []

    for document in documents:
        node_content = document.get_content()
        embedding = embedding_model.get_text_embedding(node_content)
        all_embedding.append((node_content, embedding, document.metadata))

    return all_embedding

if __name__ == "__main__":
    documents_to_embeddings()

