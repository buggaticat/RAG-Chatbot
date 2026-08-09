import os

from typing import List
from qdrant_client import QdrantClient, models

from .embed import documents_to_embeddings

QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
COLLECTION_NAME = "embedding_collection"
BATCH_SIZE = 64

client = QdrantClient(
    url=QDRANT_CLUSTER_ENDPOINT,
    api_key=QDRANT_APIKEY
)

def _ensure_collection(vector_size: int) -> None:
    existing_collections = {
        collection.name for collection in client.get_collections().collections
    }

    if COLLECTION_NAME in existing_collections:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

def main() -> None:
    all_embedding = documents_to_embeddings()
    Point = models.PointStruct

    if not all_embedding:
        return

    _ensure_collection(vector_size=len(all_embedding[0][1]))

    points: List[models.PointStruct] = []
    for i, (_, embedding, metadata) in enumerate(all_embedding):
        point = Point(
            id=i,
            vector=embedding,
            payload=metadata,
        )
        points.append(point)

        if len(points) == BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            points = []

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )


if __name__ == "__main__":
    main()
    
