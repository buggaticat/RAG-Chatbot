import json
import os
from typing import Any, Dict, List

import boto3
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "open_ragbench/pdf/arxiv/corpus/"

s3 = boto3.client("s3")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def _safe_metadata(value: Any) -> Any:
    """Convert nested values into metadata-friendly primitives."""
    if isinstance(value, dict):
        return {k: _safe_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_metadata(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _chunk_text(
    text: str,
    base_metadata: Dict[str, Any],
    source_field: str,
) -> List[Document]:
    if not text:
        return []

    chunks = splitter.split_text(text)
    documents: List[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        documents.append(
            Document(
                text=chunk,
                metadata={
                    **base_metadata,
                    "source_field": source_field,
                    "chunk_index": chunk_index,
                },
            )
        )

    return documents


def _build_documents_from_paper(json_data: Dict[str, Any]) -> List[Document]:
    """
    Build retrieval documents from the paper JSON while only chunking:
    - abstract
    - sections[*].text
    """
    base_metadata = {
        "paper_id": json_data.get("id"),
        "title": json_data.get("title"),
        "authors": _safe_metadata(json_data.get("authors", [])),
        "categories": _safe_metadata(json_data.get("categories", [])),
        "updated": json_data.get("updated"),
        "published": json_data.get("published"),
    }

    documents: List[Document] = []

    abstract = json_data.get("abstract", "")
    documents.extend(_chunk_text(abstract, base_metadata, "abstract"))

    for section in json_data.get("sections", []) or []:
        section_id = section.get("section_id")
        section_metadata = {
            **base_metadata,
            "section_id": section_id,
            "tables": _safe_metadata(section.get("tables", {})),
            "images": _safe_metadata(section.get("images", {})),
        }

        documents.extend(
            _chunk_text(
                section.get("text", ""),
                section_metadata,
                "sections.text",
            )
        )

    return documents


def build_all_documents() -> List[Document]:
    all_documents: List[Document] = []

    continuation_token = None

    while True:
        request_kwargs = {
            "Bucket": BUCKET_NAME,
            "Prefix": S3_PREFIX,
        }
        if continuation_token:
            request_kwargs["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**request_kwargs)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            json_data = json.loads(file_obj["Body"].read().decode("utf-8"))
            documents = _build_documents_from_paper(json_data)
            all_documents.extend(documents)

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")

    return all_documents

if __name__ == "__main__":
    documents = build_all_documents()
    print(f"Built {len(documents)} documents")


{
  "title": "Paper Title",
  "sections": [
    {
      "section_id": 0,
      "text": "Section text content with placeholders for tables/images",
      "tables": {
        "table_id1": "markdown_table_string"
      },
      "images": {
        "image_id1": "base64_encoded_string"
      }
    }
  ],
  "id": "Paper ID",
  "authors": ["Author1", "Author2"],
  "categories": ["Category1", "Category2"],
  "abstract": "Abstract text",
  "updated": "Updated date",
  "published": "Published date"
}
