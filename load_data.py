import os
from tempfile import TemporaryDirectory

import boto3
from huggingface_hub import snapshot_download


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REPO_ID = "vectara/open_ragbench"
HF_TOKEN = os.getenv("HF_TOKEN")

# Download the processed dataset assets used for RAG ingestion.
FILES_TO_DOWNLOAD = [
    "pdf/arxiv/answers.json",
    "pdf/arxiv/corpus/**",
    "pdf/arxiv/qrels.json",
    "pdf/arxiv/queries.json",
]


def main() -> None:
    if BUCKET_NAME == "your-aws-s3-bucket-name":
        raise ValueError("Set S3_BUCKET_NAME before running this script.")

    if not HF_TOKEN:
        raise ValueError("Set HF_TOKEN to a Hugging Face read token before running this script.")

    s3_client = boto3.client("s3")

    with TemporaryDirectory() as temp_dir:
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=temp_dir,
            allow_patterns=FILES_TO_DOWNLOAD,
        )

        for root, _, files in os.walk(local_dir):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
                s3_key = f"datasets/{REPO_ID}/{relative_path}"
                s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
                print(f"Uploaded {relative_path} to s3://{BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    main()

