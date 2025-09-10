import os
from google.cloud import storage

def download_model(bucket_name, blob_name, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    blob.download_to_filename(dest_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {dest_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--dest", default="/app/model_service/models/deepfake_model.onnx")
    args = p.parse_args()
    download_model(args.bucket, args.blob, args.dest)
