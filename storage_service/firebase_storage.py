# storage_service/firebase_storage.py

import os
import tempfile
import firebase_admin
from firebase_admin import credentials, storage


class FirebaseStorage:
    def __init__(self, bucket_name: str = None):
        """
        Initialize Firebase Storage client.
        Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON.
        """
        if not firebase_admin._apps:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not cred_path:
                raise RuntimeError(
                    "GOOGLE_APPLICATION_CREDENTIALS not set. "
                    "Set it to your Firebase service account key JSON file."
                )

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                "storageBucket": bucket_name or os.getenv("FIREBASE_STORAGE_BUCKET")
            })

        self.bucket = storage.bucket()

    def upload_fileobj(self, file_obj, destination_blob_name: str) -> str:
        """
        Upload a file-like object to Firebase Storage.
        Returns the gs:// path of the uploaded file.
        """
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_file(file_obj, content_type=file_obj.content_type)
        return f"gs://{self.bucket.name}/{destination_blob_name}"

    def upload_file(self, file_path: str, destination_blob_name: str) -> str:
        """
        Upload a local file path to Firebase Storage.
        Returns the gs:// path.
        """
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        return f"gs://{self.bucket.name}/{destination_blob_name}"

    def download_blob_to_local(self, blob_path: str) -> str:
        """
        Download a blob from Firebase Storage to a temporary local file.
        blob_path should be the gs://bucket/path style string.
        Returns the local file path.
        """
        if not blob_path.startswith("gs://"):
            raise ValueError("blob_path must start with gs://")

        # Extract object path
        parts = blob_path.replace("gs://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid gs:// path format")

        _, object_path = parts
        blob = self.bucket.blob(object_path)

        temp_fd, temp_path = tempfile.mkstemp()
        os.close(temp_fd)  # close file descriptor
        blob.download_to_filename(temp_path)
        return temp_path

    def get_public_url(self, blob_path: str) -> str:
        """
        Convert a gs:// path to a public HTTPS URL (if your bucket allows public access).
        """
        if not blob_path.startswith("gs://"):
            raise ValueError("blob_path must start with gs://")

        parts = blob_path.replace("gs://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid gs:// path format")

        bucket_name, object_path = parts
        return f"https://storage.googleapis.com/{bucket_name}/{object_path}"
