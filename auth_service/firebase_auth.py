import firebase_admin
from firebase_admin import credentials, auth
import os

FIREBASE_CRED_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT", "/secrets/firebase-service-account.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    default_app = firebase_admin.initialize_app(cred)


def verify_firebase_token(id_token):
    """
    Verifies Firebase ID token and returns decoded token dict.
    Raises on failure.
    """
    decoded_token = auth.verify_id_token(id_token)
    return decoded_token
