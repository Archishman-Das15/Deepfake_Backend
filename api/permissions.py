from rest_framework.permissions import BasePermission
from auth_service.firebase_auth import verify_firebase_token

class IsFirebaseAuthenticated(BasePermission):
    """
    Check Firebase ID token in Authorization header 'Bearer <token>'
    """

    def has_permission(self, request, view):
        auth = request.headers.get("Authorization")
        if not auth:
            return False
        parts = auth.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        token = parts[1]
        try:
            request.user_firebase = verify_firebase_token(token)
            return True
        except Exception:
            return False
