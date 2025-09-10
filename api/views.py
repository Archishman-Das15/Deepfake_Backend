import uuid
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UploadSerializer
from .permissions import IsFirebaseAuthenticated
from storage_service.firebase_storage import FirebaseStorage
from model_service.deepfake_model import DeepfakeModel
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes

# Initialize model singleton (loads at import)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model_service/models/deepfake_model.onnx")
model = DeepfakeModel(MODEL_PATH)

storage = FirebaseStorage()  # uses GOOGLE_APPLICATION_CREDENTIALS env var


class HealthCheckView(APIView):
    def get(self, request):
        return Response({"status": "ok"}, status=200)


class UploadAndAnalyzeView(APIView):
    permission_classes = [IsFirebaseAuthenticated]

    def post(self, request):
        serializer = UploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"error": serializer.errors}, status=400)

        f = serializer.validated_data["file"]
        filename = f"{uuid.uuid4().hex}_{f.name}"

        # Upload file to Firebase Storage and get public URL or gs:// path
        blob_path = storage.upload_fileobj(f, filename)
        # blob_path is a gs://bucket/path or https link depending on implementation
        # Model expects a local path or downloadable URL.
        try:
            local_path = storage.download_blob_to_local(blob_path)
        except Exception as e:
            return Response({"error": "storage download failed", "detail": str(e)}, status=500)

        # Run inference (DeepfakeModel handles images & videos)
        try:
            result = model.predict(local_path)
        except Exception as e:
            return Response({"error": "inference failed", "detail": str(e)}, status=500)

        # Optionally save report to DB (integration point)
        report_id = uuid.uuid4().hex
        report = {
            "report_id": report_id,
            "user_id": request.user_firebase.get("uid"),
            "file_name": filename,
            "storage_path": blob_path,
            "result": result,
        }
        # Here call integration hook or database save (Integration teammate will implement)
        # e.g., from database.queries import save_report; save_report(report)

        # Clean up local file
        try:
            os.remove(local_path)
        except:
            pass

        return Response(report, status=200)


class ResultRetrieveView(APIView):
    permission_classes = [IsFirebaseAuthenticated]

    def get(self, request, report_id):
        # Integration: query DB for report by report_id
        # return the saved report structure
        return Response({"detail": "Integration: implement DB lookup"}, status=501)


@api_view(["POST"])
@permission_classes([IsFirebaseAuthenticated])
def detect_deepfake(request):
    f = request.FILES.get("file")
    if not f:
        return Response({"error": "No file provided"}, status=400)

    filename = f"{uuid.uuid4().hex}_{f.name}"
    blob_path = storage.upload_fileobj(f, filename)

    try:
        local_path = storage.download_blob_to_local(blob_path)
        result = model.predict(local_path)
        os.remove(local_path)
    except Exception as e:
        return Response({"error": "detection failed", "detail": str(e)}, status=500)

    return Response({"result": result}, status=200)
