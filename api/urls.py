from django.urls import path
from .views import (
    UploadAndAnalyzeView,
    ResultRetrieveView,
    HealthCheckView,
)

urlpatterns = [
    path("upload/", UploadAndAnalyzeView.as_view(), name="upload_analyze"),
    path("results/<str:report_id>/", ResultRetrieveView.as_view(), name="result_retrieve"),
    path("health/", HealthCheckView.as_view(), name="health"),
]
