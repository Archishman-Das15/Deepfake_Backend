from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),     # Django admin panel
    path('api/', include('api.urls')),   # Your API endpoints
]
