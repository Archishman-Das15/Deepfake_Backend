# 🛡️ DeepFake Defense Backend

## 🔹 Overview
This is the **backend service** for the **DeepFake Defense WebApp**.  
It is built using **Django + Django REST Framework** and integrates with:
- **Firebase** for authentication & file storage
- **ONNX/Torch** models for deepfake detection
- **Database** (PostgreSQL / Firestore – handled by DB teammate)

---

## 📂 Project Structure

DeepFake-Backend/
│── backend/ # Django project root
│ ├── settings.py # Django settings (add Firebase/DB configs here)
│ ├── urls.py # Root URL routes
│
│── api/ # REST API app
│ ├── views.py # Upload, detect, results, health
│ ├── urls.py # API routes
│ ├── serializers.py # Upload validation
│ ├── permissions.py # Firebase auth check
│
│── storage_service/ # Firebase storage integration
│ └── firebase_storage.py
│
│── model_service/ # ML model + preprocessing
│ ├── deepfake_model.py # Loads & runs inference
│ └── preprocess.py # Face detection / preprocessing
│
│── database/ (to be implemented by DB teammate)
│ ├── schemas/ # Report / User schemas
│ ├── queries/ # Save & fetch reports
│
│── requirements.txt # Dependencies
│── manage.py
---

## 🔹 Setup (Local Development)

### 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)
2. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt
If something is missing, install manually:

pip install django djangorestframework firebase-admin torch torchvision torchaudio facenet-pytorch opencv-python-headless
3. Run migrations
python manage.py makemigrations
python manage.py migrate
4. Run the development server
python manage.py runserver