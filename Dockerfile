FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential ffmpeg libsndfile1 git

# copy files
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Collect static if any (not needed now)
ENV PORT=8080
EXPOSE ${PORT}

CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8080", "--workers", "3"]
