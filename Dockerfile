# ====================================================================== #
# Dockerfile - NEU-DET Steel Defect Classifier - Cloud Deployment
# ====================================================================== #
# Multi-stage build:
#   Stage 1: TensorFlow Serving image that loads the SavedModel.
#   Stage 2: Python slim image running the Flask API gateway.
#
# Pada platform cloud (Railway/Heroku) hanya satu port yang terekspos,
# sehingga stage 2 menghasilkan image final yang berisi Flask app.
# TF Serving dijalankan sebagai sidecar terpisah atau sebagai bagian
# dari docker-compose pada deployment produksi.
# ====================================================================== #

FROM python:3.10-slim AS runtime

# ---------------------------------------------------------------- #
# 1. System dependencies
# ---------------------------------------------------------------- #
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- #
# 2. Python dependencies
# ---------------------------------------------------------------- #
WORKDIR /app

COPY requirements-docker.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------- #
# 3. Application code & model
# ---------------------------------------------------------------- #
COPY app.py .
COPY serving_model/ ./serving_model/

# ---------------------------------------------------------------- #
# 4. Runtime configuration
# ---------------------------------------------------------------- #
ENV PORT=8080 \
    TF_SERVING_URL=http://localhost:8501/v1/models/neu_det_cnn:predict \
    MODEL_NAME=neu_det_cnn

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8080}/health || exit 1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120 app:app"]
