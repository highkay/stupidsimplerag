FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    FASTEMBED_CACHE_PATH=/app/model_cache \
    FASTEMBED_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions \
    GUNICORN_WORKERS=2 \
    GUNICORN_THREADS=1

# System deps for optional native wheels (qdrant-client, fastembed, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Preload FastEmbed cache into image (keeps runtime cold-start low)
RUN mkdir -p ${FASTEMBED_CACHE_PATH}
COPY preload_models.py ./preload_models.py
RUN python preload_models.py && rm preload_models.py

COPY . .

EXPOSE 8000
# Use Gunicorn + Uvicorn workers for production
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-1} -b 0.0.0.0:8000 app.main:app --graceful-timeout 30 --timeout 120"]
