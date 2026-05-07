# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    APP_TIMEZONE=Asia/Shanghai \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=1

# Runtime system deps only. Python deps should resolve from wheels.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl tzdata; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url ${PIP_INDEX_URL} -r requirements.txt

ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    FASTEMBED_CACHE_PATH=/app/model_cache \
    FASTEMBED_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions

# Preload FastEmbed cache into image (keeps runtime cold-start low)
ARG PRELOAD_FASTEMBED=1
RUN mkdir -p ${FASTEMBED_CACHE_PATH}
COPY model_cache/ ${FASTEMBED_CACHE_PATH}/
COPY preload_models.py ./preload_models.py
RUN --mount=type=cache,id=fastembed-cache-v2,target=/tmp/fastembed-cache \
    set -eux; \
    if [ "${PRELOAD_FASTEMBED}" = "1" ]; then \
        cp -a ${FASTEMBED_CACHE_PATH}/. /tmp/fastembed-cache/; \
        FASTEMBED_CACHE_PATH=/tmp/fastembed-cache python preload_models.py; \
        cp -a /tmp/fastembed-cache/. ${FASTEMBED_CACHE_PATH}/; \
    else \
        echo "Skipping FastEmbed preload"; \
    fi; \
    rm preload_models.py

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Use Gunicorn + Uvicorn workers for production
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${GUNICORN_WORKERS:-1} --threads ${GUNICORN_THREADS:-1} -b 0.0.0.0:8000 app.main:app --graceful-timeout 30 --timeout 120"]
