FROM python:3.10-slim

WORKDIR /app

ENV FASTEMBED_CACHE_PATH=/app/model_cache \
    FASTEMBED_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions

# 安装系统依赖 (如有需要)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ${FASTEMBED_CACHE_PATH}
COPY preload_models.py ./preload_models.py
RUN python preload_models.py && rm preload_models.py

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
