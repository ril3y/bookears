FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY asr/ ./asr/
COPY bakeoff/ ./bakeoff/
COPY server.py .

# Model cache: bind-mount your host model dir here to skip re-downloading
# docker-compose.yml maps /mnt/llm-models/stt-models:/models by default
VOLUME /models

# NeMo downloads to HuggingFace cache — point both caches at /models
ENV MODEL_CACHE_DIR=/models
ENV HF_HOME=/models
ENV TORCH_HOME=/models

EXPOSE 8200

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8200/health')"

CMD ["python3", "server.py"]
