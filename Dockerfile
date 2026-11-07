# syntax=docker/dockerfile:1.4
# VoxSplit Production Dockerfile
# Optimized for Python 3.11, CTranslate2 (faster-whisper), libsndfile, and int8 CPU inference

FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HF_HOME=/home/appuser/.cache/huggingface

# Install system audio codecs, OpenMP runtime (CTranslate2 requirement), and health check tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies into system Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and sample assets
COPY app/ ./app/
COPY samples/ ./samples/
COPY pytest.ini .

# Create cache directory with proper ownership
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /app /home/appuser

# Switch to unprivileged user
USER appuser

# Expose HTTP and WebSocket port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

