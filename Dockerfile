# syntax=docker/dockerfile:1
# ToxiClean AI — OpenEnv Docker Image
#
# Serves:
#   • FastAPI REST API  → POST /reset, POST /step, GET /state  (OpenEnv spec)
#   • Gradio Web UI     → GET  /ui                              (interactive demo)
#   • Health check      → GET  /health
#
# Build:
#   docker build -t toxiclean-ai .
#
# Run:
#   docker run --rm -p 7860:7860 \
#     -e HF_TOKEN=hf_... \
#     -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
#     toxiclean-ai

FROM python:3.10-slim

# ── Metadata ──────────────────────────────────────────────────────────────
LABEL maintainer="ToxiClean AI Team"
LABEL description="OpenEnv RL environment for intelligent content moderation"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (layer-cached) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY core/       ./core/
COPY inference.py .
COPY server.py    .
COPY app.py       .
COPY openenv.yaml .
COPY .env.example .

# ── Non-root user for security ─────────────────────────────────────────────
RUN useradd -m -s /bin/bash toxiclean \
    && chown -R toxiclean:toxiclean /app
USER toxiclean

# ── Environment defaults (override at runtime via -e or .env) ──────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV PORT=7860

# ── Port ───────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ───────────────────────────────────────────────────────────
# Pings /health — same endpoint the submission validator chain uses
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

# ── Default command ─────────────────────────────────────────────────────────
# server.py starts uvicorn, which serves FastAPI (/reset /step /state) +
# Gradio UI mounted at /ui
CMD ["python", "server.py"]
