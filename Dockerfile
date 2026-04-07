# syntax=docker/dockerfile:1
# ToxiClean AI — OpenEnv Docker Image
# Python 3.10 slim for a lean production-ready image

FROM python:3.10-slim

# ── Metadata ──────────────────────────────────────────────────────────────
LABEL maintainer="ToxiClean AI Team"
LABEL description="OpenEnv RL environment for intelligent content moderation"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────
# curl is useful for health-checks in orchestrated environments
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
# Copy the env package, inference script, and config
COPY env/ ./env/
COPY inference.py .
COPY openenv.yaml .

# Optional: copy .env.example so users know what to set
COPY .env.example .

# ── Non-root user for security ─────────────────────────────────────────────
RUN useradd -m -s /bin/bash toxiclean
USER toxiclean

# ── Environment defaults ───────────────────────────────────────────────────
# These can be overridden at runtime via -e flags or a .env file
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini

# ── Health check ───────────────────────────────────────────────────────────
# Verifies the env package is importable; exits quickly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from env import ToxiCleanEnv; ToxiCleanEnv()" || exit 1

# ── Default command ─────────────────────────────────────────────────────────
CMD ["python", "inference.py"]
