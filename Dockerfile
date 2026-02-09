# ─────────────────────────────────────────────────────────────
# CertiRAG Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1: Install dependencies
#   Stage 2: Copy application code
#
# Usage:
#   docker build -t certirag .
#   docker run -it --rm -p 8501:8501 certirag
#   docker run -it --rm --gpus all -p 8501:8501 certirag  # GPU mode
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[lite,dev]"

# Copy application code
COPY . .

# Install the project itself
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/index data/cache eval_results

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run Streamlit demo
ENV CERTIRAG_MODE=lite
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

# ─────────────────────────────────────────────────────────────
# GPU variant (use as build target: --target gpu)
# ─────────────────────────────────────────────────────────────
FROM base AS gpu

# Install GPU dependencies
RUN pip install --no-cache-dir -e ".[full,dev]"

ENV CERTIRAG_MODE=full
