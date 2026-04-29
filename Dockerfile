# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Create a non-root user for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and change ownership to the non-root user
COPY --chown=user . .

# Hugging Face Spaces requires port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Use gunicorn for production
CMD exec gunicorn --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --keep-alive 5 \
    app:app
