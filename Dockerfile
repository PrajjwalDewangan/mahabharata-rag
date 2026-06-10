# Dockerfile for Hugging Face Spaces
# HF Spaces expects the app on port 7860

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (data is added separately — see below)
COPY api/        ./api/
COPY generation/ ./generation/
COPY retrieval/  ./retrieval/
COPY ui/         ./ui/

# data/ (chroma + chunks.json) must be present at build time.
# Add it here — Git LFS or direct copy.
COPY data/chunks.json   ./data/chunks.json
COPY data/chroma/       ./data/chroma/

# HF Spaces runs as a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

# ENVIRONMENT=production → Groq backend
# GROQ_API_KEY is injected via HF Spaces Secrets (never hardcoded)
ENV ENVIRONMENT=production
ENV PORT=7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]