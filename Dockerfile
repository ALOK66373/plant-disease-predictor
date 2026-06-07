# ─────────────────────────────────────────────────────────────────────────────
# Plant Disease Predictor — Dockerfile
# Base: slim Python 3.10 image (TF 2.15 supports Python 3.8–3.11)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System labels ─────────────────────────────────────────────────────────────
LABEL maintainer="Plant Disease App"
LABEL description="Streamlit app for Plant Disease Prediction using TensorFlow"

# ── Set environment variables ─────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Install system dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (cached layer) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app.py .

# ── Copy model and class indices (must exist locally before docker build) ─────
# Place these files in the same folder as your Dockerfile:
#   • plant_disease_prediction_model.h5
#   • class_indices.json
COPY plant_disease_prediction_model.h5 .
COPY class_indices.json .

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run the app ───────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
