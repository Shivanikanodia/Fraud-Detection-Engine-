# -----------------------------
# Base image (stable for ML)
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# (needed for lightgbm, xgboost)
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements & install
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application files
# -----------------------------
COPY apps.py .
COPY fraud_xgb_pipeline.joblib .
COPY templates/ templates/

# -----------------------------
# Expose Flask port
# -----------------------------
EXPOSE 5000

# -----------------------------
# Run Flask app
# -----------------------------
CMD ["python", "apps.py"]
