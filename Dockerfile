# Use a minimal Python image
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Define arguments
ARG ENVIRONMENT
ARG HOST
ARG PORT
ARG LOG_LEVEL
ARG MODEL_PATH
ARG HUGGINGFACE_MODEL
ARG BASELINE_MODEL_PATH
ARG VECTORIZER_PATH

# Export as environment variables
ENV ENVIRONMENT=$ENVIRONMENT
ENV HOST=$HOST
ENV PORT=$PORT
ENV LOG_LEVEL=$LOG_LEVEL
ENV MODEL_PATH=$MODEL_PATH
ENV HUGGINGFACE_MODEL=$HUGGINGFACE_MODEL
ENV BASELINE_MODEL_PATH=$BASELINE_MODEL_PATH
ENV VECTORIZER_PATH=$VECTORIZER_PATH

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (avoid unnecessary files)
COPY . .

# Copy models
COPY src/models /app/src/models

# Expose FastAPI port
EXPOSE 7860

# Default command: Run API using Gunicorn with Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:7860", "src.api.main:app"]
