# Use Python 3.13 slim as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Copy the model files into the correct directory inside the container
COPY src/models/baseline_model.pkl /app/models/baseline_model.pkl

# Expose FastAPI port
EXPOSE 7860

# Default command (Gunicorn for production)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:7860", "src.api.main:app"]
