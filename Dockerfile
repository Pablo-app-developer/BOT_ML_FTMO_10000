# Dockerfile for Antigravity FTMO Bot (Crypto/Linux)

# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for logs and data if they don't exist
RUN mkdir -p logs data models shared

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check (optional, checks if process is running)
# HEALTHCHECK CMD pgrep python || exit 1

# Default command (can be overridden in docker-compose)
# Currently set to run the main execution loop (interface)
CMD ["python", "main.py"]
