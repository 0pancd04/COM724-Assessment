# Use an official Python runtime as a parent image
FROM python:3.11-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy requirements and install dependencies with uv
COPY requirements.txt /app/requirements.txt
RUN uv venv && uv pip install -r requirements.txt

# Copy application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

EXPOSE 80

# Launch the application using Uvicorn from the virtual environment
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
