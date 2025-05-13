# Use an official Python runtime as a parent image
FROM python:3.11-bullseye

WORKDIR /app

# Install system and venv dependencies
RUN apt-get update && apt-get install -y \
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
    python3-venv \
    && apt-get clean

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip within the venv
RUN pip install --upgrade pip

# Copy and install Python dependencies inside the venv
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

EXPOSE 80

# Launch the application using Uvicorn from the venv
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
