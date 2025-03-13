# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (including git for version control)
RUN apt-get update && apt-get install -y gcc git

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire directory (includes the app folder)
COPY . /app

# Expose the port the app runs on
EXPOSE 80

# Run the FastAPI app with uvicorn in reload mode for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
