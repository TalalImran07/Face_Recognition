# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for building dlib and CMake
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to utilize Docker's cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy the rest of the application code
COPY . /app

# Expose ports for gRPC and Flask
EXPOSE 50051
EXPOSE 50052

# Command to run the gRPC server when the container starts
CMD ["python", "main.py"]
