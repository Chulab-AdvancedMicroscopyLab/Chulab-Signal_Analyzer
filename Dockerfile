# Dockerfile for Zarr-Tiff Format Transform
FROM python:3.10-slim

# Set up basic package
RUN apt-get update && apt-get install -y \
    build-essential \
    pocl-opencl-icd \
    opencl-headers \
    clinfo

# Install Package dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

COPY cell_analyzer.py .
COPY vessel_analyzer.py .