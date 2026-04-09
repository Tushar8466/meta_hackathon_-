FROM python:3.10-slim

# Install system dependencies for OpenCV/Pillow if needed
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variables (defaults)
ENV API_BASE_URL="http://0.0.0.0:7860"
ENV MODEL_NAME="crop-disease-v1"

# HF Spaces expect port 7860
EXPOSE 7860

# Run the environment server using the module syntax
CMD ["python", "-m", "server.app"]