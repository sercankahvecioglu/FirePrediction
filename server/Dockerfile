FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements-docker.txt ./requirements.txt
RUN ls -la /app && python -V && pip -V

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/RAW_IMAGES data/CLOUD_IMAGES data/TILES_IMAGES \
    data/FOREST_IMAGES data/FIRE_IMAGES data/METADATA data/DISPLAY

# Expose both ports (main API and Telegram bot API)
EXPOSE 5001 5002

# Use the startup script to run both services
CMD ["python", "start_services.py"]
