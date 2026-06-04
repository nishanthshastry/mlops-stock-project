FROM python:3.11-slim-bookworm

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install "dvc[s3]"

# Copy project files
COPY . .

# Copy startup script
COPY entrypoint.sh /app/entrypoint.sh

# Make startup script executable
RUN chmod +x /app/entrypoint.sh

# Set Python path
ENV PYTHONPATH=src

# Expose FastAPI port
EXPOSE 8080

# Start container
CMD ["/app/entrypoint.sh"]