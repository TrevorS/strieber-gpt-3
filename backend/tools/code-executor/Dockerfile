FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 executor
USER executor

# Default command (will be overridden by docker run)
CMD ["python3"]
