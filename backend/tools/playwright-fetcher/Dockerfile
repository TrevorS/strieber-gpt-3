# Playwright HTML Fetcher Service
# Provides headless browser automation for JavaScript-rendered pages

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only for efficiency)
RUN playwright install --with-deps chromium

# Copy application code
COPY fetcher.py .

# Create non-root user for security
RUN useradd -m -u 1000 playwright-user && \
    chown -R playwright-user:playwright-user /app

USER playwright-user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Expose port
EXPOSE 8005

# Run the fetcher service
CMD ["python", "fetcher.py"]
