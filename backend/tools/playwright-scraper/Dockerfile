# ABOUTME: Playwright web scraper for local HTML fetching
# Provides REST API for fetching and rendering web pages with JavaScript support
# Used by local_reader MCP server for privacy-preserving web content extraction

FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scraper_server.py .

# Expose port for API
EXPOSE 8000

# Run the scraper server
CMD ["python", "scraper_server.py"]
