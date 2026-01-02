# ============================================================================
# Responses API Dockerfile
# Multi-stage build for OpenAI Responses API adapter
# ============================================================================

# Build stage
FROM rust:1.91-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies only
RUN cargo build --release && rm -rf src

# Copy actual source code and migrations
COPY src ./src
COPY migrations ./migrations

# Touch main.rs to trigger rebuild of actual code (not just dependencies)
RUN touch src/main.rs src/lib.rs

# Build the actual application
RUN cargo build --release

# Test stage - includes cargo for running tests
FROM builder AS test
COPY tests ./tests
CMD ["cargo", "test", "--", "--nocapture"]

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/target/release/responses-api /app/responses-api

# Create non-root user and data directory for persistent storage
RUN useradd -r -s /bin/false appuser && \
    mkdir -p /data/containers && \
    chown -R appuser:appuser /data
USER appuser

# Default environment variables
# Note: MODELS_CONFIG is required and must be set at runtime
# Example: MODELS_CONFIG={"models":[{"id":"gpt-4","url":"http://llama:8000"}]}
ENV PORT=8000 \
    HOST=0.0.0.0 \
    MAX_TOOL_ITERATIONS=10 \
    TIMEOUT_SECS=300 \
    RESPONSE_TTL_SECS=3600 \
    RUST_LOG=responses_api=info,tower_http=info

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the server
CMD ["/app/responses-api"]
