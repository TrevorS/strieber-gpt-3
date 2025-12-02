# Build stage
FROM node:22-alpine AS builder
WORKDIR /app

# Build arg for API URL (baked into client-side code)
# Must include /v1 as the OpenAI client appends resource paths to this base
ARG VITE_RESPONSES_API_URL=http://responses-api:8000/v1

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Build application with API URL
ENV VITE_RESPONSES_API_URL=${VITE_RESPONSES_API_URL}
RUN npm run build

# Prune dev dependencies
RUN npm prune --production

# Runtime stage
FROM node:22-alpine
WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy built application
COPY --from=builder /app/build build/
COPY --from=builder /app/node_modules node_modules/
COPY package.json .

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup
USER appuser

# Environment
ENV NODE_ENV=production
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["node", "build"]
