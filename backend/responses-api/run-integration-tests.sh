#!/usr/bin/env bash
# ============================================================================
# Integration Test Runner Script
# ============================================================================
# Runs integration tests against the actual llama.cpp server and MCP services.
#
# Usage:
#   ./run-integration-tests.sh              # Run all tests
#   ./run-integration-tests.sh llama        # Run llama.cpp tests only
#   ./run-integration-tests.sh responses    # Run responses API tests only
#   ./run-integration-tests.sh tools        # Run tool calling tests only
#
# Prerequisites:
#   The main services must be running:
#     docker compose up -d llama-server responses-api mcp-weather mcp-code-interpreter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if main services are running
check_services() {
    echo_info "Checking if required services are running..."

    if ! docker ps | grep -q strieber-llama-server; then
        echo_warn "llama-server is not running. Start it with: docker compose up -d llama-server"
        return 1
    fi

    if ! docker ps | grep -q strieber-responses-api; then
        echo_warn "responses-api is not running. Start it with: docker compose up -d responses-api"
        return 1
    fi

    echo_info "Services are running."
    return 0
}

# Wait for services to be healthy
wait_for_health() {
    echo_info "Waiting for services to be healthy..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:9010/health > /dev/null 2>&1 && \
           curl -sf http://localhost:9150/health > /dev/null 2>&1; then
            echo_info "All services are healthy!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 2
        ((attempt++))
    done

    echo_error "Services did not become healthy in time"
    return 1
}

# Run tests based on argument
run_tests() {
    local test_filter="${1:-all}"

    echo_info "Building test container..."
    docker compose -f docker-compose.test.yml build responses-api-test

    case "$test_filter" in
        llama)
            echo_info "Running llama.cpp integration tests..."
            docker compose -f docker-compose.test.yml run --rm responses-api-test \
                cargo test --release --test llama_integration -- --test-threads=1 --nocapture
            ;;
        responses)
            echo_info "Running responses API integration tests..."
            docker compose -f docker-compose.test.yml run --rm responses-api-test \
                cargo test --release --test responses_api_integration -- --test-threads=1 --nocapture
            ;;
        tools)
            echo_info "Running tool calling integration tests..."
            docker compose -f docker-compose.test.yml run --rm responses-api-test \
                cargo test --release --test tool_calling_integration -- --test-threads=1 --nocapture
            ;;
        all)
            echo_info "Running all integration tests..."
            docker compose -f docker-compose.test.yml run --rm responses-api-test
            ;;
        *)
            echo_error "Unknown test filter: $test_filter"
            echo "Usage: $0 [llama|responses|tools|all]"
            exit 1
            ;;
    esac
}

# Clean up test containers
cleanup() {
    echo_info "Cleaning up test containers..."
    docker compose -f docker-compose.test.yml down --remove-orphans 2>/dev/null || true
}

# Main
main() {
    local test_filter="${1:-all}"

    # Trap cleanup on exit
    trap cleanup EXIT

    # Check services
    if ! check_services; then
        echo_error "Required services are not running. Please start them first."
        echo_info "Run: cd $(dirname "$SCRIPT_DIR") && docker compose up -d llama-server responses-api"
        exit 1
    fi

    # Wait for health
    if ! wait_for_health; then
        exit 1
    fi

    # Run tests
    run_tests "$test_filter"
}

main "$@"
