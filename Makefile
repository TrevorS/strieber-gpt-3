# ============================================================================
# strieber-gpt-3 Makefile
# ============================================================================
# Build and manage llama.cpp container with CLI utilities

.PHONY: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Core Commands
# ============================================================================

build: ## Build llama.cpp container (includes all CLI utilities)
	docker compose -f compose.llama.yml build

up: ## Start llama-server container
	docker compose -f compose.llama.yml up -d

down: ## Stop llama-server container
	docker compose -f compose.llama.yml down

restart: ## Restart llama-server container
	docker compose -f compose.llama.yml restart

logs: ## Show llama-server logs (follow mode)
	docker compose -f compose.llama.yml logs -f

shell: ## Open bash shell in container
	docker compose -f compose.llama.yml exec llama-server /bin/bash

status: ## Check container status
	docker compose -f compose.llama.yml ps

# ============================================================================
# Health & Testing
# ============================================================================

health: ## Check llama-server health endpoint
	@curl -sf http://localhost:8000/health && echo "✓ llama-server is healthy" || echo "✗ llama-server not responding"

health-verbose: ## Check health with verbose output
	curl -v http://localhost:8000/health

# ============================================================================
# Llama.cpp CLI Utilities
# ============================================================================
# All utilities are available in the container. Use these helpers to run them.
# For more options on each tool, append -- --help to see all flags

llama-server-exec: ## Execute llama-server directly (use for quick testing)
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-server

llama-bench: ## Run llama-bench performance benchmarks
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-bench

llama-cli: ## Interactive CLI interface for model inference
	docker compose -f compose.llama.yml exec -it llama-server /app/bin/llama-cli

llama-quantize: ## Quantize GGUF models to different formats
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-quantize

llama-perplexity: ## Calculate model perplexity on input text
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-perplexity

llama-embedding: ## Generate embeddings for text input
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-embedding

llama-tokenize: ## Tokenize text using model's tokenizer
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-tokenize

llama-imatrix: ## Generate importance matrix for model optimization
	docker compose -f compose.llama.yml exec llama-server /app/bin/llama-imatrix

# Generic utility runner: make llama-util CMD="--help"
llama-util: ## Generic utility runner (usage: make llama-util CMD="/app/bin/llama-bench --help")
	docker compose -f compose.llama.yml exec llama-server /bin/bash -c "$(CMD)"

# ============================================================================
# Benchmarking
# ============================================================================

bench-sequential: ## Run sequential benchmark (prefill/generation at various context depths)
	@mkdir -p benchmarks
	@TIMESTAMP=$$(date +%Y%m%d-%H%M%S); \
	echo "Sequential benchmark started at $$TIMESTAMP"; \
	docker compose -f compose.llama.yml exec llama-server \
	  /app/bin/llama-bench \
	  -m /models/gpt-oss-20b-Q4_K_M.gguf \
	  -fa 1 -d 0,4096,8192,16384,32768 -p 2048 -n 32 -ub 2048 -mmp 0 -o jsonl \
	  2>&1 | tee benchmarks/bench-sequential-$$TIMESTAMP.log; \
	echo "✓ Sequential benchmark completed at $$TIMESTAMP"

bench-parallel: ## Run parallel benchmark (128k context, batch sizes 1-16)
	@mkdir -p benchmarks
	@TIMESTAMP=$$(date +%Y%m%d-%H%M%S); \
	echo "Parallel benchmark started at $$TIMESTAMP"; \
	docker compose -f compose.llama.yml exec llama-server \
	  /app/bin/llama-batched-bench \
	  -m /models/gpt-oss-20b-Q4_K_M.gguf \
	  -fa 1 -c 131072 -ub 2048 -npp 4096,8192 -ntg 32 -npl 1,2,4,8,16 --no-mmap -o jsonl \
	  2>&1 | tee benchmarks/bench-parallel-$$TIMESTAMP.log; \
	echo "✓ Parallel benchmark completed at $$TIMESTAMP"

bench-extreme: ## Run extreme stress test (300k context, batch sizes 1-32)
	@mkdir -p benchmarks
	@TIMESTAMP=$$(date +%Y%m%d-%H%M%S); \
	echo "Extreme benchmark started at $$TIMESTAMP"; \
	docker compose -f compose.llama.yml exec llama-server \
	  /app/bin/llama-batched-bench \
	  -m /models/gpt-oss-20b-Q4_K_M.gguf \
	  -fa 1 -c 300000 -ub 2048 -npp 4096,8192 -ntg 32 -npl 1,2,4,8,16,32 --no-mmap -o jsonl \
	  2>&1 | tee benchmarks/bench-extreme-$$TIMESTAMP.log; \
	echo "✓ Extreme benchmark completed at $$TIMESTAMP"

bench-all: ## Run all benchmarks sequentially (sequential -> parallel -> extreme)
	@echo "Running all benchmarks..."
	make bench-sequential
	@echo ""
	make bench-parallel
	@echo ""
	make bench-extreme
	@echo "✓ All benchmarks completed!"
	@ls -lah benchmarks/

bench-list: ## List all benchmark results
	@echo "Benchmark results:"
	@ls -lah benchmarks/ 2>/dev/null || echo "No benchmarks run yet. Run 'make bench-sequential' or 'make bench-parallel'."

# ============================================================================
# Container Utilities
# ============================================================================

list-binaries: ## List all available llama.cpp binaries in container
	docker compose -f compose.llama.yml exec llama-server ls -lah /app/bin/

copy-binary: ## Copy a binary from container to host (usage: make copy-binary BIN=llama-bench)
	docker compose -f compose.llama.yml exec llama-server cat /app/bin/$(BIN) > ./$(BIN)
	chmod +x ./$(BIN)
	@echo "Copied $(BIN) to current directory and made executable"

# ============================================================================
# Model Management
# ============================================================================

download-20b: ## Download 20B model (~16GB)
	@mkdir -p models
	cd models && curl -L -O https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf

download-120b: ## Download 120B model (~63GB, 3 parts)
	@mkdir -p models
	cd models && \
		curl -L -O https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00001-of-00003.gguf && \
		curl -L -O https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00002-of-00003.gguf && \
		curl -L -O https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00003-of-00003.gguf

models-info: ## Show information about downloaded models
	@echo "Models in ./models/:"
	@ls -lh models/ 2>/dev/null || echo "No models directory found. Run 'make download-20b' or 'make download-120b' first."

# ============================================================================
# Development & Debugging
# ============================================================================

docker-inspect: ## Inspect the llama-server image
	docker image inspect strieber-llama-server:latest | jq '.'

container-stats: ## Show real-time container resource usage
	docker stats strieber-llama-server

prune: ## Remove unused Docker images and containers
	docker system prune -f

clean: ## Clean up everything (stop containers, remove images)
	docker compose -f compose.llama.yml down
	docker image rm strieber-llama-server:latest 2>/dev/null || true
	docker system prune -f
