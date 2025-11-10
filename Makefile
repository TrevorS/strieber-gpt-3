# strieber-gpt-3 Makefile

.PHONY: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Core Commands

build: ## Build all Docker images
	docker compose -f compose.yml build

up: ## Start all services
	docker compose -f compose.yml up -d

down: ## Stop all services
	docker compose -f compose.yml down --remove-orphans

restart: ## Restart all services
	docker compose -f compose.yml restart

logs: ## Show logs from all services (follow mode)
	docker compose -f compose.yml logs -f

shell: ## Open bash shell in llama-server container
	docker compose -f compose.yml exec llama-server /bin/bash

status: ## Check container status
	docker compose -f compose.yml ps

# Health & Management

health: ## Check llama-server health endpoint
	@curl -sf http://localhost:8000/health && echo "✓ llama-server is healthy" || echo "✗ llama-server not responding"

llama-util: ## Run llama.cpp utilities (usage: make llama-util CMD="/app/bin/llama-bench --help")
	docker compose -f compose.yml exec llama-server /bin/bash -c "$(CMD)"

# LLM Console Client

llm-build: ## Build llm-client Docker image
	docker build -f Dockerfile.llm-client -t llm-client:latest .

llm-setup: ## Initialize llm API key (one-time setup)
	docker run --rm -it \
		--network strieber-gpt-3_strieber-net \
		-v ~/.config/io.datasette.llm:/root/.config/io.datasette.llm \
		llm-client:latest \
		llm keys set local

llm-chat: ## Start interactive console chat with llm
	docker run --rm -it \
		--network strieber-gpt-3_strieber-net \
		-v ~/.config/io.datasette.llm:/root/.config/io.datasette.llm \
		llm-client:latest \
		llm chat -m gpt-oss-120b

# Docker Cleanup

prune: ## Remove unused Docker images and containers
	docker system prune -f

clean: ## Clean up everything (stop containers, remove images, prune volumes)
	docker compose -f compose.yml down --remove-orphans --volumes
	docker image rm strieber-llama-server:latest 2>/dev/null || true
	docker system prune -f

nuke: ## Force-remove all strieber containers and images
	docker ps -a --filter "name=strieber-*" -q | xargs -r docker rm -f
	docker images --filter "reference=strieber-*" -q | xargs -r docker rmi -f
	docker system prune -f

# ==========================================================================
# ComfyUI Management Commands
# ==========================================================================

comfyui-build: ## Build ComfyUI Docker image
	docker compose -f compose.yml build comfyui

comfyui-logs: ## Show ComfyUI logs (follow mode)
	docker compose -f compose.yml logs -f comfyui

comfyui-shell: ## Open bash shell in ComfyUI container
	docker compose -f compose.yml exec comfyui /bin/bash

comfyui-restart: ## Restart ComfyUI service
	docker compose -f compose.yml restart comfyui

comfyui-health: ## Check ComfyUI health status
	@docker inspect strieber-comfyui --format='{{.State.Health.Status}}' 2>/dev/null || echo "not running"
	@curl -sf http://localhost:9040 && echo "✓ ComfyUI web UI responding" || echo "✗ ComfyUI not responding"
