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
