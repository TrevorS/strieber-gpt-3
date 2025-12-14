# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

strieber-gpt-3 is a self-hosted AI inference stack running on DGX Spark Blackwell (128GB GPU). It includes:
- **LLM inference** via llama.cpp (gpt-oss-120b, 63GB model)
- **Chat UI** built with Svelte 5 + SvelteKit 2
- **Responses API** - Rust backend that orchestrates inference and tool execution
- **MCP tool servers** - Python services for web search, code execution, weather, web reading, and image generation

## Architecture

```
Chat UI (Svelte 5)  ──HTTP──►  responses-api (Rust)  ──HTTP──►  llama-server (llama.cpp)
         :9300                        :9150                           :9010
                                        │
                                        ├──►  MCP servers (Python)
                                        │       - web_search :9110
                                        │       - code_interpreter :9120
                                        │       - reader :9130
                                        │       - weather :9100
                                        │       - comfy_zimage :9141
                                        │       - lora_trainer :9145
                                        │
                                        └──►  llama-server-qwen-vl :9020 (vision)
```

**Key architectural decisions:**
- Server-side tool execution: `responses-api` invokes MCP servers directly (not client-side tool calling)
- Conversation chaining via `previous_response_id` (not simple message arrays)
- All services communicate via HTTP on Docker network `strieber-net`

## Project Validation

Run validation in order: **Format → Lint → Type Check → Test**

### Rust Backend (backend/responses-api/)
```bash
docker compose run --rm backend-dev cargo fmt
docker compose run --rm backend-dev cargo clippy -- -D warnings
docker compose run --rm backend-dev cargo test

# Run single test
docker compose run --rm backend-dev cargo test test_name
```

### Frontend (frontend/)
```bash
docker compose run --rm frontend-dev npm run format
docker compose run --rm frontend-dev npm run lint
docker compose run --rm frontend-dev npm run check
```

### Python MCP Servers (backend/tools/mcp_servers/)
```bash
cd backend/tools/mcp_servers

# Format and lint run locally (ruff not in container)
source .venv/bin/activate && ruff format .
source .venv/bin/activate && ruff check --fix .

# Tests run in Docker (has all deps like pyyaml)
docker exec strieber-mcp-lora-trainer pytest -v /app
docker exec strieber-mcp-lora-trainer pytest -v /app/tests/test_file.py::TestClass::test_method
```

## Docker Development

All runtime tools run inside containers. Dev tools (ruff for Python linting) can run locally via venv.

### Common Commands
```bash
make up                    # Start all services
make down                  # Stop all services
make logs                  # Follow logs
make status                # Container status
make health                # Check llama-server health
make test                  # Run responses-api integration tests
```

### Frontend Development
```bash
docker compose run --rm frontend-dev npm install
docker compose run --rm frontend-dev npm install <package>
docker compose run --rm frontend-dev npm run <script>
docker compose up chat-ui  # Run dev server
```

### E2E Visual Testing
```bash
docker compose run --rm playwright-test
```
Screenshots saved to `frontend/test-results/screenshots/`

### MCP Server Testing
Interactive debugging tool for testing MCP servers:
```bash
# Run from inside any MCP container
docker exec strieber-mcp-lora-trainer python /app/mcp_test.py servers
docker exec strieber-mcp-lora-trainer python /app/mcp_test.py ping <server>
docker exec strieber-mcp-lora-trainer python /app/mcp_test.py list <server>
docker exec strieber-mcp-lora-trainer python /app/mcp_test.py call <server> <tool> '{"arg": "value"}'
```

### Docker Naming Convention
- **Service names** (docker compose, network DNS): `mcp-lora-trainer`, `mcp-web-search`
- **Container names** (docker exec): `strieber-mcp-lora-trainer`, `strieber-mcp-web-search`

### Rebuilding Containers
After code changes, rebuild and restart specific services:
```bash
docker compose build --no-cache mcp-lora-trainer && docker compose up -d mcp-lora-trainer
```

## Tech Stack Reference

| Component | Technology | Key Files |
|-----------|------------|-----------|
| Frontend | Svelte 5, SvelteKit 2, Tailwind CSS 4, shadcn-svelte | `frontend/src/` |
| Backend API | Rust + Axum | `backend/responses-api/src/` |
| MCP Servers | Python + FastAPI | `backend/tools/mcp_servers/` |
| Container orchestration | Docker Compose | `compose.yml` |

### Frontend Patterns (Svelte 5)
- Use runes: `$state()`, `$derived()`, `$effect()`, `$props()`
- Class-based stores in `src/lib/stores/*.svelte.ts`
- shadcn-svelte components in `src/lib/components/ui/`

### Backend Patterns (Rust)
- Axum handlers in `src/server/handlers.rs`
- Tool execution loop in `src/execution/executor.rs`
- MCP client in `src/mcp/client.rs`

## Shell Compatibility (zsh)

The host shell is zsh. Avoid nested `$(...)` with complex quotes. Prefer:
```bash
jq -r '.id' /tmp/resp.json  # Good
```
Over:
```bash
PREV_ID=$(python3 -c "import json; print(json.load(open('/tmp/resp.json'))['id'])")  # Breaks in zsh
```

## Project Permissions
- **Project Type**: personal
- **Direct Commits Allowed**: yes
