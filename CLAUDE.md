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
```

### Frontend (frontend/)
```bash
docker compose run --rm frontend-dev npm run format
docker compose run --rm frontend-dev npm run lint
docker compose run --rm frontend-dev npm run check
```

### Python MCP Servers (backend/tools/mcp_servers/)
```bash
source .venv/bin/activate && ruff format .
source .venv/bin/activate && ruff check --fix .
source .venv/bin/activate && pytest -v
```

## Docker Development

All tools run inside containers. Never install Node, Rust, or Python dependencies on the host.

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
