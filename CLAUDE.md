## Docker Development

Tools and commands should run inside Docker containers, not on the host system.

### Frontend Development

Use the `frontend-dev` service for all npm/node commands to avoid permission issues:

```bash
# Install dependencies
docker compose run --rm frontend-dev npm install

# Add a package
docker compose run --rm frontend-dev npm install <package>

# Run dev server (use chat-ui service instead for this)
docker compose up chat-ui

# Run any npm script
docker compose run --rm frontend-dev npm run <script>
```

The `frontend-dev` service:
- Runs as UID/GID 1000 (matches host user)
- Mounts `./frontend` to `/app`
- Uses `node:22-alpine` image
- Part of `dev` profile (won't start with regular `docker compose up`)

### Other Services

For other project tools, check if there's a corresponding container in compose.yml before running commands on the host.

### E2E Visual Testing

A `playwright-e2e` skill exists for visual testing workflow. After UI changes:

```bash
docker compose run --rm playwright-test
```

Then read screenshots from `frontend/test-results/screenshots/` to verify the UI.

### Shell Compatibility (zsh)

The host shell is zsh, which parses commands differently than bash. To avoid parse errors:

- **Avoid nested `$(...)`** with complex quotes or parentheses inside
- **Run commands in separate steps** instead of chaining with command substitution
- **Use simple tools**: prefer `jq` over `python3 -c` for JSON parsing
- **Write to temp files** instead of inline parsing when extracting values

Bad (causes zsh parse errors):
```bash
PREV_ID=$(python3 -c "import json; print(json.load(open('/tmp/resp.json'))['id'])")
```

Good:
```bash
python3 -c "import json; print(json.load(open('/tmp/resp.json'))['id'])" > /tmp/prev_id.txt
# or just use jq
jq -r '.id' /tmp/resp.json
```
