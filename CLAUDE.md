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