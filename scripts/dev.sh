#!/usr/bin/env bash
# Start the whole Cortex stack: backend (:8000), studio (:5173), MCP server (:3000).
#   scripts/dev.sh            local only
#   scripts/dev.sh --tunnel   also expose the backend through a Cloudflare quick tunnel
#                             (for a Manufact-hosted MCP server; remote callers can only
#                             reach /api/tools with CORTEX_MCP_KEY)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/.cortex-data/logs"
mkdir -p "$LOGS"
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

(cd "$ROOT/backend" && uv sync -q && exec uv run uvicorn cortex.api.app:app --host 127.0.0.1 --port 8000) \
  > "$LOGS/backend.log" 2>&1 & pids+=($!)
(cd "$ROOT/web" && { [ -d node_modules ] || pnpm install --frozen-lockfile; } && exec pnpm dev --host 127.0.0.1) \
  > "$LOGS/web.log" 2>&1 & pids+=($!)
if [ -f "$ROOT/mcp/package.json" ]; then
  (cd "$ROOT/mcp" && { [ -d node_modules ] || npm ci; } && exec npm run dev) > "$LOGS/mcp.log" 2>&1 & pids+=($!)
fi

for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8000/api/health >/dev/null && break; sleep 1; done
echo "backend  http://127.0.0.1:8000/api   (logs: $LOGS/backend.log)"
echo "studio   http://127.0.0.1:5173"
echo "mcp      http://localhost:3000/mcp   ->  claude mcp add --transport http cortex http://localhost:3000/mcp"

if [[ "${1:-}" == "--tunnel" ]]; then
  command -v cloudflared >/dev/null || { echo "install cloudflared: brew install cloudflared"; exit 1; }
  cloudflared tunnel --url http://127.0.0.1:8000 > "$LOGS/tunnel.log" 2>&1 & pids+=($!)
  for _ in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1 || true)
    [ -n "$url" ] && break; sleep 1
  done
  echo "tunnel   ${url:-<see $LOGS/tunnel.log>}/api   (set CORTEX_API_URL to this for the hosted MCP server)"
fi
wait
