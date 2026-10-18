#!/usr/bin/env bash
# Start the whole Cortex stack: backend (:8000), studio (:5173), MCP server (:3000).
#   scripts/dev.sh            local only
#   scripts/dev.sh --runner <url>  also run the MCP runner: dials out to a Manufact-hosted MCP
#                             server at <url> and serves its tool calls from this backend
#                             (no tunnel; needs the same CORTEX_MCP_KEY here and on the server)
#   scripts/dev.sh --mcp-tunnel    expose the local MCP server at a stable
#                             https://<name>.tunnel.mcp-use.run/mcp (mcp-use's own tunnel)
#   scripts/dev.sh --tunnel   expose the backend through a Cloudflare quick tunnel (URL changes
#                             on every restart; remote callers can only reach /api/tools with CORTEX_MCP_KEY)
set -euo pipefail
tunnel=0 mcp_tunnel=0 runner=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tunnel) tunnel=1 ;;
    --mcp-tunnel) mcp_tunnel=1 ;;
    --runner) runner="${2:?--runner needs the hosted MCP server URL}"; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
  shift
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/.cortex-data/logs"
mkdir -p "$LOGS"
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# --timeout-graceful-shutdown: an open studio tab holds an event stream, and without a cap uvicorn waits for it
# forever on stop, so the old backend lingers (still running jobs) next to the restarted one
(cd "$ROOT/backend" && uv sync -q && exec uv run uvicorn cortex.api.app:app --host 127.0.0.1 --port 8000 \
  --timeout-graceful-shutdown 3) \
  > "$LOGS/backend.log" 2>&1 & pids+=($!)
# always sync deps (fast when unchanged): a pulled lockfile change must not leave the studio broken
(cd "$ROOT/web" && pnpm install --frozen-lockfile --silent && exec pnpm dev --host 127.0.0.1) \
  > "$LOGS/web.log" 2>&1 & pids+=($!)
if [ -f "$ROOT/mcp/package.json" ]; then
  (cd "$ROOT/mcp" && npm install --no-audit --no-fund --silent && if [ "$mcp_tunnel" = 1 ]; then exec npm run dev -- --tunnel; else exec npm run dev; fi) > "$LOGS/mcp.log" 2>&1 & pids+=($!)
fi

for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8000/api/health >/dev/null && break; sleep 1; done
echo "backend  http://127.0.0.1:8000/api   (logs: $LOGS/backend.log)"
echo "studio   http://127.0.0.1:5173"
echo "mcp      http://localhost:3000/mcp   ->  claude mcp add --transport http cortex http://localhost:3000/mcp"

if [ "$mcp_tunnel" = 1 ]; then
  for _ in $(seq 1 60); do
    murl=$(grep -oE 'https://[a-z0-9-]+\.tunnel\.mcp-use\.run/mcp' "$LOGS/mcp.log" | head -1 || true)
    [ -n "$murl" ] && break; sleep 1
  done
  echo "mcp tun  ${murl:-<see $LOGS/mcp.log>}   (same URL across restarts; .mcp-use/state/tunnel.json keeps it)"
fi

if [ -n "$runner" ]; then
  (cd "$ROOT/backend" && exec uv run python -m cortex.mcp_runner --server "$runner") > "$LOGS/runner.log" 2>&1 & pids+=($!)
  sleep 3
  grep -E 'connected|rejected|disabled|not found|cannot reach' "$LOGS/runner.log" | tail -1 || true
  echo "runner   -> $runner   (logs: $LOGS/runner.log)"
fi

if [ "$tunnel" = 1 ]; then
  command -v cloudflared >/dev/null || { echo "install cloudflared: brew install cloudflared"; exit 1; }
  cloudflared tunnel --url http://127.0.0.1:8000 > "$LOGS/tunnel.log" 2>&1 & pids+=($!)
  for _ in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1 || true)
    [ -n "$url" ] && break; sleep 1
  done
  echo "tunnel   ${url:-<see $LOGS/tunnel.log>}/api   (set CORTEX_API_URL to this for the hosted MCP server)"
fi
wait
