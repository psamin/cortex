#!/usr/bin/env bash
# Run Cortex's own planner model on Futurama and connect the local backend to it.
#   scripts/vlm_futurama.sh up      submit the vLLM job, wait for it, open the tunnel, update .env
#   scripts/vlm_futurama.sh status  job state, endpoint and tunnel health
#   scripts/vlm_futurama.sh down    close the tunnel and cancel the job (release the GPU)
# Needs the GT VPN and the `futurama` SSH alias. Restart the backend after `up` so it
# picks up PLANNER_BASE_URL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${FUTURAMA_SSH_HOST:-futurama}"
REMOTE_DIR="cortex/serve"
LOCAL_PORT="${VLM_LOCAL_PORT:-8001}"
MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
STATE="$ROOT/.cortex-data/vlm_job"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST")

set_env() {  # set_env KEY VALUE in .env (created if missing, kept private)
  touch "$ROOT/.env"; chmod 600 "$ROOT/.env"
  grep -v "^$1=" "$ROOT/.env" > "$ROOT/.env.tmp" || true
  echo "$1=$2" >> "$ROOT/.env.tmp"; mv "$ROOT/.env.tmp" "$ROOT/.env"
}

tunnel_pid() { pgrep -f "ssh .*-L ${LOCAL_PORT}:" || true; }

case "${1:-}" in
up)
  "${SSH[@]}" hostname >/dev/null || { echo "cannot reach $HOST (is the GT VPN connected?)"; exit 1; }
  KEY="$(grep -E '^PLANNER_API_KEY=.+' "$ROOT/.env" 2>/dev/null | cut -d= -f2- || true)"
  [ -n "$KEY" ] || KEY="$(openssl rand -hex 16)"
  "${SSH[@]}" "mkdir -p ~/$REMOTE_DIR"
  rsync -az "$ROOT/infra/futurama/serve_vlm.sbatch" "$HOST:~/$REMOTE_DIR/serve_vlm.sbatch"
  JOB="$("${SSH[@]}" "cd ~/$REMOTE_DIR && rm -f endpoint.json && sbatch --parsable \
        --export=ALL,VLM_API_KEY=$KEY,MODEL=$MODEL serve_vlm.sbatch ~/$REMOTE_DIR")"
  mkdir -p "$(dirname "$STATE")"; echo "$JOB" > "$STATE"
  echo "submitted job $JOB ($MODEL); waiting for the server (queue + install + model load)..."
  for _ in $(seq 1 240); do
    EP="$("${SSH[@]}" "cat ~/$REMOTE_DIR/endpoint.json 2>/dev/null" || true)"
    [ -n "$EP" ] && break
    S="$("${SSH[@]}" "squeue -h -j $JOB -o %T" || true)"
    [ -z "$S" ] && { echo "job $JOB left the queue; log: ~/$REMOTE_DIR/slurm-vlm-$JOB.out"; exit 1; }
    printf "\r  job %s: %s   " "$JOB" "$S"; sleep 15
  done
  echo
  NODE="$(echo "$EP" | sed -E 's/.*"node":"([^"]+)".*/\1/')"
  PORT="$(echo "$EP" | sed -E 's/.*"port":([0-9]+).*/\1/')"
  [ -n "$(tunnel_pid)" ] && kill "$(tunnel_pid)" || true
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "${LOCAL_PORT}:${NODE}:${PORT}" "$HOST"
  curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null \
    || { echo "tunnel open but the server does not answer"; exit 1; }
  set_env PLANNER_BASE_URL "http://127.0.0.1:${LOCAL_PORT}/v1"
  set_env PLANNER_API_KEY "$KEY"
  echo "ready: $MODEL on $NODE:$PORT -> http://127.0.0.1:${LOCAL_PORT}/v1 (job $JOB). Restart the backend."
  ;;
status)
  JOB="$(cat "$STATE" 2>/dev/null || true)"
  [ -n "$JOB" ] && echo "job $JOB: $("${SSH[@]}" "squeue -h -j $JOB -o '%T %M %N'" || echo unknown)"
  echo "tunnel: $([ -n "$(tunnel_pid)" ] && echo "open on :$LOCAL_PORT" || echo closed)"
  KEY="$(grep -E '^PLANNER_API_KEY=' "$ROOT/.env" | cut -d= -f2- || true)"
  curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:${LOCAL_PORT}/v1/models" | head -c 300 \
    || echo "endpoint not answering"
  echo
  ;;
down)
  [ -n "$(tunnel_pid)" ] && kill "$(tunnel_pid)" && echo "tunnel closed"
  JOB="$(cat "$STATE" 2>/dev/null || true)"
  [ -n "$JOB" ] && "${SSH[@]}" "scancel $JOB" && echo "cancelled job $JOB" && rm -f "$STATE"
  echo "Remove PLANNER_BASE_URL from .env (or leave it; the planner reports it unreachable) and restart the backend."
  ;;
*) echo "usage: $0 up|status|down"; exit 2 ;;
esac
