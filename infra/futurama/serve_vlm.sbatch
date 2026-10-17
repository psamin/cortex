#!/bin/bash
# Serve Cortex's own planner model (Qwen3-VL) with vLLM's OpenAI-compatible API on one
# Futurama GPU. Submitted by scripts/vlm_futurama.sh:
#   sbatch --export=ALL,VLM_API_KEY=<key>,MODEL=Qwen/Qwen3-VL-8B-Instruct serve_vlm.sbatch <serve_dir>
# Writes <serve_dir>/endpoint.json = {node, port, job, model} once the server is up; the
# Mac reaches it through `ssh -L <local>:<node>:<port> futurama`.
#SBATCH -J cortex-vlm
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o slurm-vlm-%j.out
set -euo pipefail

SERVE_DIR="$1"
[ -d "$SERVE_DIR" ] || SERVE_DIR="${SERVE_DIR/#\/nethome\//\/nethome-instruction\/}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
# Several jobs can share a node: derive a unique port from the job id.
PORT=$((20000 + SLURM_JOB_ID % 20000))

SCR="/dev/shm/cortex-vlm-${SLURM_JOB_ID}"
mkdir -p "$SCR"
trap 'rm -f "$SERVE_DIR/endpoint.json"; rm -rf "$SCR"' EXIT INT TERM
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" VLLM_CACHE_ROOT="$SCR/vllm" TRITON_CACHE_DIR="$SCR/triton" \
       TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" PATH="$SCR/bin:$HOME/.local/bin:$PATH"

echo "[cortex-vlm] node=$(hostname) port=$PORT gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "vllm==0.31.0"

vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" --api-key "$VLM_API_KEY" \
  --max-model-len 32768 --limit-mm-per-prompt '{"image": 8}' \
  --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.90 &
SERVER=$!
for _ in $(seq 1 180); do  # wait up to 30 min for download + load
  if curl -sf -H "Authorization: Bearer $VLM_API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    printf '{"node":"%s","port":%d,"job":%d,"model":"%s"}\n' "$(hostname)" "$PORT" "$SLURM_JOB_ID" "$MODEL" \
      > "$SERVE_DIR/endpoint.json"
    echo "[cortex-vlm] READY"
    break
  fi
  kill -0 "$SERVER" 2>/dev/null || { echo "[cortex-vlm] server exited during startup"; exit 1; }
  sleep 10
done
wait "$SERVER"
