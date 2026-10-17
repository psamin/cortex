# Source inside a Slurm job: installs Ollama into job-scoped /dev/shm, starts it on
# 127.0.0.1, pulls $DC_MODELS. Everything here is disposable and removed on exit.
set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a Slurm job}"
export SCR="/dev/shm/${USER}-dc-${SLURM_JOB_ID}"
mkdir -p "$SCR"
cleanup() {
  [ -n "${OLLAMA_PID:-}" ] && kill "$OLLAMA_PID" 2>/dev/null && wait "$OLLAMA_PID" 2>/dev/null || true
  rm -rf "$SCR"
}
trap cleanup EXIT TERM INT

# Unique port per job: several jobs can share a node, and a second server on the same
# port silently fails to bind (requests would then hit another job's server).
export DC_OLLAMA_PORT="${DC_OLLAMA_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export OLLAMA_HOST="127.0.0.1:${DC_OLLAMA_PORT}"
export OLLAMA_MODELS="$SCR/models"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
export UV_CACHE_DIR="$SCR/uv-cache" UV_PROJECT_ENVIRONMENT="$SCR/venv"
export HF_HOME="$SCR/hf"

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.12.6}"
cd "$SCR"
if ! curl -fsSL -o ollama.tgz "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz"; then
  echo "tgz download failed; trying tar.zst" >&2
  curl -fsSL -o ollama.tar.zst "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
  mkdir -p ollama && tar --use-compress-program=unzstd -xf ollama.tar.zst -C ollama
else
  mkdir -p ollama && tar -xzf ollama.tgz -C ollama
fi
export PATH="$SCR/ollama/bin:$PATH"
ollama --version || true

ollama serve > "${DC_LOG_DIR:-.}/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 60); do curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 1; done
kill -0 "$OLLAMA_PID" 2>/dev/null || { echo "FATAL: ollama serve exited (port $DC_OLLAMA_PORT in use?)" >&2; exit 1; }
for m in ${DC_MODELS:-qwen3:4b}; do ollama pull "$m"; done
cd - >/dev/null

# Python env via uv (installed to job scratch if missing).
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/uvbin" UV_NO_MODIFY_PATH=1 sh >/dev/null
  export PATH="$SCR/uvbin:$PATH"
fi
