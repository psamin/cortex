#!/bin/bash
# DreamCache experiment job: starts a private Ollama inside the allocation, then runs
# one or more experiment configs. Usage:
#   sbatch scripts/experiment.sbatch configs/smoke.yaml [configs/more.yaml ...]
# Env knobs: DC_MODELS (models to pull), DC_WORKERS, DC_BENCH=1 (run Ollama benchmark first)
#SBATCH -J dreamcache
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$REPO" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
cd "$REPO" 2>/dev/null || { REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"; cd "$REPO"; }
export HOME="${HOME/#\/nethome\//\/nethome-instruction\/}"
mkdir -p logs results
export DC_LOG_DIR="$REPO/logs"
source scripts/ollama_env.sh
export UV_PYTHON_INSTALL_DIR="$SCR/py"
# Run from a private snapshot of the code so later syncs cannot change a running job.
mkdir -p "$SCR/code" && cp -r "$REPO/src" "$REPO/scripts" "$REPO/configs" "$REPO/analysis" "$REPO/pyproject.toml" "$REPO/uv.lock" "$SCR/code/"
[ -f "$REPO/GIT_COMMIT" ] && cp "$REPO/GIT_COMMIT" "$SCR/code/"
cd "$SCR/code"
ln -s "$REPO/results" results
ln -s "$REPO/logs" logs
uv sync -q --python 3.12 --extra dev
export OLLAMA_HOST_URL="http://$OLLAMA_HOST"
export DC_LLM_CACHE="$SCR/llm_cache.sqlite"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "logs/gpu-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true; cleanup' EXIT TERM INT
if [ "${DC_BENCH:-0}" = "1" ]; then
  for m in ${DC_MODELS:-qwen3:4b}; do
    uv run python scripts/bench_ollama.py --model "$m" --out "results/bench-${m//[:\/]/_}-${SLURM_JOB_ID}.json"
  done
fi
if [ "${DC_PROBE:-0}" = "1" ]; then
  uv run python scripts/prompt_probe.py --models ${DC_MODELS:-qwen3:4b} --prompts ${DC_PROBE_PROMPTS:-v1 v2 v3} --out "results/probe-${SLURM_JOB_ID}.json"
fi
if [ "${DC_CODEBENCH:-0}" = "1" ]; then
  uv run python -m dreamcache.codebench.run --out results/codebench --model "${DC_CB_MODEL:-qwen3:4b}" --repeats "${DC_CB_REPEATS:-1}"
fi
for cfg in "$@"; do
  uv run python -m dreamcache.experiments.runner "$cfg" --out results --workers "${DC_WORKERS:-8}"
done
