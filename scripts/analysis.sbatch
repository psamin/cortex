#!/bin/bash
# CPU-only analysis on the cluster: oracle criticality labels (privileged Q_ref rollouts)
# + detector evaluation. Usage: sbatch scripts/analysis.sbatch RUN_DIR [RUN_DIR...]
# Env: AN_OUT, AN_METHODS
#SBATCH -J dc-analysis
#SBATCH -p general
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" 2>/dev/null || { REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"; cd "$REPO"; }
export HOME="${HOME/#\/nethome\//\/nethome-instruction\/}"
SCR="/dev/shm/${USER}-an-${SLURM_JOB_ID}"; mkdir -p "$SCR"; trap 'rm -rf "$SCR"' EXIT TERM INT
export UV_CACHE_DIR="$SCR/uv" UV_PROJECT_ENVIRONMENT="$SCR/venv" UV_PYTHON_INSTALL_DIR="$SCR/py"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/uvbin" UV_NO_MODIFY_PATH=1 sh >/dev/null; export PATH="$SCR/uvbin:$PATH"; }
uv sync -q --python 3.12
OUT="${AN_OUT:-results/analysis}"; mkdir -p "$OUT"
uv run python -m dreamcache.experiments.label_states "$@" --out "$OUT/labels.jsonl" \
  --methods ${AN_METHODS:-A_greedy_logged C_sel_0.2 C_sel_0.4 C_sel_0.6 C_sel_0.8} --n 16 --workers 32
for d in 0.1 0.2 0.4; do
  uv run python analysis/detector_eval.py "$OUT/labels.jsonl" --out "$OUT/detector_delta$d" --delta $d
done
echo ANALYSIS_DONE
