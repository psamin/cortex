#!/bin/bash
# Experiment 2 offline stage (CPU only): build transition rows + SQLite execution graph from
# logged trajectories, then evaluate predictors (H1, H4).
#   sbatch scripts/specgraph_offline.sbatch TRAIN_RUNS... -- EVAL_RUNS...
# Env: SG_OUT (output dir), SG_ACTOR_METHODS="wf1=m1+m2;wf2=m3" (use + between methods: sbatch --export splits on commas)
#SBATCH -J sg-offline
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" 2>/dev/null || { REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"; cd "$REPO"; }
export HOME="${HOME/#\/nethome\//\/nethome-instruction\/}"
SCR="/dev/shm/${USER}-sg-${SLURM_JOB_ID}"; mkdir -p "$SCR"; trap 'rm -rf "$SCR"' EXIT TERM INT
export UV_CACHE_DIR="$SCR/uv" UV_PROJECT_ENVIRONMENT="$SCR/venv" UV_PYTHON_INSTALL_DIR="$SCR/py"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/uvbin" UV_NO_MODIFY_PATH=1 sh >/dev/null; export PATH="$SCR/uvbin:$PATH"; }
uv sync -q --python 3.12
OUT="${SG_OUT:-results/specgraph}"; mkdir -p "$OUT"
TRAIN=(); EVAL=(); dst=TRAIN
for x in "$@"; do if [ "$x" = "--" ]; then dst=EVAL; continue; fi; eval "$dst+=(\"$x\")"; done
rm -f "$OUT/rows_train.jsonl" "$OUT/rows_eval.jsonl" "$OUT/graph.sqlite"
# workflow label = actor identity; methods listed as WORKFLOW=method1,method2;...
IFS=';' read -ra SGROUPS <<< "${SG_ACTOR_METHODS:-actor8b=L_greedy_8b,LS_lossless_heur,LS_lossless_small,LS_lossless_both}"
for g in "${SGROUPS[@]}"; do
  wf="${g%%=*}"; ms="${g#*=}"; ms="${ms//,/ }"; ms="${ms//+/ }"
  uv run python -m dreamcache.specgraph.dataset "${TRAIN[@]}" --methods $ms --workflow "$wf" --out "$OUT/rows_train.jsonl" --db "$OUT/graph.sqlite"
  [ ${#EVAL[@]} -gt 0 ] && uv run python -m dreamcache.specgraph.dataset "${EVAL[@]}" --methods $ms --workflow "$wf" --out "$OUT/rows_eval.jsonl"
done
[ -s "$OUT/rows_eval.jsonl" ] && uv run python analysis/specgraph_predictors.py --train "$OUT/rows_train.jsonl" --eval "$OUT/rows_eval.jsonl" --out "$OUT"
echo SG_OFFLINE_DONE
