#!/bin/bash
# Cortex ACT training job for the GT Futurama cluster. Submitted by
# backend/cortex/services/training_service.py after in-app approval:
#   sbatch --export=ALL,STEPS=20000,BATCH=8,SAVE=10000,REPO_ID=cortex/x train_act.sbatch <job_dir>
# <job_dir> holds dataset/ (LeRobotDataset v3, rsynced from Cortex). Results go
# straight to <job_dir>/output in home so a walltime kill keeps finished checkpoints.
#SBATCH -J cortex-act
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail

JOB_DIR="$1"
# /nethome/<user> is a self-referencing symlink on compute nodes; remap to the autofs path.
[ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
cd "$JOB_DIR"

SCR="/dev/shm/cortex-${SLURM_JOB_ID}"
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT INT TERM
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" \
       TRITON_CACHE_DIR="$SCR/triton" MUJOCO_GL=egl PATH="$SCR/bin:$HOME/.local/bin:$PATH"

echo "[cortex] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
# Copy inputs into job scratch so later rsyncs cannot change a running job.
cp -r dataset "$SCR/dataset"
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training]==0.6.1"
echo "[cortex] lerobot $(uv pip show lerobot | grep Version)"

rm -rf output
lerobot-train \
  --dataset.repo_id="$REPO_ID" --dataset.root="$SCR/dataset" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps="$STEPS" --batch_size="$BATCH" --num_workers=6 \
  --save_freq="$SAVE" --log_freq=100 --seed=1000 \
  --output_dir="$JOB_DIR/output" --job_name=cortex_act --wandb.enable=false
echo "[cortex] TRAINING_DONE"
