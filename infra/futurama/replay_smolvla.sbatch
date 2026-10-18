#!/bin/bash
# Replay a trained SmolVLA policy on its held-out real demos and render videos + metrics.
#   sbatch --export=ALL,TASK=jenga -J cortex-replay-jenga replay_smolvla.sbatch <job_dir>
# Uses the newest checkpoint in <job_dir>/output/<TASK>/; writes <job_dir>/replay/<TASK>/.
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -w nvidia-h200
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail

JOB_DIR="$1"
[ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
cd "$JOB_DIR"
CKPT=$(ls -d "$JOB_DIR/output/$TASK"/*_[0-9]* | sort | tail -1)
OUT="$JOB_DIR/replay/$TASK"
mkdir -p "$OUT"

SCR="/dev/shm/cortex-${SLURM_JOB_ID}"
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT INT TERM
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" HF_LEROBOT_HOME="$SCR/lerobot" PATH="$SCR/bin:$HOME/.local/bin:$PATH"

echo "[cortex] replay task=$TASK checkpoint=$CKPT node=$(hostname)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
cp prepare_dataset.py replay_policy.py "$SCR/"
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training,smolvla]==0.6.1" matplotlib "imageio[ffmpeg]"

python "$SCR/prepare_dataset.py" "$TASK" "$SCR/data"
python "$SCR/replay_policy.py" "$TASK" "$CKPT" "$SCR/data" "$OUT"
echo "[cortex] REPLAY_DONE"
