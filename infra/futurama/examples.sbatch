#!/bin/bash
# Render a few full example demo videos for a sim task without touching its dataset.
#   TASK=screwdriver N=2 sbatch -J sim-examples examples.sbatch <job_dir>
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
JOB_DIR="$1"; [ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
cd "$JOB_DIR"
SCR="/dev/shm/cortex-${SLURM_JOB_ID}"; mkdir -p "$SCR"; trap 'rm -rf "$SCR"' EXIT INT TERM
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$JOB_DIR/code" PATH="$SCR/bin:$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
uv venv -q --python 3.12 "$SCR/venv"; source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training,smolvla]==0.6.1" "mujoco>=3.3.7"
mkdir -p "$JOB_DIR/$TASK/examples"
python -m cortex.sim.tasks.run demos "$TASK" "$SCR/ds" --episodes "${N:-2}" --examples "$JOB_DIR/$TASK/examples" \
  --n-examples "${N:-2}" --noise-deg 0
echo "[sim] EXAMPLES_DONE"
