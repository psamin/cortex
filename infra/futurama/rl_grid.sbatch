#!/bin/bash
# Massively parallel RL video: CEM on the pick-and-place skill, 50 rollouts at once, then the grid video.
#SBATCH -p general
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
[ -r "$HOME/.bashrc" ] || export HOME="${HOME/#\/nethome\//\/nethome-instruction\/}"  # /nethome loops on some compute nodes
D="$HOME/cortex-rlvid"
[ -d "$D" ] || D="${D/#\/nethome\//\/nethome-instruction\/}"
CODE=/nethome-instruction/pi34/cortex-sim/code
SCR="/dev/shm/rlvid-${SLURM_JOB_ID}"
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT INT TERM
cp -r "$CODE" "$SCR/code"
export UV_CACHE_DIR="$SCR/uv" UV_PYTHON_INSTALL_DIR="$SCR/python" HF_HOME="$SCR/hf" CORTEX_CACHE="$SCR/cache" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
       PYTHONPATH="$SCR/code" PATH="$SCR/bin:$HOME/.local/bin:/nethome-instruction/pi34/.local/bin:$PATH"
export GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[rl] node=$(hostname -s) gpu=$GPU_NAME cores=$(nproc)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset]==0.6.1" "mujoco>=3.3.7" av pillow pydantic
python -c "import cortex.rl.proposals, cortex.rl.skill, cortex.sim.tasks.pick_place; print(\"[rl] imports ok\")"
OUT="$D/out"
rm -rf "$OUT"; mkdir -p "$OUT"
python "$D/rl_grid.py" "$OUT" "${ITERS:-6}" "${POP:-50}"
python "$D/render_video.py" "$OUT"
rm -f "$OUT"/iter*.npy
echo "[rl] ALL_DONE"
