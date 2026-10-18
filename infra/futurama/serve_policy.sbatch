#!/bin/bash
# Live GPU inference server for the trained SmolVLA policies (see serve_policy.py).
#   sbatch -J cortex-serve [--gres=gpu:h200:1] serve_policy.sbatch <job_dir>
# Prints "[live] <step>" progress lines and "[serve] READY port=<p>" in the log, and writes
# <job_dir>/live/server.json {node, port}. The studio's GPU tab submits this and tunnels to it
# (backend/cortex/services/live_policy_service.py).
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail

JOB_DIR="$1"
[ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
cd "$JOB_DIR"
SCR="/dev/shm/cortex-${SLURM_JOB_ID}"
mkdir -p "$SCR" "$JOB_DIR/live"
trap 'rm -rf "$SCR"' EXIT INT TERM
SIM_DIR="${SIM_DIR:-$(dirname "$JOB_DIR")/cortex-sim}"  # sim-trained policies (sim_task.sbatch) and cortex code
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" HF_LEROBOT_HOME="$SCR/lerobot" PATH="$SCR/bin:$HOME/.local/bin:$PATH" \
       MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$SIM_DIR/code"
PORT=$((20000 + SLURM_JOB_ID % 20000))  # unique per job: several jobs can share the node

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
echo "[live] Installing LeRobot on $(hostname -s) ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
cp prepare_dataset.py serve_policy.py serve_sim.py "$SCR/"
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training,smolvla]==0.6.1" "mujoco>=3.3.7"

for t in conveyor suture jenga; do  # held-out episodes for every task that has a checkpoint
  ls -d "$JOB_DIR/output/$t"/*_[0-9]* >/dev/null 2>&1 || continue
  echo "[live] Preparing held-out episodes: $t"
  python "$SCR/prepare_dataset.py" "$t" "$SCR/data-$t"
done
mkdir -p "$SCR/data"
for d in "$SCR"/data-*; do cp -r "$d"/* "$SCR/data/" 2>/dev/null || true; done
python - "$SCR" <<'EOF'
import json, sys
from pathlib import Path
scr = Path(sys.argv[1]); merged = {}
for f in sorted(scr.glob("data-*/split.json")):
    merged.update(json.loads(f.read_text()))
(scr / "data" / "split.json").write_text(json.dumps(merged))
EOF
echo "[live] Loading the policies onto the GPU"
echo "{\"node\": \"$(hostname -s)\", \"port\": $PORT, \"job\": $SLURM_JOB_ID}" > "$JOB_DIR/live/server.json"
python "$SCR/serve_policy.py" "$JOB_DIR" "$SCR/data" "$PORT" "$SIM_DIR"
