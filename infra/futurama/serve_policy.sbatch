#!/bin/bash
# Live GPU inference server for the trained SmolVLA policies (see serve_policy.py).
#   sbatch -J cortex-serve serve_policy.sbatch <job_dir>
# Prints "[serve] READY port=<p>" in the log and writes <job_dir>/live/server.json {node, port};
# connect from a laptop with: ssh -N -L 8765:<node>:<port> futurama, then open http://127.0.0.1:8765/
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -w nvidia-h200
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
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" HF_LEROBOT_HOME="$SCR/lerobot" PATH="$SCR/bin:$HOME/.local/bin:$PATH"
PORT=$((20000 + SLURM_JOB_ID % 20000))  # unique per job: several jobs can share the node

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
cp prepare_dataset.py serve_policy.py "$SCR/"
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training,smolvla]==0.6.1"

for t in conveyor suture jenga; do  # held-out episodes for every task that has a checkpoint
  ls -d "$JOB_DIR/output/$t"/*_[0-9]* >/dev/null 2>&1 && python "$SCR/prepare_dataset.py" "$t" "$SCR/data-$t"
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
echo "{\"node\": \"$(hostname -s)\", \"port\": $PORT, \"job\": $SLURM_JOB_ID}" > "$JOB_DIR/live/server.json"
python "$SCR/serve_policy.py" "$JOB_DIR" "$SCR/data" "$PORT"
