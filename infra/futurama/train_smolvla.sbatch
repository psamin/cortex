#!/bin/bash
# Fine-tune SmolVLA (~450M) on real SO-100/SO-101 demos prepared by prepare_dataset.py.
#   sbatch --export=ALL,TASK=jenga,STEPS=10000,BATCH=32 -J cortex-jenga train_smolvla.sbatch <job_dir>
#   TASK=suture runs two stages: STEPS on "surgical", then STEPS2 on "needle" (camera names mapped).
# Checkpoints go to RAM scratch (home has little free space); a copier keeps only the newest
# checkpoint's weights in <job_dir>/output/<TASK>/<stage>_<step>, so a walltime kill keeps it.
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -w nvidia-h200
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail

JOB_DIR="$1"
[ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
cd "$JOB_DIR"
OUT="$JOB_DIR/output/$TASK"
mkdir -p "$OUT"

SCR="/dev/shm/cortex-${SLURM_JOB_ID}"
mkdir -p "$SCR"
COPIER=""
cleanup() { [ -n "$COPIER" ] && kill "$COPIER" 2>/dev/null; rm -rf "$SCR"; }
trap cleanup EXIT INT TERM
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" HF_LEROBOT_HOME="$SCR/lerobot" TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" \
       TRITON_CACHE_DIR="$SCR/triton" PATH="$SCR/bin:$HOME/.local/bin:$PATH"

echo "[cortex] task=$TASK node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
cp prepare_dataset.py "$SCR/"
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training,smolvla]==0.6.1"
echo "[cortex] lerobot $(uv pip show lerobot | grep Version)"

python "$SCR/prepare_dataset.py" "$TASK" "$SCR/data"
cp "$SCR/data/split.json" "$OUT/split.json"

# keep only the newest checkpoint's weights in home (the optimizer state stays in RAM scratch)
( while true; do
    for ck in $(ls -d "$SCR"/out-*/checkpoints/*/pretrained_model 2>/dev/null | sort); do
      [ -f "$ck/model.safetensors" ] || continue
      run=$(basename "$(dirname "$(dirname "$(dirname "$ck")")")"); step=$(basename "$(dirname "$ck")")
      dest="$OUT/${run#out-}_$step"
      [ -d "$dest" ] && continue
      sleep 20; cp -r "$ck" "$dest.tmp" && mv "$dest.tmp" "$dest"
      for old in "$OUT"/*_[0-9]*; do [ "$old" != "$dest" ] && [ -d "$old" ] && rm -rf "$old"; done
      echo "[cortex] kept $dest"
    done
    sleep 30
  done ) &
COPIER=$!

train() {  # stage, policy path, steps, extra args...
  local stage=$1 policy=$2 steps=$3; shift 3
  local eps; eps=$(python -c "import json; print(json.load(open('$SCR/data/split.json'))['$stage']['train'])" | tr -d ' ')
  lerobot-train --policy.path="$policy" --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="local/$stage" --dataset.root="$SCR/data/$stage" --dataset.episodes="$eps" \
    --steps="$steps" --batch_size="$BATCH" --num_workers=7 --save_freq="$((steps / 2))" --log_freq=100 --seed=1000 \
    --output_dir="$SCR/out-$stage" --job_name="cortex_${TASK}_$stage" --wandb.enable=false "$@"
}

# smolvla_base expects observation.images.camera1..3; a dataset may provide a subset of them
cams() { local m="{" i=1 k; for k in "$@"; do m+="\"observation.images.$k\": \"observation.images.camera$i\", "; i=$((i+1)); done; echo "${m%, }}"; }
case "$TASK" in
  jenga)    train jenga lerobot/smolvla_base "$STEPS" --rename_map="$(cams main secondary_0 secondary_1)" ;;
  conveyor) train conveyor lerobot/smolvla_base "$STEPS" --rename_map="$(cams top wrist)" ;;
  suture)
    train surgical lerobot/smolvla_base "$STEPS" --rename_map="$(cams front side)"
    STAGE1=$(ls -d "$SCR"/out-surgical/checkpoints/*/pretrained_model | sort | tail -1)
    train needle "$STAGE1" "${STEPS2:-1500}" --rename_map="$(cams external wrist)" ;;
esac
sleep 90  # let the copier keep the final checkpoint
echo "[cortex] TRAINING_DONE"
