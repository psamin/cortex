#!/bin/bash
# Sim-only policies for the hand-built tasks (issue #161), on one GPU (EGL rendering + training).
#   MODE=demos TASK=jenga EPISODES=200 sbatch -J sim-demos sim_task.sbatch <job_dir>
#   (faster: PARTS=4 PART=0..3 on four GPUs, then MODE=merge with --dependency=afterok:<the four>)
#       -> <job_dir>/<task>/dataset (oracle demos, LeRobotDataset v3) and <task>/examples/*.mp4
#   MODE=train TASK=jenga VARIANT=act STEPS=30000 sbatch -J sim-act sim_task.sbatch <job_dir>
#       -> <job_dir>/<task>/runs/<variant>/ checkpoints, then eval/<step>/ closed-loop metrics + videos
# <job_dir>/code is an rsynced copy of backend/cortex. Training runs in /dev/shm (optimizer state is
# large); only the policy weights are copied to home, then evaluated from there.
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail

JOB_DIR="$1"
[ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
cd "$JOB_DIR"
SCR="/dev/shm/cortex-${SLURM_JOB_ID}"
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT INT TERM
export UV_CACHE_DIR="$SCR/uv" HF_HOME="$SCR/hf" TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" TRITON_CACHE_DIR="$SCR/triton" \
       MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$JOB_DIR/code" PATH="$SCR/bin:$HOME/.local/bin:$PATH"
T="$JOB_DIR/$TASK"
echo "[sim] node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) mode=$MODE task=$TASK"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv venv -q --python 3.12 "$SCR/venv"
source "$SCR/venv/bin/activate"
uv pip install -q "lerobot[dataset,training,smolvla]==0.6.1" "mujoco>=3.3.7"

if [ "$MODE" = demos ]; then  # SHARDS recorders in parallel, then one dataset
  # One GPU renders ~8 recorders well; more on the same GPU is slower. For more throughput, split the work
  # across GPUs: PARTS jobs with PART=0..PARTS-1 each record into $T/parts/, then one MODE=merge job.
  N="${SHARDS:-8}"; P="${PARTS:-1}"; K="${PART:-0}"; pids=(); roots=()
  if [ "$P" = 1 ]; then rm -rf "$T/dataset" "$T/examples"; OUTS="$SCR"; else OUTS="$T/parts"; fi
  [ "$K" = 0 ] && rm -rf "$T/examples"
  mkdir -p "$T" "$OUTS"
  for i in $(seq 0 $((N - 1))); do
    g=$((K * N + i))
    rm -rf "$OUTS/shard$g"
    ex=(); [ "$g" = 0 ] && [ "$P" = 1 ] && ex=(--examples "$T/examples")  # multi-part: use examples.sbatch
    python -m cortex.sim.tasks.run demos "$TASK" "$OUTS/shard$g" --episodes "${EPISODES:-200}" \
      --size "${SIZE:-240x320}" --shard "$g" --shards $((N * P)) --noise-deg "${NOISE_DEG:-2.0}" "${ex[@]}" \
      > "$OUTS/shard$g.log" 2>&1 &
    pids+=($!); roots+=("$OUTS/shard$g")
  done
  fail=0
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { fail=1; echo "[sim] shard $i failed:"; tail -5 "${roots[$i]}.log"; }; done
  [ "$fail" = 0 ] || exit 1
  if [ "$P" = 1 ]; then
    python -m cortex.sim.tasks.run merge "$TASK" "$T/dataset" "${roots[@]}"
    echo "[sim] DEMOS_DONE"
  else
    echo "[sim] PART_DONE $K/$P"
  fi
  exit 0
fi

if [ "$MODE" = merge ]; then  # after all PARTS: one dataset from $T/parts/shard*
  rm -rf "$T/dataset"
  python -m cortex.sim.tasks.run merge "$TASK" "$T/dataset" $(ls -d "$T"/parts/shard* | grep -v "\.log$" | sort -V)
  rm -rf "$T/parts"
  echo "[sim] DEMOS_DONE"
  exit 0
fi

# Training variants: same demos, different policy recipes, so we can keep what works.
OUT="$T/runs/$VARIANT"
CAMS=()
case "$VARIANT" in
  act)           ARGS=(--policy.type=act) ;;
  act_ensemble)  ARGS=(--policy.type=act --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01) ;;
  diffusion)     ARGS=(--policy.type=diffusion --policy.crop_shape="[216,288]") ;;
  smolvla)       ARGS=(--policy.path=lerobot/smolvla_base
                       --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}')
                 CAMS=(--cameras camera1=front,camera2=wrist) ;;
  *) echo "unknown VARIANT $VARIANT"; exit 2 ;;
esac
cp -r "$T/dataset" "$SCR/dataset"
rm -rf "$OUT"
mkdir -p "$OUT/checkpoints"
lerobot-train --dataset.repo_id="cortex/sim_$TASK" --dataset.root="$SCR/dataset" \
  "${ARGS[@]}" --policy.device=cuda --policy.push_to_hub=false \
  --steps="$STEPS" --batch_size="${BATCH:-32}" --num_workers=7 --save_freq="$((STEPS / 2))" --log_freq=200 --seed=1000 \
  --output_dir="$SCR/out" --job_name="sim_${TASK}_$VARIANT" --wandb.enable=false
for ck in "$SCR"/out/checkpoints/[0-9]*; do
  mkdir -p "$OUT/checkpoints/$(basename "$ck")"
  cp -r "$ck/pretrained_model" "$OUT/checkpoints/$(basename "$ck")/"
done
echo "[sim] TRAINING_DONE"

for ck in "$OUT"/checkpoints/[0-9]*/pretrained_model; do
  step=$(basename "$(dirname "$ck")")
  python -m cortex.sim.tasks.run evaluate "$TASK" "$ck" "$OUT/eval/$step" --episodes "${EVAL_EPISODES:-50}" \
    --size "${SIZE:-240x320}" "${CAMS[@]}" | tail -1
done
echo "[sim] EVAL_DONE"
