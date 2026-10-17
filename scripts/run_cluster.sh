#!/bin/bash
# From your laptop (GT VPN on): sync code to Futurama and submit a job.
#   scripts/run_cluster.sh configs/smoke.yaml [more configs]
set -euo pipefail
HOST="${DC_HOST:-pi34@planetexpress-login.cc.gatech.edu}"
DEST="${DC_DEST:-dreamcache}"
rsync -az --delete --exclude .venv --exclude results --exclude logs --exclude .git \
  --exclude __pycache__ --exclude .claude ./ "$HOST:$DEST/"
ssh "$HOST" "cd $DEST && mkdir -p logs && sbatch ${DC_SBATCH_ARGS:-} scripts/experiment.sbatch $*"
