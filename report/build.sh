#!/bin/bash
# Regenerate every table/figure in the reports from run directories (pure analysis, seconds).
#   report/build.sh            (pulls results from Futurama first if reachable)
set -euo pipefail
cd "$(dirname "$0")/.."
R=results/cluster
rsync -az pi34@planetexpress-login.cc.gatech.edu:dreamcache/results/ $R/ 2>/dev/null || echo "(offline: using local copy)"
latest() { ls -td $R/*-"$1" 2>/dev/null | head -1; }
PY=.venv/bin/python
mkdir -p report/data report/figures
run() { local name=$1; shift; local d; d=$(latest "$name") || true; [ -n "${d:-}" ] || { echo "skip $name"; return; }
  $PY analysis/summarize.py "$d" --out report/data/$name "$@" > /dev/null && echo "ok $name <- $d"; }
run val_dyn8_core --ref A_greedy --compare B_always
run val_dyn8_new --ref A_greedy --compare L_greedy_8b
run val_doorkey --ref A_greedy --compare B_always
run ablate_dyn8 --ref K3 --compare K3
run ablate_rollout_llm --ref roll_sim_bfs_matched --compare roll_sim_bfs_matched
run latency_dyn8 --ref L_greedy_8b --compare A_greedy_4b
run latency_single_h200 --ref L_greedy_8b --compare A_greedy_4b
run test_dyn8 --ref A_greedy --compare B_always
run specgraph_test_dyn8 --ref B0_no_spec --compare B1_ssd_sel_0.4
cb=$(ls -td $R/codebench/*-codebench 2>/dev/null | head -1 || true)
[ -n "$cb" ] && $PY analysis/codebench_summary.py "$cb" --out report/data/codebench > /dev/null && echo "ok codebench <- $cb"
for d in $R/analysis_val/detector_delta*; do [ -d "$d" ] && cp -r "$d" report/data/; done
[ -d $R/specgraph ] && cp $R/specgraph/predictors.json $R/specgraph/*.png report/data/ 2>/dev/null || true
find report/data -name "*.png" -exec cp {} report/figures/ \; 2>/dev/null || true
echo done
