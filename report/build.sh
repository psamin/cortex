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
$PY analysis/summarize.py $(latest val_doorkey) $(latest val_doorkey_rerun) --out report/data/val_doorkey --ref A_greedy --compare B_always > /dev/null && echo "ok val_doorkey (+rerun)"
run ablate_dyn8 --ref K3 --compare K3
run ablate_rollout_llm --ref roll_sim_bfs_matched --compare roll_sim_bfs_matched
run latency_dyn8 --ref L_greedy_8b --compare A_greedy_4b
run latency_single_h200 --ref L_greedy_8b --compare A_greedy_4b
PY_RUNS_TEST="$(ls -td $R/*-test_dyn8 $R/*-test_dyn8_part2 $R/*-test_dyn8_part3 $R/*-test_dyn8_K_ext 2>/dev/null)"; [ -n "$PY_RUNS_TEST" ] && $PY analysis/summarize.py $PY_RUNS_TEST --out report/data/test_dyn8 --ref A_greedy --compare B_always > /dev/null && echo "ok test_dyn8 (3 parts)"
run specgraph_test_dyn8 --ref B0_no_spec --compare B1_ssd_sel_0.4
cb=$(ls -td $R/codebench/*-codebench 2>/dev/null | head -1 || true)
[ -n "$cb" ] && $PY analysis/codebench_summary.py "$cb" --out report/data/codebench > /dev/null && echo "ok codebench <- $cb"
for d in $R/analysis_val/detector_delta*; do [ -d "$d" ] && cp -r "$d" report/data/; done
[ -d $R/specgraph ] && cp $R/specgraph/predictors.json $R/specgraph/*.png report/data/ 2>/dev/null || true
for f in report/data/*/*.png; do [ -f "$f" ] && cp "$f" "report/figures/$(basename "$(dirname "$f")")__$(basename "$f")"; done
echo done
for x in specgraph_test_dyn8:B0_no_spec:specgraph_latency specgraph_b8_fixed:B1_ssd_sel_0.4:specgraph_b8_fixed \
         latency_single_h200:L_greedy_8b:latency_single_paired latency_slow_actor_h200:L_greedy_32b:latency_slow_actor_h200; do
  IFS=: read -r name ref out <<< "$x"; d=$(latest "$name") || true
  [ -n "${d:-}" ] && $PY analysis/specgraph_latency.py "$d" --ref "$ref" --out report/data/$out > /dev/null 2>&1 && echo "ok $out"
done
[ -d $R/specgraph_eval ] && cp $R/specgraph_eval/predictors.json $R/specgraph_eval/predictor_vs_history.png report/data/ 2>/dev/null || true
d=$(latest ablate_parallel) || true; [ -n "${d:-}" ] && $PY analysis/summarize.py "$d" --out report/data/ablate_parallel --ref seq_K3 --compare seq_K3 > /dev/null && echo "ok ablate_parallel"
for f in report/data/*/*.png; do [ -f "$f" ] && cp "$f" "report/figures/$(basename "$(dirname "$f")")__$(basename "$f")"; done
cp $R/bench-*.json $R/probe-*.json report/data/ 2>/dev/null || true

$PY analysis/summarize.py $R/*-test_dyn8 $R/*-test_dyn8_part3 $R/*-test_dyn8_ext --out report/data/test_dyn8_300 --ref E_random_0.25 --compare B_always > /dev/null && echo "ok test_dyn8_300"
$PY analysis/summarize.py $R/*-test_dyn16 $R/*-test_dyn16_ext --out report/data/test_dyn16 --ref E_random_0.25 --compare B_always > /dev/null && echo "ok test_dyn16"
$PY analysis/summarize.py $R/*-test_dyn8_think $R/*-test_dyn8_think_ext --out report/data/test_dyn8_think --ref H_think_4b --compare H_think_4b > /dev/null && echo "ok test_dyn8_think"
$PY analysis/summarize.py $R/*-test_doorkey_a $R/*-test_doorkey_b --out report/data/test_doorkey --ref A_greedy --compare B_always > /dev/null && echo "ok test_doorkey"
$PY analysis/specgraph_latency.py $(latest specgraph_b7_pair) $(latest specgraph_b7_only) --ref B1_ssd_K2_llm_ranked --out report/data/specgraph_b7 > /dev/null 2>&1 && echo "ok specgraph_b7"
$PY analysis/summarize.py $(latest specgraph_b7_pair) $(latest specgraph_b7_only) $(latest specgraph_b7) --out report/data/specgraph_b7_success --ref B1_ssd_K2_llm_ranked --compare B1_ssd_K3 > /dev/null && echo "ok specgraph_b7_success"
for f in report/data/*/*.png; do [ -f "$f" ] && cp "$f" "report/figures/$(basename "$(dirname "$f")")__$(basename "$f")"; done
