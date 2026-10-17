Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | roll_fast_llm | 10 | 0.60 [0.30, 0.90] | 0.40 | 27.9 | 1.00 | 239.1 | 629.0 | 0.36 | 588.4 | 21.23/29.88 |
| Dynamic-Obstacles-8x8-v0 | roll_sim_bfs_matched | 10 | 1.00 [1.00, 1.00] | 0.00 | 24.8 | 1.00 | 24.8 | 74.2 | 0.07 | 64.7 | 2.95/3.34 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | roll_fast_llm | roll_sim_bfs_matched | -0.400 [-0.700, -0.100] p=0.019 | +555.131 [+386.059, +725.573] p=0.000 | +523.718 [+353.927, +696.312] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | roll_fast_llm | roll_sim_bfs_matched | -0.400 [-0.700, -0.100] p=0.019 | +555.131 [+386.059, +725.573] p=0.000 | +523.718 [+353.927, +696.312] p=0.000 |
