Episodes with runner errors (excluded, listed in episodes.jsonl): 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy | 5 | 0.00 [0.00, 0.00] | 1.00 | 16.4 | 0.00 | 16.4 | 0.0 | 0.00 | 0.0 | 0.00/0.00 |
| Dynamic-Obstacles-8x8-v0 | B_always | 5 | 1.00 [1.00, 1.00] | 0.00 | 16.0 | 1.00 | 16.0 | 0.0 | 0.16 | 0.2 | 0.01/0.02 |
| Dynamic-Obstacles-8x8-v0 | C_selective_0.4 | 5 | 1.00 [1.00, 1.00] | 0.00 | 20.0 | 0.71 | 80.0 | 0.0 | 0.16 | 0.2 | 0.01/0.02 |
| Dynamic-Obstacles-8x8-v0 | I_planner_sim | 5 | 1.00 [1.00, 1.00] | 0.00 | 18.2 | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 0.00/0.00 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy | B_always | -1.000 [-1.000, -1.000] p=0.000 | -0.164 [-0.189, -0.139] p=0.000 | -0.222 [-0.259, -0.186] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B_always | A_greedy | +1.000 [+1.000, +1.000] p=0.000 | +0.164 [+0.139, +0.189] p=0.000 | +0.222 [+0.186, +0.259] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | C_selective_0.4 | A_greedy | +1.000 [+1.000, +1.000] p=0.000 | +0.161 [+0.088, +0.292] p=0.006 | +0.223 [+0.119, +0.411] p=0.006 |
| Dynamic-Obstacles-8x8-v0 | C_selective_0.4 | B_always | +0.000 [+0.000, +0.000] p=1.000 | -0.004 [-0.071, +0.107] p=0.846 | +0.001 [-0.094, +0.159] p=0.957 |
| Dynamic-Obstacles-8x8-v0 | I_planner_sim | A_greedy | +1.000 [+1.000, +1.000] p=0.000 | +0.000 [+0.000, +0.000] p=1.000 | -0.003 [-0.006, +0.000] p=0.127 |
| Dynamic-Obstacles-8x8-v0 | I_planner_sim | B_always | +0.000 [+0.000, +0.000] p=1.000 | -0.164 [-0.189, -0.139] p=0.000 | -0.225 [-0.260, -0.190] p=0.000 |
