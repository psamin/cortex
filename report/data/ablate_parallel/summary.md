Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | parallel4 | 20 | 1.00 [1.00, 1.00] | 0.00 | 18.1 | 1.00 | 18.1 | 43.3 | 0.38 | 0.9 | 0.05/0.06 |
| Dynamic-Obstacles-8x8-v0 | seq_K3 | 20 | 1.00 [1.00, 1.00] | 0.00 | 18.1 | 1.00 | 18.1 | 43.3 | 0.34 | 43.1 | 2.44/2.68 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | parallel4 | seq_K3 | +0.000 [+0.000, +0.000] p=1.000 | +0.035 [+0.030, +0.040] p=0.000 | -42.198 [-49.624, -35.326] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | parallel4 | seq_K3 | +0.000 [+0.000, +0.000] p=1.000 | +0.035 [+0.030, +0.040] p=0.000 | -42.198 [-49.624, -35.326] p=0.000 |
