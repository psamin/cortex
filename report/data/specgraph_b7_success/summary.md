Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 97

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | B1_ssd_K2_llm_ranked | 50 | 0.98 [0.94, 1.00] | 0.02 | 40.9 | 0.90 | 163.8 | 295.2 | 0.33 | 153.9 | 3.73/4.28 |
| Dynamic-Obstacles-8x8-v0 | B1_ssd_K3 | 50 | 1.00 [1.00, 1.00] | 0.00 | 19.6 | 0.91 | 78.6 | 152.6 | 0.22 | 77.1 | 3.89/4.41 |
| Dynamic-Obstacles-8x8-v0 | B7_ssd_K2_graph_ranked | 50 | 1.00 [1.00, 1.00] | 0.00 | 19.5 | 0.90 | 77.8 | 139.4 | 0.15 | 72.4 | 3.72/4.29 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | B1_ssd_K2_llm_ranked | B1_ssd_K3 | -0.020 [-0.060, +0.000] p=0.631 | +142.640 [+108.764, +179.023] p=0.000 | +76.806 [+59.233, +95.626] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B1_ssd_K3 | B1_ssd_K2_llm_ranked | +0.020 [+0.000, +0.060] p=0.631 | -142.640 [-179.023, -108.764] p=0.000 | -76.806 [-95.626, -59.233] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B7_ssd_K2_graph_ranked | B1_ssd_K2_llm_ranked | +0.020 [+0.000, +0.060] p=0.631 | -155.972 [-191.963, -123.115] p=0.000 | -81.541 [-100.015, -64.569] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B7_ssd_K2_graph_ranked | B1_ssd_K3 | +0.000 [+0.000, +0.000] p=1.000 | -13.332 [-28.222, +0.788] p=0.074 | -4.734 [-12.451, +2.491] p=0.214 |
