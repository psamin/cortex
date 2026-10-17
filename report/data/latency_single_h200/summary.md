Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy_4b | 12 | 0.17 [0.00, 0.42] | 0.83 | 23.6 | 0.00 | 23.6 | 16.7 | 0.00 | 16.8 | 0.69/0.77 |
| Dynamic-Obstacles-8x8-v0 | K2_lossless_draft_L2 | 12 | 0.50 [0.25, 0.75] | 0.50 | 23.8 | 0.77 | 49.2 | 36.9 | 0.00 | 23.5 | 1.28/1.54 |
| Dynamic-Obstacles-8x8-v0 | K2_lossless_draft_L4 | 12 | 0.25 [0.00, 0.50] | 0.75 | 27.1 | 0.71 | 72.8 | 89.2 | 0.00 | 33.7 | 1.67/2.33 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_both | 12 | 0.50 [0.25, 0.75] | 0.50 | 27.0 | 0.00 | 77.1 | 95.0 | 0.00 | 33.8 | 1.19/1.70 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_heur | 12 | 0.42 [0.17, 0.67] | 0.58 | 26.3 | 0.00 | 39.2 | 43.2 | 0.00 | 19.3 | 1.01/1.28 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_small | 12 | 0.08 [0.00, 0.25] | 0.92 | 20.1 | 0.00 | 51.2 | 57.7 | 0.00 | 24.0 | 1.15/1.43 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | 12 | 0.33 [0.08, 0.58] | 0.67 | 25.2 | 0.00 | 25.2 | 17.7 | 0.00 | 17.7 | 0.68/0.88 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy_4b | L_greedy_8b | -0.167 [-0.500, +0.250] p=0.531 | -0.942 [-10.751, +9.073] p=0.856 | -0.944 [-10.776, +9.095] p=0.856 |
| Dynamic-Obstacles-8x8-v0 | K2_lossless_draft_L2 | L_greedy_8b | +0.167 [-0.250, +0.583] p=0.533 | +19.251 [+3.752, +37.613] p=0.025 | +5.794 [-4.351, +17.578] p=0.306 |
| Dynamic-Obstacles-8x8-v0 | K2_lossless_draft_L2 | A_greedy_4b | +0.333 [+0.000, +0.667] p=0.104 | +20.194 [+0.682, +41.808] p=0.054 | +6.738 [-7.112, +21.773] p=0.366 |
| Dynamic-Obstacles-8x8-v0 | K2_lossless_draft_L4 | L_greedy_8b | -0.083 [-0.417, +0.250] p=0.818 | +71.515 [+43.580, +101.702] p=0.000 | +15.961 [+6.769, +26.490] p=0.002 |
| Dynamic-Obstacles-8x8-v0 | K2_lossless_draft_L4 | A_greedy_4b | +0.083 [-0.250, +0.417] p=0.820 | +72.458 [+39.761, +105.176] p=0.000 | +16.905 [+3.103, +30.033] p=0.013 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_both | L_greedy_8b | +0.167 [-0.167, +0.500] p=0.422 | +77.311 [+49.750, +107.033] p=0.000 | +16.138 [+4.954, +28.253] p=0.005 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_both | A_greedy_4b | +0.333 [+0.000, +0.667] p=0.103 | +78.253 [+49.789, +106.148] p=0.000 | +17.082 [+5.726, +28.154] p=0.002 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_heur | L_greedy_8b | +0.083 [+0.000, +0.250] p=0.617 | +25.516 [+11.185, +44.551] p=0.006 | +1.630 [-6.501, +11.145] p=0.716 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_heur | A_greedy_4b | +0.250 [-0.167, +0.667] p=0.318 | +26.458 [+9.255, +44.703] p=0.003 | +2.574 [-7.156, +11.346] p=0.588 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_small | L_greedy_8b | -0.250 [-0.583, +0.083] p=0.222 | +40.060 [+16.963, +64.320] p=0.000 | +6.338 [-5.177, +17.565] p=0.275 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_small | A_greedy_4b | -0.083 [-0.333, +0.167] p=0.762 | +41.003 [+15.663, +66.825] p=0.001 | +7.282 [-5.127, +20.021] p=0.267 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | A_greedy_4b | +0.167 [-0.250, +0.500] p=0.531 | +0.942 [-9.073, +10.751] p=0.856 | +0.944 [-9.095, +10.776] p=0.856 |
