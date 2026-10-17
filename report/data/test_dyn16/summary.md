Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-16x16-v0 | A_greedy | 100 | 0.03 [0.00, 0.07] | 0.95 | 52.0 | 0.00 | 52.0 | 275.2 | 0.00 | 256.5 | 5.43/6.96 |
| Dynamic-Obstacles-16x16-v0 | B_always | 100 | 1.00 [1.00, 1.00] | 0.00 | 32.7 | 1.00 | 32.7 | 156.8 | 0.77 | 116.9 | 3.90/8.06 |
| Dynamic-Obstacles-16x16-v0 | C_risk_only_0.4 | 100 | 0.77 [0.68, 0.85] | 0.00 | 147.7 | 0.14 | 147.7 | 845.4 | 0.53 | 702.8 | 5.52/6.65 |
| Dynamic-Obstacles-16x16-v0 | E_random_0.25 | 100 | 0.26 [0.18, 0.35] | 0.74 | 56.4 | 0.23 | 56.4 | 295.1 | 0.37 | 229.7 | 5.25/6.37 |
| Dynamic-Obstacles-16x16-v0 | I_planner_obs | 100 | 0.60 [0.50, 0.69] | 0.00 | 146.3 | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 0.00/0.00 |
| Dynamic-Obstacles-16x16-v0 | I_planner_sim | 100 | 1.00 [1.00, 1.00] | 0.00 | 32.8 | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 0.00/0.00 |
| Dynamic-Obstacles-16x16-v0 | L_greedy_8b | 100 | 0.16 [0.09, 0.23] | 0.80 | 60.9 | 0.00 | 60.9 | 281.5 | 0.00 | 256.9 | 3.96/8.81 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-16x16-v0 | A_greedy | E_random_0.25 | -0.230 [-0.320, -0.140] p=0.000 | -20.332 [-65.793, +26.041] p=0.392 | +26.844 [-16.117, +69.912] p=0.235 |
| Dynamic-Obstacles-16x16-v0 | A_greedy | B_always | -0.970 [-1.000, -0.930] p=0.000 | +117.639 [+74.061, +162.726] p=0.000 | +139.634 [+98.183, +182.164] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | B_always | E_random_0.25 | +0.740 [+0.650, +0.820] p=0.000 | -137.970 [-177.823, -98.014] p=0.000 | -112.790 [-149.918, -75.218] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | C_risk_only_0.4 | E_random_0.25 | +0.510 [+0.400, +0.620] p=0.000 | +550.447 [+489.140, +610.887] p=0.000 | +473.086 [+412.628, +532.998] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | C_risk_only_0.4 | B_always | -0.230 [-0.310, -0.150] p=0.000 | +688.417 [+644.027, +732.633] p=0.000 | +585.876 [+540.449, +630.819] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | E_random_0.25 | B_always | -0.740 [-0.820, -0.650] p=0.000 | +137.970 [+98.014, +177.823] p=0.000 | +112.790 [+75.218, +149.918] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_obs | E_random_0.25 | +0.340 [+0.210, +0.460] p=0.000 | -295.496 [-331.198, -259.669] p=0.000 | -229.657 [-262.201, -197.313] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_obs | B_always | -0.400 [-0.500, -0.310] p=0.000 | -157.526 [-169.259, -145.837] p=0.000 | -116.867 [-129.801, -104.058] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_sim | E_random_0.25 | +0.740 [+0.650, +0.820] p=0.000 | -295.496 [-331.198, -259.669] p=0.000 | -229.683 [-262.229, -197.340] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_sim | B_always | +0.000 [+0.000, +0.000] p=1.000 | -157.526 [-169.259, -145.837] p=0.000 | -116.893 [-129.827, -104.084] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | L_greedy_8b | E_random_0.25 | -0.100 [-0.210, +0.010] p=0.095 | -13.950 [-73.276, +48.828] p=0.657 | +27.180 [-28.154, +85.341] p=0.351 |
| Dynamic-Obstacles-16x16-v0 | L_greedy_8b | B_always | -0.840 [-0.910, -0.770] p=0.000 | +124.020 [+70.403, +181.035] p=0.000 | +139.970 [+90.525, +193.059] p=0.000 |
