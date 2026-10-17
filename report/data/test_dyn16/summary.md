Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-16x16-v0 | A_greedy | 50 | 0.04 [0.00, 0.10] | 0.96 | 51.7 | 0.00 | 51.7 | 258.2 | 0.00 | 243.3 | 5.24/6.75 |
| Dynamic-Obstacles-16x16-v0 | B_always | 50 | 1.00 [1.00, 1.00] | 0.00 | 33.1 | 1.00 | 33.1 | 148.9 | 0.70 | 114.9 | 3.99/7.18 |
| Dynamic-Obstacles-16x16-v0 | C_risk_only_0.4 | 50 | 0.82 [0.72, 0.92] | 0.00 | 141.7 | 0.15 | 141.7 | 809.8 | 0.47 | 686.8 | 5.47/6.81 |
| Dynamic-Obstacles-16x16-v0 | E_random_0.25 | 50 | 0.24 [0.12, 0.36] | 0.76 | 49.9 | 0.21 | 49.9 | 260.6 | 0.29 | 206.3 | 5.17/6.54 |
| Dynamic-Obstacles-16x16-v0 | I_planner_obs | 50 | 0.58 [0.44, 0.72] | 0.00 | 142.9 | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 0.00/0.00 |
| Dynamic-Obstacles-16x16-v0 | I_planner_sim | 50 | 1.00 [1.00, 1.00] | 0.00 | 33.4 | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 0.00/0.00 |
| Dynamic-Obstacles-16x16-v0 | L_greedy_8b | 50 | 0.20 [0.10, 0.32] | 0.78 | 60.0 | 0.00 | 60.0 | 172.6 | 0.00 | 155.9 | 2.66/4.64 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-16x16-v0 | A_greedy | B_always | -0.960 [-1.000, -0.900] p=0.000 | +108.583 [+53.415, +165.062] p=0.000 | +128.354 [+76.230, +181.821] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | B_always | A_greedy | +0.960 [+0.900, +1.000] p=0.000 | -108.583 [-165.062, -53.415] p=0.000 | -128.354 [-181.821, -76.230] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | C_risk_only_0.4 | A_greedy | +0.780 [+0.640, +0.900] p=0.000 | +552.107 [+467.575, +632.753] p=0.000 | +443.495 [+358.575, +526.505] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | C_risk_only_0.4 | B_always | -0.180 [-0.300, -0.080] p=0.002 | +660.690 [+599.505, +720.738] p=0.000 | +571.848 [+509.905, +632.833] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | E_random_0.25 | A_greedy | +0.200 [+0.080, +0.320] p=0.002 | +2.676 [-53.783, +60.237] p=0.926 | -36.939 [-88.867, +16.668] p=0.169 |
| Dynamic-Obstacles-16x16-v0 | E_random_0.25 | B_always | -0.760 [-0.880, -0.640] p=0.000 | +111.259 [+59.745, +164.102] p=0.000 | +91.415 [+44.044, +139.774] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_obs | A_greedy | +0.540 [+0.400, +0.680] p=0.000 | -258.191 [-310.680, -207.095] p=0.000 | -243.257 [-292.541, -195.224] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_obs | B_always | -0.420 [-0.560, -0.280] p=0.000 | -149.608 [-165.228, -134.660] p=0.000 | -114.904 [-132.346, -98.600] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_sim | A_greedy | +0.960 [+0.900, +1.000] p=0.000 | -258.191 [-310.680, -207.095] p=0.000 | -243.278 [-292.562, -195.247] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | I_planner_sim | B_always | +0.000 [+0.000, +0.000] p=1.000 | -149.608 [-165.228, -134.660] p=0.000 | -114.924 [-132.363, -98.623] p=0.000 |
| Dynamic-Obstacles-16x16-v0 | L_greedy_8b | A_greedy | +0.160 [+0.040, +0.280] p=0.020 | -85.637 [-148.858, -23.680] p=0.007 | -87.429 [-147.154, -29.646] p=0.003 |
| Dynamic-Obstacles-16x16-v0 | L_greedy_8b | B_always | -0.800 [-0.900, -0.680] p=0.000 | +22.946 [-19.763, +70.022] p=0.313 | +40.925 [+0.249, +85.334] p=0.059 |
