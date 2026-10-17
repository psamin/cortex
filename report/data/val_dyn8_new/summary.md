Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy | 50 | 0.14 [0.06, 0.24] | 0.86 | 23.1 | 0.00 | 23.1 | 67.5 | 0.00 | 64.7 | 2.85/3.53 |
| Dynamic-Obstacles-8x8-v0 | J_verify_always | 50 | 0.56 [0.42, 0.70] | 0.44 | 30.8 | 1.00 | 61.6 | 120.9 | 0.00 | 67.4 | 0.99/5.89 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | 50 | 0.56 [0.42, 0.70] | 0.44 | 30.8 | 0.91 | 150.8 | 883.1 | 0.00 | 271.6 | 9.01/10.19 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | 49 | 0.16 [0.06, 0.27] | 0.39 | 179.7 | 1.00 | 359.4 | 543.2 | 0.00 | 523.0 | 3.13/3.62 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | 50 | 0.40 [0.26, 0.54] | 0.60 | 28.9 | 0.00 | 28.9 | 75.1 | 0.00 | 73.8 | 2.51/3.19 |
| Dynamic-Obstacles-8x8-v0 | M_vote5_sel_0.4 | 26 | 0.04 [0.00, 0.12] | 0.96 | 9.1 | 0.92 | 77.7 | 489.8 | 0.00 | 168.1 | 21.54/28.65 |
| Dynamic-Obstacles-8x8-v0 | O_cascade_sel_0.4 | 50 | 0.38 [0.26, 0.52] | 0.62 | 25.4 | 0.92 | 124.6 | 659.5 | 0.00 | 290.9 | 12.37/15.04 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy | L_greedy_8b | -0.260 [-0.400, -0.140] p=0.000 | -7.587 [-24.988, +9.745] p=0.388 | -9.142 [-25.780, +7.314] p=0.271 |
| Dynamic-Obstacles-8x8-v0 | J_verify_always | A_greedy | +0.420 [+0.280, +0.560] p=0.000 | +53.394 [+36.379, +72.433] p=0.000 | +2.789 [-15.583, +23.539] p=0.781 |
| Dynamic-Obstacles-8x8-v0 | J_verify_always | L_greedy_8b | +0.160 [-0.020, +0.340] p=0.112 | +45.807 [+21.466, +71.830] p=0.000 | -6.353 [-27.406, +16.140] p=0.570 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | A_greedy | +0.420 [+0.280, +0.560] p=0.000 | +815.620 [+683.716, +960.133] p=0.000 | +206.953 [+169.745, +247.481] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | L_greedy_8b | +0.160 [-0.020, +0.340] p=0.112 | +808.033 [+668.508, +961.213] p=0.000 | +197.812 [+152.874, +246.176] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | A_greedy | +0.020 [-0.122, +0.163] p=0.884 | +474.949 [+403.666, +543.701] p=0.000 | +457.596 [+390.063, +522.884] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | L_greedy_8b | -0.245 [-0.429, -0.061] p=0.009 | +467.316 [+396.453, +534.606] p=0.000 | +448.491 [+381.244, +512.387] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | A_greedy | +0.260 [+0.140, +0.400] p=0.000 | +7.587 [-9.745, +24.988] p=0.388 | +9.142 [-7.314, +25.780] p=0.271 |
| Dynamic-Obstacles-8x8-v0 | M_vote5_sel_0.4 | A_greedy | -0.077 [-0.231, +0.077] p=0.440 | +424.666 [+328.508, +527.810] p=0.000 | +105.196 [+66.341, +148.061] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | M_vote5_sel_0.4 | L_greedy_8b | -0.423 [-0.615, -0.231] p=0.000 | +407.047 [+312.282, +507.970] p=0.000 | +86.793 [+48.315, +128.145] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | O_cascade_sel_0.4 | A_greedy | +0.240 [+0.100, +0.380] p=0.001 | +592.037 [+484.279, +707.057] p=0.000 | +226.266 [+171.687, +284.920] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | O_cascade_sel_0.4 | L_greedy_8b | -0.020 [-0.120, +0.080] p=0.846 | +584.450 [+487.316, +687.691] p=0.000 | +217.124 [+173.034, +264.802] p=0.000 |
