Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy | 50 | 0.14 [0.06, 0.24] | 0.86 | 23.1 | 0.00 | 23.1 | 67.5 | 0.00 | 64.7 | 2.85/3.53 |
| Dynamic-Obstacles-8x8-v0 | J_verify_always | 50 | 0.56 [0.42, 0.70] | 0.44 | 30.8 | 1.00 | 61.6 | 120.9 | 0.00 | 67.4 | 0.99/5.89 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | 50 | 0.56 [0.42, 0.70] | 0.44 | 30.8 | 0.91 | 150.8 | 883.1 | 0.00 | 271.6 | 9.01/10.19 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | 50 | 0.16 [0.06, 0.26] | 0.38 | 181.3 | 1.00 | 362.5 | 550.9 | 0.00 | 530.5 | 3.13/3.84 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | 50 | 0.40 [0.26, 0.54] | 0.60 | 28.9 | 0.00 | 28.9 | 75.1 | 0.00 | 73.8 | 2.51/3.19 |
| Dynamic-Obstacles-8x8-v0 | M_vote5_sel_0.4 | 50 | 0.14 [0.06, 0.24] | 0.86 | 12.8 | 0.91 | 109.3 | 692.0 | 0.00 | 251.1 | 20.87/29.57 |
| Dynamic-Obstacles-8x8-v0 | N_mcts_always | 50 | 1.00 [1.00, 1.00] | 0.00 | 24.8 | 1.00 | 24.8 | 107.0 | 3.99 | 97.7 | 4.02/6.71 |
| Dynamic-Obstacles-8x8-v0 | N_mcts_sel_0.4 | 50 | 0.98 [0.94, 1.00] | 0.02 | 25.9 | 0.90 | 103.7 | 648.4 | 3.72 | 220.1 | 7.97/16.10 |
| Dynamic-Obstacles-8x8-v0 | O_cascade_sel_0.4 | 50 | 0.38 [0.26, 0.52] | 0.62 | 25.4 | 0.92 | 124.6 | 659.5 | 0.00 | 290.9 | 12.37/15.04 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy | L_greedy_8b | -0.260 [-0.400, -0.140] p=0.000 | -7.587 [-24.988, +9.745] p=0.388 | -9.142 [-25.780, +7.314] p=0.271 |
| Dynamic-Obstacles-8x8-v0 | J_verify_always | A_greedy | +0.420 [+0.280, +0.560] p=0.000 | +53.394 [+36.379, +72.433] p=0.000 | +2.789 [-15.583, +23.539] p=0.781 |
| Dynamic-Obstacles-8x8-v0 | J_verify_always | L_greedy_8b | +0.160 [-0.020, +0.340] p=0.112 | +45.807 [+21.466, +71.830] p=0.000 | -6.353 [-27.406, +16.140] p=0.570 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | A_greedy | +0.420 [+0.280, +0.560] p=0.000 | +815.620 [+683.716, +960.133] p=0.000 | +206.953 [+169.745, +247.481] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | L_greedy_8b | +0.160 [-0.020, +0.340] p=0.112 | +808.033 [+668.508, +961.213] p=0.000 | +197.812 [+152.874, +246.176] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | A_greedy | +0.020 [-0.120, +0.160] p=0.890 | +483.350 [+410.725, +551.784] p=0.000 | +465.804 [+396.192, +531.059] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | L_greedy_8b | -0.240 [-0.420, -0.080] p=0.007 | +475.763 [+404.205, +543.386] p=0.000 | +456.662 [+388.060, +521.127] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | A_greedy | +0.260 [+0.140, +0.400] p=0.000 | +7.587 [-9.745, +24.988] p=0.388 | +9.142 [-7.314, +25.780] p=0.271 |
| Dynamic-Obstacles-8x8-v0 | M_vote5_sel_0.4 | A_greedy | +0.000 [-0.140, +0.140] p=1.000 | +624.516 [+523.920, +729.224] p=0.000 | +186.423 [+145.592, +230.235] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | M_vote5_sel_0.4 | L_greedy_8b | -0.260 [-0.440, -0.080] p=0.004 | +616.929 [+515.894, +722.678] p=0.000 | +177.281 [+135.533, +221.294] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | N_mcts_always | A_greedy | +0.860 [+0.760, +0.940] p=0.000 | +43.513 [+30.052, +56.439] p=0.000 | +33.089 [+20.168, +45.702] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | N_mcts_always | L_greedy_8b | +0.600 [+0.460, +0.740] p=0.000 | +35.926 [+20.064, +52.033] p=0.000 | +23.947 [+7.554, +40.350] p=0.004 |
| Dynamic-Obstacles-8x8-v0 | N_mcts_sel_0.4 | A_greedy | +0.840 [+0.740, +0.940] p=0.000 | +584.598 [+528.429, +642.548] p=0.000 | +155.492 [+128.539, +183.442] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | N_mcts_sel_0.4 | L_greedy_8b | +0.580 [+0.440, +0.720] p=0.000 | +577.011 [+516.717, +638.853] p=0.000 | +146.350 [+113.326, +180.619] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | O_cascade_sel_0.4 | A_greedy | +0.240 [+0.100, +0.380] p=0.001 | +592.037 [+484.279, +707.057] p=0.000 | +226.266 [+171.687, +284.920] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | O_cascade_sel_0.4 | L_greedy_8b | -0.020 [-0.120, +0.080] p=0.846 | +584.450 [+487.316, +687.691] p=0.000 | +217.124 [+173.034, +264.802] p=0.000 |
