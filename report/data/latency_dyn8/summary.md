Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy_4b | 20 | 0.20 [0.05, 0.40] | 0.80 | 24.1 | 0.00 | 24.1 | 46.6 | 0.00 | 46.6 | 1.80/2.82 |
| Dynamic-Obstacles-8x8-v0 | C_sel_0.4 | 1 | 1.00 [1.00, 1.00] | 0.00 | 25.0 | 0.92 | 100.0 | 324.3 | 0.26 | 171.1 | 6.47/9.99 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | 18 | 0.50 [0.28, 0.72] | 0.50 | 31.0 | 0.93 | 152.3 | 418.8 | 0.00 | 229.2 | 7.33/8.52 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | 20 | 0.15 [0.00, 0.30] | 0.20 | 225.4 | 1.00 | 450.9 | 451.0 | 0.00 | 451.6 | 1.95/2.45 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_both | 20 | 0.45 [0.25, 0.65] | 0.55 | 31.8 | 0.00 | 92.2 | 469.1 | 0.00 | 157.3 | 4.91/7.81 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_heur | 20 | 0.35 [0.15, 0.55] | 0.65 | 24.1 | 0.00 | 36.4 | 143.0 | 0.00 | 66.4 | 3.39/5.09 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_small | 20 | 0.25 [0.10, 0.45] | 0.75 | 21.8 | 0.00 | 55.1 | 233.1 | 0.00 | 99.2 | 4.64/6.17 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | 20 | 0.40 [0.20, 0.60] | 0.60 | 22.4 | 0.00 | 22.4 | 53.4 | 0.00 | 53.4 | 2.24/3.29 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | A_greedy_4b | L_greedy_8b | -0.200 [-0.500, +0.100] p=0.243 | -6.810 [-31.389, +18.097] p=0.588 | -6.807 [-31.406, +18.121] p=0.588 |
| Dynamic-Obstacles-8x8-v0 | C_sel_0.4 | L_greedy_8b | +1.000 [+1.000, +1.000] p=0.000 | +243.476 [+243.476, +243.476] p=0.000 | +89.965 [+89.965, +89.965] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | C_sel_0.4 | A_greedy_4b | +0.000 [+0.000, +0.000] p=1.000 | +241.054 [+241.054, +241.054] p=0.000 | +87.516 [+87.516, +87.516] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | L_greedy_8b | +0.056 [-0.333, +0.444] p=0.894 | +360.481 [+217.927, +519.412] p=0.000 | +170.842 [+90.735, +260.082] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | J_verify_sel_0.4 | A_greedy_4b | +0.278 [+0.000, +0.556] p=0.101 | +367.723 [+227.097, +523.163] p=0.000 | +178.082 [+98.240, +263.213] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | L_greedy_8b | -0.250 [-0.500, +0.000] p=0.060 | +397.590 [+345.011, +441.839] p=0.000 | +398.160 [+345.507, +442.476] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | K_specdraft_L4 | A_greedy_4b | -0.050 [-0.250, +0.150] p=0.817 | +404.400 [+343.058, +457.349] p=0.000 | +404.967 [+343.532, +457.983] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_both | L_greedy_8b | +0.050 [-0.250, +0.350] p=0.861 | +415.783 [+272.363, +580.531] p=0.000 | +103.910 [+60.068, +152.723] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_both | A_greedy_4b | +0.250 [+0.000, +0.500] p=0.063 | +422.593 [+273.265, +597.604] p=0.000 | +110.717 [+59.393, +171.567] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_heur | L_greedy_8b | -0.050 [-0.350, +0.250] p=0.867 | +89.629 [+53.609, +126.690] p=0.000 | +13.021 [-7.341, +33.174] p=0.198 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_heur | A_greedy_4b | +0.150 [-0.100, +0.400] p=0.326 | +96.439 [+53.562, +137.766] p=0.000 | +19.828 [-8.799, +45.457] p=0.154 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_small | L_greedy_8b | -0.150 [-0.450, +0.150] p=0.385 | +179.786 [+110.739, +248.208] p=0.000 | +45.764 [+13.799, +76.143] p=0.003 |
| Dynamic-Obstacles-8x8-v0 | LS_lossless_small | A_greedy_4b | +0.050 [-0.150, +0.250] p=0.822 | +186.596 [+119.115, +255.867] p=0.000 | +52.572 [+18.591, +85.528] p=0.002 |
| Dynamic-Obstacles-8x8-v0 | L_greedy_8b | A_greedy_4b | +0.200 [-0.100, +0.500] p=0.243 | +6.810 [-18.097, +31.389] p=0.588 | +6.807 [-18.121, +31.406] p=0.588 |
