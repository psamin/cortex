Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | B0_no_spec | 20 | 0.45 [0.25, 0.65] | 0.55 | 34.5 | 0.00 | 34.5 | 24.5 | 0.00 | 24.5 | 0.70/0.78 |
| Dynamic-Obstacles-8x8-v0 | B1_ssd_sel_0.4 | 20 | 1.00 [1.00, 1.00] | 0.00 | 22.1 | 0.91 | 88.6 | 102.0 | 0.23 | 50.0 | 2.23/2.50 |
| Dynamic-Obstacles-8x8-v0 | B2_predict_no_exec | 20 | 0.50 [0.30, 0.70] | 0.50 | 35.1 | 0.00 | 35.1 | 24.9 | 0.00 | 25.0 | 0.71/0.78 |
| Dynamic-Obstacles-8x8-v0 | B3_frequency_top1 | 20 | 0.35 [0.15, 0.55] | 0.65 | 22.4 | 0.00 | 22.4 | 31.6 | 0.00 | 16.6 | 0.94/1.21 |
| Dynamic-Obstacles-8x8-v0 | B4_global_top1 | 20 | 0.35 [0.15, 0.55] | 0.65 | 24.1 | 0.00 | 24.1 | 38.1 | 0.00 | 17.5 | 0.98/1.31 |
| Dynamic-Obstacles-8x8-v0 | B5_personalized_top1 | 20 | 0.30 [0.10, 0.50] | 0.70 | 25.2 | 0.00 | 25.2 | 38.6 | 0.00 | 17.7 | 0.99/1.29 |
| Dynamic-Obstacles-8x8-v0 | B6_personalized_utility | 20 | 0.40 [0.20, 0.60] | 0.60 | 24.1 | 0.00 | 24.1 | 36.3 | 0.00 | 16.6 | 0.98/1.29 |
| Dynamic-Obstacles-8x8-v0 | B6a_personalized_utility_adaptive | 20 | 0.55 [0.35, 0.75] | 0.45 | 34.5 | 0.00 | 34.5 | 53.7 | 0.00 | 24.6 | 0.99/1.30 |
| Dynamic-Obstacles-8x8-v0 | B8_ssd_plus_prefetch | 20 | 1.00 [1.00, 1.00] | 0.00 | 20.4 | 0.91 | 81.4 | 128.6 | 0.22 | 46.8 | 2.44/3.13 |
| Dynamic-Obstacles-8x8-v0 | Heur_heuristic_top1 | 20 | 0.35 [0.15, 0.55] | 0.65 | 26.2 | 0.00 | 26.2 | 45.0 | 0.00 | 20.4 | 1.05/1.31 |
| Dynamic-Obstacles-8x8-v0 | Rand_random_sched | 20 | 0.55 [0.35, 0.75] | 0.45 | 30.9 | 0.00 | 30.9 | 55.2 | 0.00 | 26.0 | 1.04/1.33 |
| Dynamic-Obstacles-8x8-v0 | TopK3_all_actions | 20 | 0.35 [0.15, 0.55] | 0.65 | 27.4 | 0.00 | 27.4 | 121.2 | 0.00 | 31.1 | 1.28/1.82 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | B0_no_spec | B1_ssd_sel_0.4 | -0.550 [-0.750, -0.350] p=0.000 | -77.720 [-91.847, -64.348] p=0.000 | -25.469 [-34.024, -16.620] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B1_ssd_sel_0.4 | B0_no_spec | +0.550 [+0.350, +0.750] p=0.000 | +77.720 [+64.348, +91.847] p=0.000 | +25.469 [+16.620, +34.024] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B2_predict_no_exec | B0_no_spec | +0.050 [-0.300, +0.400] p=0.890 | +0.474 [-6.723, +7.300] p=0.894 | +0.492 [-6.728, +7.335] p=0.890 |
| Dynamic-Obstacles-8x8-v0 | B2_predict_no_exec | B1_ssd_sel_0.4 | -0.500 [-0.700, -0.300] p=0.000 | -77.246 [-90.697, -65.313] p=0.000 | -24.977 [-32.683, -17.315] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B3_frequency_top1 | B0_no_spec | -0.100 [-0.350, +0.150] p=0.594 | +7.173 [-5.569, +19.726] p=0.271 | -7.926 [-17.284, +0.631] p=0.079 |
| Dynamic-Obstacles-8x8-v0 | B3_frequency_top1 | B1_ssd_sel_0.4 | -0.650 [-0.850, -0.450] p=0.000 | -70.547 [-86.370, -54.926] p=0.000 | -33.395 [-41.299, -25.613] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B4_global_top1 | B0_no_spec | -0.100 [-0.350, +0.150] p=0.591 | +13.640 [+2.215, +24.838] p=0.018 | -7.076 [-15.595, +0.904] p=0.090 |
| Dynamic-Obstacles-8x8-v0 | B4_global_top1 | B1_ssd_sel_0.4 | -0.650 [-0.850, -0.450] p=0.000 | -64.080 [-77.330, -50.785] p=0.000 | -32.545 [-39.232, -25.912] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B5_personalized_top1 | B0_no_spec | -0.150 [-0.450, +0.150] p=0.439 | +14.117 [+1.060, +27.703] p=0.039 | -6.818 [-15.232, +0.920] p=0.094 |
| Dynamic-Obstacles-8x8-v0 | B5_personalized_top1 | B1_ssd_sel_0.4 | -0.700 [-0.900, -0.500] p=0.000 | -63.603 [-80.968, -44.095] p=0.000 | -32.287 [-40.725, -22.901] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B6_personalized_utility | B0_no_spec | -0.050 [-0.250, +0.150] p=0.818 | +11.834 [+0.730, +23.077] p=0.038 | -7.966 [-15.505, -1.172] p=0.029 |
| Dynamic-Obstacles-8x8-v0 | B6_personalized_utility | B1_ssd_sel_0.4 | -0.600 [-0.800, -0.400] p=0.000 | -65.886 [-82.335, -48.863] p=0.000 | -33.434 [-41.459, -25.133] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B6a_personalized_utility_adaptive | B0_no_spec | +0.100 [-0.200, +0.400] p=0.629 | +29.207 [+13.220, +47.761] p=0.002 | +0.054 [-9.167, +9.913] p=0.991 |
| Dynamic-Obstacles-8x8-v0 | B6a_personalized_utility_adaptive | B1_ssd_sel_0.4 | -0.450 [-0.650, -0.250] p=0.000 | -48.513 [-66.361, -31.535] p=0.000 | -25.415 [-33.598, -17.628] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B8_ssd_plus_prefetch | B0_no_spec | +0.550 [+0.350, +0.750] p=0.000 | +104.367 [+87.305, +122.171] p=0.000 | +22.218 [+13.003, +31.136] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | B8_ssd_plus_prefetch | B1_ssd_sel_0.4 | +0.000 [+0.000, +0.000] p=1.000 | +26.648 [+12.085, +39.770] p=0.000 | -3.251 [-9.263, +1.975] p=0.254 |
| Dynamic-Obstacles-8x8-v0 | Heur_heuristic_top1 | B0_no_spec | -0.100 [-0.350, +0.150] p=0.519 | +20.516 [+4.312, +38.663] p=0.018 | -4.132 [-12.992, +4.714] p=0.365 |
| Dynamic-Obstacles-8x8-v0 | Heur_heuristic_top1 | B1_ssd_sel_0.4 | -0.650 [-0.850, -0.450] p=0.000 | -57.204 [-72.086, -42.290] p=0.000 | -29.601 [-36.383, -22.810] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | Rand_random_sched | B0_no_spec | +0.100 [-0.250, +0.450] p=0.684 | +30.707 [+14.624, +48.026] p=0.000 | +1.424 [-8.857, +11.859] p=0.783 |
| Dynamic-Obstacles-8x8-v0 | Rand_random_sched | B1_ssd_sel_0.4 | -0.450 [-0.650, -0.250] p=0.000 | -47.013 [-64.330, -29.451] p=0.000 | -24.044 [-32.518, -15.362] p=0.000 |
| Dynamic-Obstacles-8x8-v0 | TopK3_all_actions | B0_no_spec | -0.100 [-0.450, +0.250] p=0.662 | +96.752 [+67.731, +126.987] p=0.000 | +6.596 [-2.039, +15.043] p=0.130 |
| Dynamic-Obstacles-8x8-v0 | TopK3_all_actions | B1_ssd_sel_0.4 | -0.650 [-0.850, -0.450] p=0.000 | +19.032 [-13.890, +52.284] p=0.262 | -18.873 [-28.580, -9.366] p=0.000 |
