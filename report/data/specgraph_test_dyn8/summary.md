Episodes with runner errors (excluded, listed in episodes.jsonl): 0; duplicate (method, seed) episodes skipped: 0

| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | B0_no_spec | 20 | 0.45 [0.25, 0.65] | 0.55 | 34.5 | 0.00 | 34.5 | 24.5 | 0.00 | 24.5 | 0.70/0.78 |
| Dynamic-Obstacles-8x8-v0 | B2_predict_no_exec | 20 | 0.50 [0.30, 0.70] | 0.50 | 35.1 | 0.00 | 35.1 | 24.9 | 0.00 | 25.0 | 0.71/0.78 |
| Dynamic-Obstacles-8x8-v0 | B3_frequency_top1 | 20 | 0.35 [0.15, 0.55] | 0.65 | 22.4 | 0.00 | 22.4 | 31.6 | 0.00 | 16.6 | 0.94/1.21 |
| Dynamic-Obstacles-8x8-v0 | B4_global_top1 | 20 | 0.35 [0.15, 0.55] | 0.65 | 24.1 | 0.00 | 24.1 | 38.1 | 0.00 | 17.5 | 0.98/1.31 |
| Dynamic-Obstacles-8x8-v0 | B5_personalized_top1 | 20 | 0.30 [0.10, 0.50] | 0.70 | 25.2 | 0.00 | 25.2 | 38.6 | 0.00 | 17.7 | 0.99/1.29 |
| Dynamic-Obstacles-8x8-v0 | B6_personalized_utility | 20 | 0.40 [0.20, 0.60] | 0.60 | 24.1 | 0.00 | 24.1 | 36.3 | 0.00 | 16.6 | 0.98/1.29 |
| Dynamic-Obstacles-8x8-v0 | B6a_personalized_utility_adaptive | 20 | 0.55 [0.35, 0.75] | 0.45 | 34.5 | 0.00 | 34.5 | 53.7 | 0.00 | 24.6 | 0.99/1.30 |
| Dynamic-Obstacles-8x8-v0 | Rand_random_sched | 7 | 0.57 [0.14, 0.86] | 0.43 | 32.0 | 0.00 | 32.0 | 57.8 | 0.00 | 27.0 | 1.04/1.34 |
| Dynamic-Obstacles-8x8-v0 | TopK3_all_actions | 20 | 0.35 [0.15, 0.55] | 0.65 | 27.4 | 0.00 | 27.4 | 121.2 | 0.00 | 31.1 | 1.28/1.82 |

Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):

| env | method | vs | Δ success | Δ compute-s | Δ episode-s |
|---|---|---|---|---|---|
| Dynamic-Obstacles-8x8-v0 | B2_predict_no_exec | B0_no_spec | +0.050 [-0.300, +0.400] p=0.890 | +0.474 [-6.723, +7.300] p=0.894 | +0.492 [-6.728, +7.335] p=0.890 |
| Dynamic-Obstacles-8x8-v0 | B3_frequency_top1 | B0_no_spec | -0.100 [-0.350, +0.150] p=0.594 | +7.173 [-5.569, +19.726] p=0.271 | -7.926 [-17.284, +0.631] p=0.079 |
| Dynamic-Obstacles-8x8-v0 | B4_global_top1 | B0_no_spec | -0.100 [-0.350, +0.150] p=0.591 | +13.640 [+2.215, +24.838] p=0.018 | -7.076 [-15.595, +0.904] p=0.090 |
| Dynamic-Obstacles-8x8-v0 | B5_personalized_top1 | B0_no_spec | -0.150 [-0.450, +0.150] p=0.439 | +14.117 [+1.060, +27.703] p=0.039 | -6.818 [-15.232, +0.920] p=0.094 |
| Dynamic-Obstacles-8x8-v0 | B6_personalized_utility | B0_no_spec | -0.050 [-0.250, +0.150] p=0.818 | +11.834 [+0.730, +23.077] p=0.038 | -7.966 [-15.505, -1.172] p=0.029 |
| Dynamic-Obstacles-8x8-v0 | B6a_personalized_utility_adaptive | B0_no_spec | +0.100 [-0.200, +0.400] p=0.629 | +29.207 [+13.220, +47.761] p=0.002 | +0.054 [-9.167, +9.913] p=0.991 |
| Dynamic-Obstacles-8x8-v0 | Rand_random_sched | B0_no_spec | +0.286 [-0.286, +0.714] p=0.419 | +37.541 [+3.022, +70.220] p=0.033 | +6.697 [-16.550, +26.523] p=0.554 |
| Dynamic-Obstacles-8x8-v0 | TopK3_all_actions | B0_no_spec | -0.100 [-0.450, +0.250] p=0.662 | +96.752 [+67.731, +126.987] p=0.000 | +6.596 [-2.039, +15.043] p=0.130 |
