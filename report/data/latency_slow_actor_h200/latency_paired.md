Reference: L_greedy_32b. Paired over common seeds; 95% bootstrap CIs.

| method | n | success | decision latency s [95% CI] | Δ vs ref (paired) [95% CI], p | rel. | model-s / decision | Δ compute / decision | hit / step | launched: reused / late / unused | wasted model-s / ep |
|---|---|---|---|---|---|---|---|---|---|---|
| L_greedy_32b | 12 | 0.67 | 1.682 [1.580, 1.815] | +0.000 [+0.000, +0.000], p=1.000 | +0.0% | 1.681 | +0.000 | — | — / — / — | — |
| LS32_lossless_heur | 12 | 0.92 | 1.732 [1.677, 1.779] | +0.050 [-0.090, +0.173], p=0.469 | +3.0% | 3.852 | +2.171 | 0.48 | — / — / — | — |
| K2_32b_lossless_draft_L2 | 12 | 0.83 | 1.805 [1.734, 1.887] | +0.123 [+0.015, +0.219], p=0.018 | +7.3% | 2.988 | +1.307 | — | — / — / — | — |
| LS32_lossless_small | 12 | 0.83 | 2.200 [2.170, 2.231] | +0.518 [+0.365, +0.641], p=0.000 | +30.8% | 5.026 | +3.345 | 0.49 | — / — / — | — |
