Reference: B1_ssd_K2_llm_ranked. Paired over common seeds; 95% bootstrap CIs.

| method | n | success | decision latency s [95% CI] | Δ vs ref (paired) [95% CI], p | rel. | model-s / decision | Δ compute / decision | hit / step | launched: reused / late / unused | wasted model-s / ep |
|---|---|---|---|---|---|---|---|---|---|---|
| B7_ssd_K2_graph_ranked | 50 | 1.00 | 3.718 [3.645, 3.770] | -0.042 [-0.115, +0.014], p=0.205 | -1.1% | 7.158 | -0.045 | — | — / — / — | — |
| B1_ssd_K2_llm_ranked | 50 | 0.98 | 3.761 [3.743, 3.781] | +0.000 [+0.000, +0.000], p=1.000 | +0.0% | 7.203 | +0.000 | — | — / — / — | — |
