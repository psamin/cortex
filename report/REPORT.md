# DreamCache: does selective speculation help LLM agents?

**Scope.** All numbers come from runs on GT Futurama GPUs (H200 and L4) with qwen3:4b / qwen3:8b served by Ollama 0.12.6 (`think: false`). Run directories, manifests, and the scripts that produced every table are listed in [Reproduce](#reproduce). Methods and definitions are in [METHODS.md](METHODS.md). Experiment 2 (graph-based predictive execution) is reported separately in [experiments/speculative_graph/REPORT.md](../experiments/speculative_graph/REPORT.md).

**Held-out test (§2)** uses seeds 10000–10099. Method definitions were frozen in commit `d4b646d` before any test episode ran. The run was split across three GPU jobs for time, with identical definitions, and episodes were deduplicated by (method, seed). §1 and §3–§6 come from validation seeds 1000–1049, which informed the choices.

## TL;DR
1. **Decision speculation works, and *where* you trigger it matters a lot.** Held-out test (100 seeds): risk-triggered speculation reaches 1.00 success at a 32% trigger rate, vs 0.37 for random triggering at a matched rate (paired +0.63, p<0.001), vs 0.17 for 4b greedy and 0.38 for 8b greedy. Validation numbers:

   | Method | Trigger rate | Success |
   |---|---|---|
   | 4b greedy | — | 0.14 |
   | Always-on sandbox rollouts | 100% | 1.00 |
   | Observable-risk trigger | **30%** | **1.00** |
   | Random trigger | 25% | 0.26 |
   | Periodic trigger | 27% | 0.24 |
   | Oracle-regret trigger | 5% | 1.00 |

2. **The uncertainty signals in the original design don't identify consequential decisions.** Against oracle regret labels on 5,724 states (base rate 5.2%), PR-AUC is entropy 0.04, disagreement 0.05, random 0.05, and **observable risk 0.42**. The top 20% of states by risk hold 97% of all regret. The "confidently wrong" states (33 of them) are all caught by risk.
3. **"Selective" did not save compute in MiniGrid** (on test, risk-only cost +166 model-s / ep vs always-on, p<0.001). Sandbox rollouts with a heuristic policy cost about 0.2 CPU-s per episode, which is negligible. The LLM call happens on every step regardless. The disagreement signal needs 3 extra LLM calls per step, so the full detector used **5.6× the model-seconds of always-on** (542 vs 96 model-s per episode). The risk-only trigger adds no calls, but its episodes ran longer (42 vs 18.5 steps), so it used 1.8× the model-seconds. Selective speculation pays off only when the speculation itself is expensive, for example LLM rollouts (§4).
4. **LLM-only test-time compute was weak here.**

   | Method | Success |
   |---|---|
   | 4b greedy | 0.14 |
   | 8b greedy | 0.40 |
   | 4b draft → 8b verify (8b accepts 93%) | **0.56** |
   | 8b cascade on trigger | 0.38 |
   | Self-consistency vote ×5 | 0.04 |
   | Best-of-K without rollouts | 0.46 |
   | Every LLM method on DoorKey | 0.00 |
   | Rollouts on DoorKey | 0.60–0.73 |

   **Simulated consequences, not more model reasoning, are what fix decisions.**
5. **Inference speedups: none measured for speculative LLM execution.** Under both high contention (L4, 4 agents) and low contention (H200, 2 agents):
   - Lossless speculative actions (after Ye et al. 2025) and lossless speculative plan drafting (greedy speculative decoding over actions, after DSP / ISP) both **increased** mean decision latency over 8b greedy. Low contention: 0.78–1.32 s vs 0.71 s. High contention: 2.8–5.0 s vs 2.4 s.
   - Prefetch hit rates were 26–65%, and draft acceptance was 66%.
   - **Why, measured:** 4b and 8b have the *same* per-call latency on the H200 (0.69 vs 0.68 s; fixed per-call overhead dominates), so a draft is not cheaper than the target. Concurrent speculative calls also slow down the real call; misses rise from 0.68 to about 1.0 s.
   - A 32b-actor run tests the regime where theory predicts a gain (§5).
   - In the coding-agent track, pre-executing tools did cut wall-clock time. See experiment 2.

## 1. Validation results: Dynamic-Obstacles-8x8 (50 seeds, paired)
`report/data/val_dyn8_core/summary.md` and `report/data/val_dyn8_new/summary.md`

| Method | Success [95% CI] | Collision | Steps | Trigger rate | LLM calls / ep | Model-s / ep | Sim CPU-s / ep |
|---|---|---|---|---|---|---|---|
| A greedy 4b | 0.14 [0.06, 0.24] | 0.86 | 22.9 | 0 | 22.9 | 125.6 | 0 |
| L greedy 8b | 0.40 [0.26, 0.54] | 0.60 | 28.9 | 0 | 28.9 | 75.1 | 0 |
| B always-on rollouts | **1.00** [1.00, 1.00] | 0 | 18.5 | 1.00 | 18.5 | 96.4 | 0.21 |
| C selective, all signals, 0.2 | 1.00 | 0 | 18.7 | 0.97 | 75.0 | 535.2 | 0.21 |
| C selective, all signals, 0.4 | 1.00 | 0 | 19.3 | 0.90 | 77.2 | 542.1 | 0.19 |
| C selective, all signals, 0.6 | 0.96 [0.90, 1.00] | 0.04 | 24.0 | 0.64 | 95.8 | 617.4 | 0.17 |
| C selective, all signals, 0.8 | 0.58 [0.44, 0.72] | 0.42 | 34.3 | 0.21 | 137.3 | 823.6 | 0.08 |
| C entropy only, 0.4 | 0.92 [0.84, 0.98] | 0.08 | 17.7 | 0.95 | 17.7 | 97.3 | 0.18 |
| **C risk only, 0.4** | **1.00** [1.00, 1.00] | 0 | 42.0 | **0.30** | 42.0 | 172.7 | 0.14 |
| D oracle trigger (not deployable) | 1.00 | 0 | 69.7 | 0.05 | 69.7 | 286.4 | 0.03 |
| E random, rate 0.25 | 0.26 [0.14, 0.38] | 0.74 | 18.4 | 0.24 | 18.4 | 82.5 | 0.05 |
| E random, rate 0.5 | 0.58 [0.44, 0.72] | 0.42 | 19.1 | 0.49 | 19.1 | 81.3 | 0.11 |
| F periodic, every 4 | 0.24 [0.12, 0.36] | 0.76 | 18.1 | 0.27 | 18.1 | 79.8 | 0.05 |
| F periodic, every 2 | 0.54 [0.40, 0.68] | 0.46 | 22.8 | 0.51 | 22.8 | 88.2 | 0.13 |
| G best-of-K, no rollouts | 0.46 [0.32, 0.60] | 0.54 | 46.9 | 1.00 | 46.9 | 204.2 | 0 |
| J 4b draft → 8b verify (always) | 0.56 [0.42, 0.70] | 0.44 | 30.8 | 1.00 | 61.6 | 120.9 | 0 |
| J verify on trigger 0.4 | 0.56 [0.42, 0.70] | 0.44 | 30.8 | 0.91 | 150.8 | 883.1 | 0 |
| O cascade to 8b on trigger 0.4 | 0.38 [0.26, 0.52] | 0.62 | 25.4 | 0.92 | 124.6 | 659.5 | 0 |
| K plan drafting (self-reported accept) | 0.16 [0.06, 0.27] | 0.39 | 179.7 | 1.00 | 359.4 | 543.2 | 0 |
| M self-consistency vote ×5 on trigger | 0.04 [0.00, 0.12] (n=26) | 0.96 | 9.1 | 0.92 | 77.7 | 489.8 | 0 |
| I planner, SimBFS (privileged) | 1.00 | 0 | 17.8 | — | 0 | 0 | — |
| I planner, ObsBFS (observation-only) | 1.00 | 0 | 64.2 | — | 0 | 0 | — |

Notes:
- **Model-seconds** count server time for every call, including calls served from the response cache, which are charged their original cost. A_greedy's model-s in the core run reflects contention on that GPU; compare model-s only within a run.
- **Steps matter for compute.** Risk-only and oracle triggers prevent deaths, but between triggers the 4b agent dithers (turns in place), so its episodes are 2–4× longer.
- **The heuristic planners solve this task without any LLM.** ObsBFS uses no privileged information, and it is slow but safe. This is an important reality check: on this benchmark, the LLM is not the best policy.
- **K (self-reported acceptance) is a failure:** the 8b verifier accepted 0.0 drafted steps on average. The lossless K2 variant (§5) fixes this by construction.

**Selection rule** (stated before the test run): pick the selective variant with the highest val success among those with trigger rate ≤ 0.5, breaking ties by lower compute. Only C_sel_0.8 (0.58) and **C_risk_only_0.4 (1.00)** qualify, so the primary selective method is **C_risk_only_0.4**. The matched random control is E_random_0.25 (val trigger rate 0.24 vs 0.30).

> Disclosure: the frozen test config lists C_sel_0.4 / 0.6 / risk-only rather than C_sel_0.8. The rule was written over "thresholds", and was then applied over all selective detector variants run on val. Applied only to the full-detector thresholds, it would select C_sel_0.8, which isn't in the test set. Both readings are reported.

## 2. Held-out test: Dynamic-Obstacles-8x8 (seeds 10000–10099, n=100 per method)
`report/data/test_dyn8/summary.md` and `report/data/test_dyn8_vs_8b/summary.md`. No runner errors.

| Method | Success [95% CI] | Collision | Steps | Trigger rate | LLM calls / ep | Model-s / ep | Sim CPU-s / ep |
|---|---|---|---|---|---|---|---|
| A greedy 4b | 0.17 [0.10, 0.25] | 0.83 | 21.7 | 0 | 21.7 | 118.7 | 0 |
| L greedy 8b | 0.38 [0.29, 0.48] | 0.62 | 29.1 | 0 | 29.1 | 124.7 | 0 |
| B always-on rollouts | **1.00** [1.00, 1.00] | 0 | 17.6 | 1.00 | 17.6 | **102.9** | 0.18 |
| **C risk only, 0.4 (val-selected primary)** | **1.00** [1.00, 1.00] | 0 | 41.6 | **0.32** | 41.6 | 268.8 | 0.15 |
| C all signals, 0.4 | 1.00 | 0 | 18.3 | 0.90 | 73.3 | 969.6 | 0.17 |
| C all signals, 0.6 | 0.97 [0.93, 1.00] | 0.03 | 22.9 | 0.66 | 91.4 | 1089.2 | 0.16 |
| D oracle trigger (not deployable) | 1.00 | 0 | 52.0 | 0.04 | 52.0 | 352.1 | 0.02 |
| E random 0.25 (matched control) | 0.37 [0.28, 0.46] | 0.63 | 21.1 | 0.23 | 21.1 | 141.9 | 0.06 |
| E random 0.5 | 0.58 [0.48, 0.67] | 0.42 | 20.3 | 0.49 | 20.3 | 123.4 | 0.11 |
| G best-of-K, no rollouts | 0.55 [0.45, 0.65] | 0.45 | 39.9 | 1.00 | 39.9 | 213.3 | 0 |
| J 4b draft → 8b verify on trigger | 0.42 [0.33, 0.52] | 0.58 | 32.8 | 0.92 | 161.3 | 1607.5 | 0 |
| K2 lossless speculative drafting, L=4 | 0.34 [0.25, 0.43] | 0.66 | 27.8 | 0.64 | 73.7 | 773.3 | 0 |
| K plan drafting, self-reported (n=15, stopped early) | 0.40 [0.13, 0.67] | 0.60 | 91.8 | 1.00 | 183.6 | 520.2 | 0 |
| N UCT-MCTS on trigger | 1.00 | 0 | 24.2 | 0.89 | 96.6 | 1080.1 | 3.26 |
| S SwiftSage fast/slow trigger → 8b | 0.20 [0.12, 0.28] | 0.80 | 21.3 | 0.08 | 24.1 | 142.7 | 0 |
| I planner, SimBFS (privileged) / ObsBFS (observation-only) | 1.00 / 1.00 | 0 | 18.7 / 55.3 | — | 0 | 0 | — |

**Paired comparisons (Δ success [95% CI], bootstrap p):**
- **Selected selective vs matched random trigger:** C_risk 0.4 − E_random 0.25 = **+0.63 [+0.53, +0.72], p<0.001**. The detector, not just the amount of speculation, is what matters.
- **Selected selective vs always-on:** Δ success = 0.00, but Δ model-s / ep = **+165.8 [+140.6, +192.6]**. The success-vs-compute hypothesis is **rejected** for this domain: selective speculation matches always-on success, but costs more, because unprotected steps make episodes longer and every step pays for an LLM call.
- **Always-on vs 8b greedy:** +0.62 [+0.52, +0.71], p<0.001, at −21.6 model-s / ep. A small model plus simulation beats a bigger model without search, and costs less (Ablation 5).
- **Draft→verify vs 8b greedy:** +0.04 [−0.08, +0.16], p=0.57. The validation gain (0.56 vs 0.40) **did not replicate** on test.
- **Lossless drafting vs 8b greedy:** −0.04 [−0.16, +0.08], p=0.57. It's equivalent in success, as expected for a lossless method, at +649 model-s / ep.
- **SwiftSage trigger vs 8b greedy:** −0.18 [−0.30, −0.06], p=0.006. Its stuck/repeat rule fires on only 8% of steps, and it misses imminent collisions.
- **Best-of-K without rollouts vs 8b:** +0.17 [+0.04, +0.31], p=0.019. Observable-risk filtering helps, but far less than simulated rollouts (1.00).

## 3. Which signals find consequential decisions? (Ablation 4, offline)
`report/data/detector_delta0.2/`

**Setup.**
- **States:** 5,724, from the trajectories of 5 different policies (greedy-logged and C_sel at 0.2 / 0.4 / 0.6 / 0.8). This avoids evaluating only on the greedy policy's state distribution.
- **Q_ref(s,a):** 16 privileged SimBFS rollouts per action, horizon 100, common random numbers.
- **Critical:** regret of the fast action > δ. Regret is bimodal here (collision ≈ −1 vs. success ≈ +0.9), so δ ∈ {0.1, 0.2, 0.4} give **identical** labels. Base rate 5.2% (296 states). 33 states are "confidently wrong": top normalized score ≥ 0.6 and critical.

| Signal | PR-AUC | Regret captured @10% budget | @20% | Precision / recall @ thr 0.4 | Trigger rate @0.4 | Recall on confidently-wrong |
|---|---|---|---|---|---|---|
| entropy | 0.043 | 0.01 | 0.02 | 0.05 / 0.95 | 0.97 | 0.52 |
| disagreement (3 samples) | 0.052 | 0.06 | 0.17 | 0.05 / 0.85 | 0.86 | 0.94 |
| **observable risk** | **0.423** | **0.77** | **0.97** | 0.18 / **1.00** | **0.29** | **1.00** |
| entropy + disagreement | 0.042 | 0.01 | 0.02 | 0.05 / 0.95 | 0.96 | 0.58 |
| all three (0.4 / 0.3 / 0.3) | 0.224 | 0.37 | 0.76 | 0.06 / 1.00 | 0.90 | 0.97 |
| random | 0.052 | 0.10 | 0.20 | — | — | — |

**Reading.**
- The LLM's score entropy is *anti*-informative: below random at every budget. qwen3:4b's v3 scores are spread out almost everywhere (trigger rate 0.97 at 0.4).
- Disagreement among samples is at chance.
- An observation-only hazard check catches every critical state at a 29% trigger rate.
- Combining all three dilutes risk.
- This matches the online result (risk-only = always-on success at 30% of the triggers).

## 4. Ablations (val, always-on, 20 seeds) — `report/data/ablate_dyn8/`

| Ablation | Setting → success (steps) |
|---|---|
| Branch count K | K=1 → **0.15** (24.0); K=2 → 1.00 (37.5); K=3 → 1.00 (18.2) |
| Rollout depth H | H=1 → 1.00 (17.4); H=3 → 1.00 (17.3); H=5 → 1.00 (18.2); H=10 → 1.00 (21.1); H=20 → 1.00 (22.9) |
| Rollout policy | random → 0.95 (49.2); ObsBFS (observation-only) → 1.00 (18.7); SimBFS → 1.00 (18.2) |
| Evaluator | env reward only → 1.00 but 43.0 steps; shaped → 1.00 (18.2) |
| Samples per candidate | 1 → 1.00 (22.1); 4 → 1.00 (18.2) |
| Candidates | top-K by LLM → 1.00 (18.2); uniform over all legal → 1.00 (18.8); LLM sampled → 1.00 but 4× LLM calls |

- On this task a **1-step lookahead with ≥2 candidates is enough**. Collisions are immediate, observable consequences.
- Longer horizons add CPU without adding success.
- The shaped progress term mainly shortens episodes.
- Ablation 6 with the LLM itself as rollout policy is in `report/data/ablate_rollout_llm/` (if completed). Ablation 7 (parallel vs sequential branches) is in `report/data/ablate_parallel/`.

## 5. Inference speedups: speculative LLM execution (cache disabled)
`report/data/latency_single_h200/`, `report/data/latency_dyn8/`

**Low contention (H200, 2 concurrent agents, val seeds, n=12 per method):**

| Method | Mean decision latency [95% CI] | p50 / p95 | Calls / step | Prefetch hit / draft accept | Model-s / ep | Success |
|---|---|---|---|---|---|---|
| L greedy 8b (actor, reference) | **0.710** [0.683, 0.747] | 0.68 / 0.88 | 1.00 | — | 17.7 | 0.33 |
| A greedy 4b | 0.714 [0.694, 0.739] | 0.69 / 0.77 | 1.00 | — | 16.7 | 0.17 |
| LS lossless spec. actions, ObsBFS guesser | 0.777 [0.711, 0.855] | 1.01 / 1.28 | 1.49 | hit 0.41 | 43.2 | 0.42 |
| LS lossless, 4b guesser | 1.162 [1.137, 1.184] | 1.15 / 1.43 | 2.55 | hit 0.26 | 57.7 | 0.08 |
| LS lossless, both guessers (K=2) | 1.221 [1.187, 1.254] | 1.19 / 1.70 | 2.85 | hit 0.65 | 95.0 | 0.50 |
| K2 lossless draft, L=2 | 1.002 [0.931, 1.076] | 1.28 / 1.54 | 2.07 | accept 0.66 | 36.9 | 0.50 |
| K2 lossless draft, L=4 | 1.321 [1.195, 1.461] | 1.67 / 2.33 | 2.69 | accept 0.66 | 89.2 | 0.25 |

**High contention (L4, 4 concurrent agents, 20 seeds):** 8b greedy 2.41 s mean. Lossless with ObsBFS guesser 2.80 s (hit 0.42), with 4b guesser 4.46 s, with both 4.99 s. K plan drafting 2.04 s, but only because it accepts nothing and runs the 8b correction directly.

**Interpretation.**
- The latency model behind Speculative Actions gives a maximum reduction of α/(α+β)·p/(1+p). It assumes the speculative calls are free in parallel and that the guesser is much cheaper than the actor. Neither assumption held:
  1. **No draft/target gap.** qwen3:4b and 8b take 0.69 vs 0.68 s per decision on an H200, because about 260 prompt tokens plus about 60 generated tokens are dominated by fixed overhead. The heuristic guesser is free, but only 41% accurate.
  2. **Parallel calls are not free.** On the L4, throughput rises only from 1.2 to 3.0 calls/s going from 1 to 16 concurrent requests, so a second in-flight call slows the first.
- Every lossless variant is "lossless" only up to server nondeterminism. Temperature-0 outputs varied with batch composition: in the bench, `forward` was chosen 31/32 vs 26/32 at different concurrency levels. Success rates therefore differ slightly between LS and L even though both execute the 8b policy.
- **Slow-actor regime (qwen3:32b actor, guess/actor latency ratio ≈ 0.4; H200, n=12):** still no speedup. 32b greedy takes 1.682 s per decision. Lossless with the ObsBFS guesser took 1.732 s (+3%, n.s., hit 0.48), lossless drafting L=2 1.805 s (+7%, p=0.018), and lossless with the 4b guesser 2.200 s (+31%). On one local GPU, a concurrent speculative call slows the real call about as much as a hit saves. The published speedups assume API-served models with effectively unlimited parallel capacity.
- **What *did* reduce latency** (experiment 2):
  - A predictor learned from the actor's own history (60% top-1) with top-1 prefetch: −7.4% decision latency at +107% compute.
  - Prefetching the SSD agent's next fast-policy call during its ~2.2 s rollout phase: −5.8% at +35%, with success unchanged.
  - Pre-executing test runs for a coding agent: −15 to −25% task wall-clock.
  - So speculation helps when the pre-executed work does not compete with the critical path for the same resource, or when the critical path is long enough to hide it.

## 6. Secondary benchmark: DoorKey-8x8 (val, 15 seeds, 60-step cap)
| Method | Success | Steps | Model-s / ep |
|---|---|---|---|
| A greedy 4b | 0.00 | 60 (cap) | 236 |
| L greedy 8b | 0.00 | 60 | 272 |
| J verify on trigger | 0.00 | 60 | 1934 |
| K plan drafting | 0.00 | 60 | 261 |
| **B always-on rollouts** | 0.60 [0.33, 0.87] | 37.4 | 183 |
| **C selective 0.4** | 0.73 [0.47, 0.93] | 34.4 | 1116 |
| I ObsBFS / SimBFS | 0.13 / 1.00 | 56.7 / 17.7 | 0 |

- Long-horizon, key-then-door dependencies defeat both LLMs completely within 60 steps.
- Simulator rollouts with a privileged heuristic rescue most episodes. The improvement comes from the simulator-assisted rollout policy, not from the LLM.
- The first DoorKey job lost 33 episodes to 60 s Ollama timeouts. Those methods were rerun with a 300 s timeout (`val_doorkey_rerun`). Analysis uses the rerun episodes; nothing is mixed per seed.

## 7. Threats to validity
- **Privileged rollouts.** B / C / D / N and the shaped evaluator use the simulator (labeled simulator-assisted). A deployable agent would need a learned or observation-only model; ObsBFS rollouts matched SimBFS in the ablation.
- **The LLM is a weak policy here,** and a non-LLM observation-only planner solves the task. Absolute LLM numbers are prompt-sensitive; prompt v3 was chosen on dev seeds 0–19 (`results/probe-10453.json`).
- **Response cache.** It makes repeated prompts free in wall-clock terms within one run, so latency columns in the main tables are optimistic. Speedup claims use only the cache-disabled runs in §5.
- **Server nondeterminism** at temperature 0 under batching. Lossless methods are lossless only up to it.
- **Episode length couples to compute:** triggers that prevent deaths also lengthen episodes. Model-seconds per episode, not per decision, is the reported cost.
- **Run errors.** Runner errors are listed per run, and failed episodes are retried up to 2× before being recorded. Errors in this report: DoorKey first job (rerun); the ablation `parallel4` arm (pickling bug, rerun as `ablate_parallel`).
- **Statistics.** Bootstrap CIs over 50 val seeds (100 test) are wide for mid-range success rates. Paired differences are in each `summary.md`.

## Reproduce
```bash
uv sync --extra dev && uv run pytest                    # 29 CPU tests
scripts/run_cluster.sh configs/smoke.yaml               # end-to-end on a GPU
DC_SBATCH_ARGS="--gres=gpu:h200:1 -c 32 --export=ALL,DC_MODELS=qwen3:4b\ qwen3:8b,DC_WORKERS=32,OLLAMA_NUM_PARALLEL=32" \
  scripts/run_cluster.sh configs/val_dyn8_core.yaml     # likewise val_dyn8_new, test_dyn8(+_part2,_part3), latency_*, ablate_*
sbatch scripts/analysis.sbatch results/<val_dyn8_core run>   # oracle labels + detector eval (CPU)
report/build.sh                                         # regenerate all tables/figures
```
