# Learning to Speculate: graph-based predictive execution (experiment 2): design

Results are in [REPORT.md](REPORT.md). Code lives in `src/dreamcache/specgraph/` (Track A) and `src/dreamcache/codebench/` (Track B).

## Decision speculation vs. predictive execution
| | Decision speculation (SSD, experiment 1) | Predictive execution (this experiment) |
|---|---|---|
| What runs early | counterfactual futures for *different* candidate actions | the *same* work the agent will need next, before it asks for it |
| Changes the chosen action? | yes, that's the point | never; results are reused only after validation, otherwise recomputed |
| Goal | decision quality | critical-path latency at equal outcomes |
| Code | `speculation/`, `agents/agent.py` | `specgraph/prefetch.py`, `codebench/speculator.py` |

## Track A: MiniGrid. The pre-executed work is the actor's next model call
- At step t, a frozen predictor estimates P(action the agent executes | observation-only context, previous action). The context key is `specgraph/graph.py:context_key`: front/left/right cell types, goal direction bucket, nearby-obstacle bucket, and the last action.
- A scheduler picks guesses. Each guess is applied in a simulator clone that shares the real RNG, which reproduces the exact next observation if the guess is right. The actor's `decide()` for that state is then submitted to a worker pool.
- Reuse requires the sha256 of the exact prompt text, plus a model and prompt fingerprint, to match. Otherwise the result is discarded (`unused`, counted as wasted compute) and the call runs normally.
- **Caveat:** the clone needs simulator access to produce the predicted next observation. In real agent systems this corresponds to a side-effect-free tool or environment API. The actor's choices never depend on it, so it can change latency but not outcomes, up to server nondeterminism, which is measured.

## Track B: coding agent. The pre-executed work is tools (pytest, ruff, import check)
- **Tasks:** 30 bug variants (10 function templates × 3 mutations). Each appears in a train package and a disjoint eval package with different companion functions.
- **Agent:** qwen3:4b chooses every operation from {read_file, edit, run_tests, lint, deps_check, finish}. Edits change the real workspace.
- **Speculation:** while the LLM is thinking about snapshot S, the speculator may run predicted *read-only* tools on S in an isolated copy (`tools.isolated_copy`). A result is keyed by (op, snapshot sha256, tool fingerprint) and reused only on exact match. Nothing is ever run against a snapshot that doesn't exist yet.
- **Paired timing:** a reference pass makes the real LLM calls. Every policy is then replayed with the cached responses, and the agent *sleeps for the recorded server time* of each call. All policies therefore see identical decisions and identical think durations; only tool scheduling differs. `identical_op_sequence_rate` verifies this.

## Execution graph (two levels)
- **Level 1 (`workflow_edge`):** operation class → operation class counts, per workflow (actor identity) and environment.
- **Level 2 (`exec_state`):** one row per concrete operation, with workflow, env, episode, t, normalized context, previous op, op, input hash, env version, duration, speculative / reused / validated flags, and cost.
- **Storage:** SQLite (`results/specgraph/graph.sqlite`). Raw observations are never stored.

## Predictors (`specgraph/predictors.py`, `codebench/speculator.py:OpPredictor`)
- Global frequency P(op).
- Workflow-graph transition P(op | prev_op).
- Context-conditioned P(op | context), backing off to P(op | prev_op), then to P(op).
- Personalized shrinkage: (1−a)·P_global(op | ctx) + a·P_local(op | ctx, env × workflow), with a = n_local / (n_local + τ) and τ=20 (Track A) or 10 (Track B).
- Observation-only BFS heuristic as a non-learned point predictor.
- All counts use additive smoothing. Training uses only train-split trajectories; models are frozen before held-out evaluation.

## Schedulers (`specgraph/scheduler.py`; Track B equivalents in `codebench/speculator.py`)
- **Policies:** none, random, top-1, top-K, p×latency, utility, and utility_adaptive.
- **Utility:** U(o) = p(o)·T − λ·T − κ·T·n_inflight, where T is the EMA of actor-call latency and the κ term is a contention penalty.
- **Constraints:** a maximum number of in-flight workers and a budget ratio (speculative calls per real step).
- Track B also has `memo` (memoize identical requests, no prediction) and `all` (run every eligible tool for every new snapshot, which is ordinary parallelism). Together they separate prediction from plain parallelism (ablation J).

## Variants (as specified; N/A stated where not meaningful)
| ID | Track A (MiniGrid) | Track B (code) |
|---|---|---|
| B0 | 8b actor, no speculation | `none` |
| B1 | original selective SSD (4b, threshold 0.4) | N/A (no decision rollouts in Track B) |
| B2 | predictor runs, nothing executed (overhead only) | `memo` (closest analogue) |
| B3 | frequency predictor, top-1 | `top1_frequency` |
| B4 | global context predictor, top-1 | `top1_global` |
| B5 | personalized, top-1 | `top1_personalized` |
| B6 | personalized + utility scheduler (and adaptive) | `utility_personalized` |
| B7 | SSD with rollout candidates ranked by the graph predictor (`spec.candidates: graph`, `graph_rank:` in the config) | N/A |
| B8 | SSD + predictive prefetch of the 4b actor's next call | N/A |
| extra | top-K=3 (all actions, like Speculative Actions K=3), random scheduler, heuristic predictor | `all`, `random`, `topk2` |
