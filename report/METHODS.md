# DreamCache: methods and design

This file covers the parts that don't depend on results. Results are in [REPORT.md](REPORT.md).

## Environment and access levels
- **Benchmarks:** MiniGrid `Dynamic-Obstacles-8x8-v0` (primary) and `DoorKey-8x8-v0` (secondary). `Dynamic-Obstacles-16x16` is open as tracking package P1.
- **Deployable inputs:** the fast policy and the difficulty detector see only the 7×7 egocentric observation, the mission string, the facing direction, and the agent's own recent actions. Rendering is in `environments/observations.py`.
- **Simulator-assisted:** sandbox rollouts, UCT-MCTS, the shaped evaluator's progress term, the SimBFS rollout policy, and the lossless-speculation clones all use the full simulator state. These are labeled as such.
- **Oracle (not deployable):** `Q_ref` uses many privileged rollouts. Its compute is excluded from deployment cost.

## Cloning
- Clones are `copy.deepcopy(env.unwrapped)`. Tests verify that a clone preserves agent position and direction, the grid, carried objects, moving obstacles (as independent objects), step count, and RNG state. They also verify that identical actions from a clone produce identical transitions, and that sandbox steps never mutate the parent. See `tests/test_cloning.py`.
- **Decision sandboxes are reseeded** with fresh RNG streams. They therefore never see the real environment's future obstacle motion. Common random numbers are shared across candidates within one decision.
- **Lossless prefetch clones are not reseeded.** They exist only to reproduce the exact next state if the actor takes the guessed action. They never influence which action is chosen.

## Fast policy (LLM)
- **Serving:** qwen3:4b (small) and qwen3:8b (big) via the Ollama `/api/chat` endpoint, `think: false`, JSON-schema structured output validated with Pydantic, temperature 0 for decisions.
- **Prompt:** v3, chosen on dev seeds with `scripts/prompt_probe.py`. It gives the environment rules, an ASCII view, the visible objects with relative coordinates, and the front/left/right cells, and asks for a ≤25-word reason followed by 0–10 scores for every legal action.
- **Scores:** normalized and treated as uncalibrated. The greedy action is the argmax.
- **Stochastic proposals:** 3 samples at temperature 1.0, used for the disagreement signal.

## Difficulty detector
`difficulty = (α·H_norm + β·disagreement + γ·risk) / (α+β+γ)`, with default weights 0.4 / 0.3 / 0.3.
- **H_norm:** entropy of the normalized scores divided by log(number of legal actions).
- **Disagreement:** 1 − mode share among the greedy action plus the 3 proposals, rescaled to [0,1].
- **Risk:** observable only. Risk is 1.0 if the chosen action is `forward` into a visible wall or obstacle. Otherwise it is driven by nearby visible obstacles and the number of safe exits.

## Speculative executor (decision speculation, SSD)
- **Steps:** K candidates (top-K by LLM score, fast action first) × M=4 sampled futures each. Each future clones the simulator, applies the candidate, then runs H steps of the rollout policy (default: the SimBFS heuristic) and is scored.
- **Score:** shaped utility = 10·success − 10·collision − 0.01·steps + env reward + progress. Progress is a phase-consistent remaining-plan distance (agent→key→door→goal), plus one-off key and door bonuses.
- **Selection:** the action with the highest mean utility. Ties go to the fast action.

## Conditions
| ID | Name | Description |
|---|---|---|
| A | greedy | qwen3:4b argmax, no search |
| L | greedy 8b | stronger model, no search |
| B | always-on | speculate at every step |
| C | selective | speculate when difficulty ≥ threshold (0.2 / 0.4 / 0.6 / 0.8; entropy-only and risk-only variants) |
| D | oracle-triggered | speculate when Q_ref regret of the fast action > δ=0.2 (8 samples). Not deployable. |
| E / F | random / periodic triggers | controls at matched rates |
| G | best-of-K, no rollouts | candidate score minus observable risk |
| I | heuristic planners | SimBFS (privileged) and ObsBFS (observation-only) |
| J | draft → verify | 4b proposes, 8b accepts or overrides (always, or on trigger) |
| O | cascade | on trigger, 8b decides from scratch (FrugalGPT-style routing) |
| M | self-consistency | majority vote over 5 samples, on trigger |
| N | UCT-MCTS | the `mcts` package (pbsinclair42, MIT), on trigger or always |
| K | speculative plan drafting | 4b drafts L=4 actions and 8b reports an accepted prefix length. Not lossless. |
| K2 | lossless speculative drafting | after DSP / ISP. 8b decides on every drafted state in parallel; accept the prefix where draft == 8b argmax, plus 8b's action at the first mismatch (greedy speculative decoding) |
| S | SwiftSage trigger | ported fast/slow rule; slow = 8b |
| LS | lossless speculative actions | after Speculative Actions, 2025. The actor is 8b. Guessers (ObsBFS heuristic, 4b, or both) predict the action; the actor's next call is prefetched on the predicted next state. Targets latency only. |

## Compute accounting
| Quantity | What it measures |
|---|---|
| `llm_seconds` | model-server time summed over calls. This is the GPU-work proxy. Cached calls are charged their original cost. |
| `sim_cpu_seconds` | process CPU time spent in sandboxes |
| `wall_seconds` | end-to-end decision latency, measured |
| `compute_seconds` | `llm_seconds` + `sim_cpu_seconds` |

Latency claims come **only** from runs with the response cache disabled and identical worker counts (`configs/latency_dyn8.yaml`, `configs/specgraph_test_dyn8.yaml`). In the main runs, the shared cache makes wall-clock latency optimistic for methods whose prompts repeat across methods.

## Statistics
- Seeds are paired across methods.
- Means are reported with 95% bootstrap CIs (10k resamples).
- Paired bootstrap differences are reported with centered-bootstrap two-sided p-values.

## Seed splits
| Split | Seeds | Use |
|---|---|---|
| dev | 0–19 | prompt and model probing |
| val | 1000–1049 | all selection |
| test | 10000–10099 | final numbers; config frozen in commit `d4b646d` before the run |
