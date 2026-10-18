# RL sandboxes: design (#80)

Status: **validated against current research** (see [research/RL_VALIDATION.md](research/RL_VALIDATION.md), about 35 papers).
Design changes from the first draft are marked **(changed)**.

## Where it sits in the product loop

```
 evaluate policy ─► weaknesses ─► propose sandboxes ─► agent accepts ─► sandboxes run (live) ─► targeted demos
     ▲            (measured)     (predicted uplift,    (MCP, within        (CEM / residual /        (+ curation filter)
     │                            cost, P(uplift≥δ))    standing budget)     re-targeting)               │
     │                                                                                                  ▼
  log predicted vs realised ◄── paired re-evaluation ◄── human-approved retrain ◄── new dataset version
  uplift → refit the transfer model   (all conditions, McNemar)                    (provenance: sandbox, params, seeds)
```

- **This is the same chain as `improve_policy`:** targeted demos → dataset version → approval-gated retrain. Sandboxes are one more *source* of targeted demos, next to agent or teleop collection and imports.
- **Only a re-evaluated ACT checkpoint counts as a measured improvement of the policy.**

## 1. Weaknesses

Weaknesses come from the latest **learned-checkpoint** evaluation; the scripted expert never counts. For each condition:
- success with its Wilson 95% interval, and n;
- the failure-mode breakdown measured from sim telemetry;
- the gap to target.

**(changed) Ranking:** the score uses the gap at the lower Wilson bound, `max(0, target − ci95_hi)` weighted by confidence, so a noisy condition is neither over- nor under-ranked.

**(changed) Clusters:** failures are also grouped by **initial-state region**, meaning object pose bins (Mulligan, 2610.05882), and by **failure mode** (DISEIL, 2609.08123). VLM or Claude Code hypotheses (#70) attach to clusters as **unverified** labels.

## 2. Sandbox methods

| Method | When | Compute | Output |
|---|---|---|---|
| **`cem_skill`** (default) | The scripted skill can do the task in that condition (pilot ≥ 30%) | Local CPU | Targeted demos |
| **`mimicgen`** (changed: new) | The skill's pilot is under 30%. Successful agent or human teleop demos exist | Local CPU | Re-targeted demos to the weak condition's object poses. Each one passes a sim success check |
| **`residual_rl`** (changed: new, advanced) | A GPU is selected (Futurama) and a trained ACT exists | GPU | A deployable ACT+residual checkpoint (evaluated like any other) **or** distillation demos |

- **`cem_skill`** (RAPS / MAPLE precedent; PA-RL 2412.06685 is the closest "improve, then distil" analogue):
  - CEM over 5–9 corrections to the validated IK pick-and-place skill: grasp offset, depth, yaw, close timing, carry and release height. The parameter set and return weights are chosen by the dominant failure mode.
  - Population 32, 20% elites, at most 15 iterations, smoothing plus a std floor, common random numbers per iteration.
  - Every action still goes through IK validation.
  - The skill uses privileged sim state, so it is labelled an oracle.
- **`residual_rl`** follows the ResiP (2407.16677), ResFiT (2509.19301) and Policy Decorator (2412.13630) recipe:
  - frozen ACT, plus a small Gaussian residual on the executed chunk;
  - an SAC-style off-policy learner with a privileged-state (asymmetric) critic;
  - a sparse reward plus the failure-mode term;
  - a residual clamp of 2 cm or less and 10° or less per step.

  PAC-ACT (2607.09590) is listed as experimental. DSRL and DPPO are diffusion-specific, so they're not offered.
- **(changed) Recovery-rich demos.** Near-perfect scripted demos teach ACT no recovery.
  - **DART-style noise:** 10–30% of tuned-skill demos use perturbed waypoints, and are kept only if successful.
  - **Optional DAgger relabel:** roll out the current ACT in the weak condition, then label those states with the tuned skill's re-planned action. This needs a skill that re-plans from arbitrary states.

## 3. Sandbox environment

Domain randomization is centred on the weak condition: object offset and yaw, light scale, and friction and mass of the task object. It uses the agent's intake scene when one is built (#50/#51).

**(changed) ADR-style widening:** when tuned-skill success inside the current ranges exceeds 80%, the ranges widen. This repeats until the condition's neighbourhood is covered, and the final ranges are stored in the sandbox manifest.

## 4. Estimates before accepting (changed)

The first draft's k/t formula has no literature basis. It's kept only as an **explicitly labelled prior**.

- **Pilot (measured).** n base-skill episodes in the sandbox, plus a **short CEM pilot** (2–3 iterations). Together they give `predicted_sandbox_success`, a Beta posterior over success, with `source: measured`.
- **Policy uplift.** This comes from a **per-project, per-condition saturating transfer curve** fitted on past rounds' (predicted → realised) pairs, with a bootstrap interval. The template is Factored Scaling Curves (2505.07728).
  - Until 3 or more rounds exist, it uses the prior (k ∈ [0.3, 0.7], t ∈ [0.3, 0.6]) with `source: prior`.
- **Also returned:** `predicted_regression_elsewhere` (the change on other conditions), `p_improve_ge_delta` (P(uplift ≥ δ)) and `uplift_per_hour`.
- **Cost (measured throughput).** Episodes, CPU hours, GPU hours, wall-clock estimate and $. Local runs cost $0.

## 5. MCP surface

| Tool | Returns |
|---|---|
| `get_policy_weaknesses(project_id, checkpoint_id?)` | Ranked weaknesses: Wilson intervals, failure modes, initial-state clusters, unverified hypotheses |
| `propose_rl_sandboxes(project_id, budget?)` | Per sandbox: `weakness_id`, `method`, `pilot{n, successes, wilson_95}`, `predicted_sandbox_success{mean, ci90, source}`, `predicted_policy_uplift{mean, ci90, source}`, `predicted_regression_elsewhere`, `cost{episodes, cpu_hours, gpu_hours, wall_clock_est, usd}`, `uplift_per_hour`, `p_improve_ge_delta`, `assumptions[]` |
| `accept_rl_sandboxes(project_id, ids, budget_cap)` | Spawns the sandboxes. Agents may accept local CPU work within the project's standing budget; GPU or paid work needs a human approval |
| `get_rl_sandbox_status(project_id \| sandbox_id)` | Live: iteration, episodes, success with interval, elite parameters, current ranges, elapsed time, cost so far |
| `stop_rl_sandbox(sandbox_id)` | Stops a sandbox; it reports why it stopped |

## 6. Stopping rules (changed)

- **CEM** stops on any of:
  - the elite success **lower** Wilson bound reaches the target;
  - less than 2 pp improvement over 3 iterations;
  - the budget is exhausted.
- **Demo generation** stops at N demos (default 20–50 per condition). Targeted data is capped at **30–40% of the dataset**, to limit interference with the other conditions.
- **The round** stops if the re-evaluated ACT fails to improve the weak condition in 2 consecutive rounds (paired test). The condition is then flagged for human or teleop demos.

## 7. Evaluation protocol (changed)

- Base vs tuned skill, and old vs new ACT, run on **matched held-out seeds**, compared with **McNemar's test**.
- **STEP-style sequential stopping** (2503.10966), capped at about 200 episodes per condition.
  - For scale: detecting 0.60 → 0.75 needs about 150 episodes per arm, and 0.80 → 0.90 about 200.
- **All conditions** are evaluated, to catch regressions.

## 8. Live stats and visualisation

- **Every sim worker** (collection, evaluation, sandboxes) streams compact stats over SSE:
  - per step: phase, joints, TCP and return so far;
  - per episode: success, failure mode, return;
  - running totals: success with a Wilson band, episodes/s and sim fps.
- **A sandbox card** in the studio, in the Kinetic style (#52), shows:
  - an episode thumbnail grid;
  - a success-vs-iteration line with its Wilson band;
  - the CEM elite-parameter chart;
  - a failure-mode stacked bar and a budget meter.
- **The round view** shows predicted vs realised uplift.

## PR plan

1. Weaknesses and proposals: **merged** (#82). PR 2 adds the Wilson-lower-bound ranking and the initial-state clusters.
2. The sandbox engine:
   - `skill`, `sandbox_env` (DR plus ADR widening), `learner` (CEM with the settings above);
   - `workers/rl_worker.py`: pilot, train, held-out evaluation on matched seeds with McNemar, and DART-noised demo generation, streaming live stats.
3. `services/rl_service.py`, a migration for `rl_sandboxes` and `rl_rounds`, and the MCP tools above, with the Beta and prior estimate fields.
4. Closing the loop: demos into a dataset version (with the cap and provenance), the approval-gated retrain, paired re-evaluation of all conditions, logging predicted vs realised, and refitting the transfer curve.
5. Live sim stats (SSE) and the studio sandbox cards.
6. Later: `mimicgen` re-targeting, `residual_rl` on Futurama, and the DAgger relabel pass.
