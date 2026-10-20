# Cortex v3: the trustworthy loop

Status: **proposed**. It extends [PRODUCT_SPEC.md](PRODUCT_SPEC.md) (v2). [ROADMAP.md](ROADMAP.md) tracks what is built.
The evidence behind it is the October 2026 research report on how labs train robot policies (summarized in
section 1).

## 0. Positioning

Cortex is the autonomous experiment loop for robot policies, built on measurements you can trust. An agent
runs every step that doesn't move hardware: checking data, building datasets, training, evaluating in sim,
choosing what to collect next. A person runs the robot and owns the judgement calls: real trials,
spend, and anything physical.

The product metric is **verified real-robot improvement per human hour**, not "steps automated".

## 1. What the research changed

| Finding | Consequence for Cortex |
|---|---|
| No tool covers the whole loop. The pain sits at four seams: real-robot evaluation, data quality, sim-to-real, and on-robot serving. Formats have mostly settled on LeRobot v3. | Keep LeRobot v3. Build the seams first. |
| Sim demos added to 10–400 real demos give +15 to +40 points on real robots. Planners, replay and LLM-written solvers produce them; an agent driving the arm step by step is the weakest source. | Agent-driven collection stays as a way to bootstrap an unscripted task. Bulk data comes from planners and replay. Datasets mix real and sim on purpose. |
| Teams lose weeks to data problems that never show in the loss: idle frames, wrong normalization stats, camera swaps, configs that don't match the data. | A **readiness check** runs before every training proposal. |
| Real-world evaluation is manual and not reproducible. Sim evaluation tracks real results only when the scene is matched (r ≈ 0.92 on matched scenes vs 0.24 on generic ones). | **Real trials** are first-class data, compared with the sim results for the same checkpoint. Cortex reports how well sim predicts real for each project. |
| Labs accept agents for off-robot work and distrust them on hardware. | Agents may log what an operator reports. Only a person judges a real trial (section 5). |

## 2. New stages

The v2 pipeline (intake → sim → demos → dataset → train → evaluate → improve) gains two gates:

```
demos ──► [readiness] ──► dataset ──► train ──► sim eval ──► [real trials] ──► improve
                                                      └────── sim ↔ real ───────┘
```

### 2.1 Readiness (the data doctor)

`check_dataset_readiness(project_id, episode_ids?|dataset_version_id?)` returns a verdict and fixes.

Per-episode checks (pure numpy, sim and real alike):

| Code | Severity | What it catches |
|---|---|---|
| `idle_head` / `idle_tail` | warn | more than 1 s of no motion at the start or end; the policy learns to wait. Teleop jitter counts as no motion |
| `gripper_unused` | warn | the gripper never opens or closes in a pick task |
| `action_jump` | warn | a single step more than 8× the episode's median step: a teleop glitch or dropped frames |
| `action_state_lag` | warn | actions trail the recorded state by 3+ frames: the streams are out of sync |

Dataset-level checks:

| Code | Severity | What it catches |
|---|---|---|
| `fps_mismatch` | error | episodes recorded at different frame rates |
| `shape_mismatch` | error | state or action dimension differs between episodes |
| `unit_mismatch` | error | one source's joint range is ~57× another's (degrees vs radians), so real and sim can't be mixed as-is |
| `length_outlier` | info | an episode is more than 3 MADs from the median length |
| `few_demos` | warn | fewer demos than the collection plan targets for a condition |
| `mostly_idle_data` | warn | over 40% of human-recorded steps are idle (planner holds are deliberate and not counted) |

Verdict: `ready`, `ready_with_warnings` or `blocked` (any error). `propose_training_job` runs the check and puts
the verdict in its result and rationale. A blocked dataset can still be proposed: the human approver sees why.

### 2.2 Real trials and sim ↔ real

A **real trial** is one rollout of a checkpoint on the physical arm, judged by a person:
`{checkpoint_id, condition, success, failure_mode?, judged_by, notes, source: studio|agent}`.

- The studio logs trials in one click per rollout (✓ / ✗, optional failure note).
- `log_real_trials` lets an agent record what the operator reported, with `judged_by` naming that person.
  Agent-logged trials are labelled as such everywhere.
- `get_sim_real_report(project_id)` returns, per checkpoint and condition: real n, successes, Wilson 95% CI;
  the sim rate on the same condition; the gap. Per checkpoint, the gap and the ranking use only the conditions
  both sides measured. Across checkpoints it returns the Spearman rank correlation of sim vs real success, once 3 or more checkpoints have both. Below that it says there isn't enough data
  instead of guessing.
- `get_next_actions` suggests real trials for a checkpoint that passed sim evaluation but has none.

### 2.3 Later (not in this round)

In ranked order from the research:
1. Co-training mixes: real-to-sim ratio as a dataset option, real oversampling.
2. Pretrained VLA fine-tuning (π0/π0.5, GR00T) on managed cloud GPUs.
3. Agent-written planner constraints that expand a few real demos (MimicGen-style) instead of step-by-step arm control.
4. Cross-run experiment comparison, then deployment serving and an intervention-driven data flywheel.

## 3. Studio

- **Improve › Datasets:** each version shows a readiness badge (Ready / Warnings / Blocked). Expanding it
  lists the findings, one line each, with the episodes involved.
- **Improve › Evaluations:** a **Real robot** section per checkpoint: a trial logger (condition, ✓ / ✗), the
  real rate with its interval next to the sim rate, and one line for sim ↔ real agreement.
- Copy follows the v2 rules: short, measured values over adjectives, no raw error text.

## 4. MCP and REST

| Tool | REST | Writes |
|---|---|---|
| `check_dataset_readiness` | `GET /api/datasets/{id}/readiness`, `POST /api/projects/{pid}/readiness` | no |
| `log_real_trials` | `POST /api/projects/{pid}/real-trials` | yes |
| `get_sim_real_report` | `GET /api/projects/{pid}/sim-real` | no |

## 5. Safety invariants (additions)

1. A real trial's success is a person's judgement. An agent can record it only as a report from a named
   person, and the record keeps `source: agent`.
2. Readiness never edits or deletes data. It reports; people and agents decide what to exclude.
3. Nothing in v3 moves hardware. The existing gates of v2 section 5 are unchanged.
