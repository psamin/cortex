# Cortex product spec v2: agent-driven policy training for your arm

Status: **proposed**. This document is the target. [ROADMAP.md](ROADMAP.md) tracks what is built.
Where this spec and the code disagree, the code describes today and this spec describes where it is going.

## 1. One sentence

Any agent connects to Cortex over MCP, describes the arm and the objects in front of it, names a task,
and Cortex returns a policy tuned for that arm and task. The loop: simulation first, then supervised
real-world demonstrations that GPT-6 Astra drives and resets itself, then targeted demos or RL until
the policy meets its target.

## 2. Users

| User | What they do |
|---|---|
| **Calling agent** (Claude Code, any MCP client) | The primary user. Runs intake, starts stages, reads results, asks a human for approval when Cortex requires it. |
| **Operator** (a human at the arm) | Arms the hardware, keeps a hand near the e-stop, confirms real-world runs, spot-checks success verdicts. Cortex never moves a real arm without one. |
| **Studio viewer** (a human in the browser) | Watches progress, approves training and real-robot stages, reviews diagnoses. |

## 3. The pipeline

```
 0 INTAKE ──► 1 SIM BUILD ──► 2 SIM STAGE ──► 3 REAL STAGE ──► 4 REAL EVAL ──► 5 IMPROVE ──► 6 DELIVER
 (agent      (scene from      (Astra cyclic   (Astra cyclic    (policy rollouts  (more demos      (policy +
  describes   the agent's      demos in sim,   demos on the     on the arm,       or RL, chosen    model card)
  arm, items, description,     ACT pretrain,   real arm, fine-  Astra resets      by evidence)        │
  task)       reach check)     sim eval)       tune on real)    between trials)       │               │
                                                                     ▲                │               │
                                                                     └────────────────┘               │
```

Every stage is a step of the Autopilot state machine. It can be started, stopped and resumed, and it
writes its evidence to the lineage graph.

### 3.0 Intake (MCP-first)

The calling agent does not need to know Cortex's schema up front. It calls `cortex.start_intake`, and
Cortex answers with the questions it still needs. The agent answers with whatever it knows, and Cortex
validates the answers and returns the remaining questions, until the spec is complete.

| Spec | Fields | Notes |
|---|---|---|
| `RobotProfile` | arm model (`so101` now; registry for more), serial port / connection, joint limits, **home pose** ("position 0"), gripper range, max joint speed, calibration file | Unknown arms are refused with the list of supported models. Joint limits default to the model's and can only be narrowed. |
| `Cameras` | role (`overhead`, `wrist`, `front`), device index or URL, resolution, extrinsics (or "calibrate for me") | Astra needs one overhead or angled view of the table. A camera without extrinsics gets a calibration step before the real stage. |
| `Workspace` | table bounds, forbidden zones | Defaults come from the arm's reach. |
| `ObjectSpec[]` | name, shape primitive (`box`, `cylinder`, `sphere`), size, mass, colour, graspable | A water bottle is a `cylinder`, radius 0.033 m, height 0.20 m, 0.5 kg. |
| `PlaceZone[]` | name, centre and size on the table | "Spot A", "the shelf corner". |
| `TaskSpec v2` | pick object, place zone(s), **reset policy** (`return_to_origin` / `random_zone` / `cycle_zones`), success target, demo budget, money and GPU limits | |

Output: a validated `ProjectSpec`, which also contains the questions Cortex could not resolve itself
(for example "the bottle (0.5 kg) is above the SO-101's rated payload of 0.25 kg; continue?").

### 3.1 Sim build

- Generate a MuJoCo scene from the spec: the pinned EnvHub table, room, cameras and arm, plus the
  agent's objects as primitives and the place zones as flat markers. The upstream env takes a `scene`
  path, so Cortex never forks it.
- **Reachability check**: IK to every object start region and every zone, at grasp and carry height.
  Unreachable zones are returned to the agent as an intake error, before any compute is spent.

### 3.2 The cyclic demonstration protocol (sim and real)

A *cycle* is two segments. Only the first one is training data.

```
 DEMO segment (recorded as a LeRobot episode)       RESET segment (recorded as a reset trace)
 home ─► observe ─► grasp object ─► place in zone ─► home ─► grasp object ─► place at origin or new spot ─► home
```

- Both segments are driven by GPT-6 Astra through the validated tool API. Astra picks targets and the
  local IK controller executes them. On real hardware, the safety controller vets every tick.
- **Episode boundaries are home-to-home.** That gives the policy a consistent start and end state, which
  matches how it will be run.
- **Success comes from a verifier, never from the model's claim.**
  - In sim it is the simulator's object pose.
  - On a real arm it is the *perception verifier*: the object is segmented in the overhead camera inside
    the target zone, the gripper reports open and empty, and the arm reached home.
  - A failed demo segment is a diagnostic, not an episode. A failed reset segment pauses the loop and
    asks the operator to reset by hand.
- **The reset policy decides the next start.**
  - `return_to_origin` repeats the same start state.
  - `random_zone` and `cycle_zones` move the start around, which gives positional diversity for free.
- The same reset segment is reused between **policy evaluation trials** in 3.4. Real evaluation needs no
  human reset either.

### 3.3 Stages and gates

| Stage | Done when | Gate to enter |
|---|---|---|
| Sim stage | ACT trained on sim demos reaches `sim_target` (default 0.8) on held-out sim seeds | Spec complete, reach check passed |
| Real stage | `real_demo_target` verified real demos collected (default 30) | Hardware armed **and** operator present **and** the studio (or operator token) approved the real stage |
| Real eval | `n` rollouts per zone with Wilson CIs | Same as real stage |
| Improve | Policy meets `real_target`, or the iteration or money budget is spent | Training approval rules from v1 still apply |

The sim-pretrained checkpoint is fine-tuned on real demos. Sim demos stay in the mix, labelled by
source, so the dataset lineage shows exactly which demos trained what.

### 3.4 Improve: more demos or RL

After each evaluation, Cortex picks one action from the evidence and records why:

| Evidence | Action |
|---|---|
| Failures concentrated in a start region or zone that has few demos (coverage gap) | **Targeted demos** there: real if the failure shows on the real arm, else sim |
| Failures spread out with near misses (grasp offset, release too early) and enough demos | **RL refinement in sim**: a residual policy on top of the ACT checkpoint, trained with PPO against the shaped reward, under domain randomization of the scene the agent described. Re-evaluated in sim, then on the real arm. |
| Lighting or appearance failures | Augmentation (v1) |
| Target met, or budget spent | Stop and deliver |

RL never runs on the real arm. RL results count only once they are verified on the real arm.

### 3.5 Deliver

`cortex.get_policy` returns:
- the checkpoint (LeRobot `pretrained_model/`);
- a **model card**: robot profile, task, scene, data provenance (sim, real, agent, oracle), training
  runs, success rates in sim and on the real arm with CIs, known weak conditions;
- a run command for `lerobot` on the agent's own arm.

## 4. MCP surface (added in v2)

| Tool | Purpose |
|---|---|
| `start_intake(prompt?)` | Creates a draft project; returns `{project_id, questions[]}` |
| `describe_robot(project_id, robot)` | Sets `RobotProfile` and cameras; returns remaining questions |
| `describe_workspace(project_id, objects, zones, workspace?)` | Sets objects and zones |
| `set_task(project_id, pick, place, reset_policy, targets…)` | Sets TaskSpec v2 |
| `get_intake_status(project_id)` | `{complete, questions[], spec}` |
| `build_simulation(project_id)` | Scene generation and reach check; returns a preview image key |
| `start_cyclic_collection(project_id, backend: sim\|real, cycles)` | Starts the 3.2 protocol |
| `start_rl_refinement(project_id, checkpoint_id, budget)` | Proposes RL refinement (approval rules apply) |
| `get_policy(project_id)` | Best policy, model card, run command |

All existing v1 tools stay. Real-robot tools return `approval_required` with a studio link, and are
always refused for remote callers.

## 5. Safety invariants (unchanged from v1, extended)

1. Astra never emits motor commands. Every target goes through validation, IK and, on hardware, the
   per-tick safety controller: joint limits, velocity, workspace, watchdog, e-stop.
2. Real motion requires `CORTEX_ENABLE_HARDWARE=1`, a local caller, an operator-confirmed session and
   a calibration file. Unattended real mode is refused.
3. The watchdog trips on a stale command stream and holds position. It is not fed by the loop that
   checks it.
4. Training, real-robot stages and paid jobs need human approval or a standing policy the human set.

## 6. Non-goals for v2

- Arms other than SO-101 are not shipped. The `RobotProfile` registry is the extension point.
- No mesh import for objects (primitives only), no deformables.
- No RL on hardware.
