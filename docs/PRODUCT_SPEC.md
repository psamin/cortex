# Cortex product spec v2: agent-driven policy training for your arm

Status: **proposed**. This document is the target; [ROADMAP.md](ROADMAP.md) tracks what is built.
Where this spec and the code disagree, the code describes today and this spec describes where it is going.

## 1. One sentence

Any agent connects to Cortex over MCP, describes the arm and the objects in front of it, names a task,
and Cortex returns a policy tuned for that arm and task. It gets there in four steps:
1. Simulation first.
2. Rounds of supervised real-world demonstrations that GPT-6 Astra drives and resets itself.
3. Fine-tuning after each round.
4. Targeted demos or RL until the policy meets its target.

## 1a. MVP scope: everything runs in physics sim

The MVP keeps the whole pipeline in MuJoCo. Stage 3, "real rounds", runs against a **physics-sim
twin** of the agent's scene instead of hardware:

- **Stage 2 (sim pretrain).** Fast, privileged scripted demos under broad domain randomization, used
  to pretrain ACT.
- **Stage 3 (twin rounds).** GPT-6 Astra drives the arm in the twin through the full cycle protocol
  (3.2): home → pick → place → home → reset → home. The rules mirror a real arm:
  - **Camera mode only.** Astra sees what a real camera would see and never reads simulator poses.
  - **Held-out randomization.** The twin uses seeds and physics parameters (friction, mass, lighting,
    camera pose) that the pretrain never sees. It stands in for the gap between sim and real.
  - **Success from the perception verifier.** The verifier reads only the overhead image and the
    gripper, not the simulator's state. The simulator's ground truth is logged next to it, so the
    verifier's accuracy is measured before the verifier ever meets hardware.
- **Rounds.** Each round fine-tunes the latest checkpoint and runs a short twin evaluation, until the
  target is met or the budget runs out.
- **The agent describes the scene and the environment.** That covers objects, place zones, the table
  (size and colour), lighting, static obstacles and camera placement, all through `update_intake`.
  Cortex builds the MuJoCo scene from that description. The agent then calls `capture_frame` to look
  at the render and patches anything that doesn't match.
- **The twin's gap is structural, not just new seeds.** On top of the scene, the twin adds:
  - 1–3 ticks of action and observation latency;
  - PD-gain mismatch, gripper backlash and compliance;
  - camera intrinsics jitter, sensor noise and JPEG compression;
  - unseen textures and distractor objects.

  Latency and actuator dynamics are the biggest real-world gaps, so the twin models them.
- **Three randomization sets.** Set A is used for pretraining, set B for the rounds, and a held-out
  set C only for the final score and the model card.
- **Labels the way a real arm would see them.** Twin training data is gated on the verifier's verdict,
  not on simulator ground truth. Ground truth is logged anyway, so the model card can report the
  verifier's confusion matrix.
- **A robot backend interface from day one.** The twin implements the same `RobotBackend` interface
  that the hardware backend will. Moving to a real arm then means swapping one class.
- **Running twin rounds without Astra.** A camera-mode scripted oracle with pixel-pointing noise can
  drive twin rounds, labelled as such, while the Astra route is blocked.

The hardware track (sections 3.1 calibration, 3.3 real gates and 5.2–5.3, plus roadmap P7–P9) is
**post-MVP**, kept on separate modules and branches. Nothing in the MVP path imports it. The intake
still accepts `mode: "real"` so agents can describe hardware, but in the MVP, real-mode stages answer
"not available yet".

## 2. Users

| User | Role |
|---|---|
| **Calling agent** (Claude Code, any MCP client) | The primary user. Runs intake, starts stages, reads results, and asks a human when Cortex requires approval. |
| **Operator** (a human at the arm) | Arms the hardware and holds the deadman (section 5). Confirms real-world runs, spot-checks success verdicts, and resets by hand when recovery fails. Cortex never moves a real arm without one. |
| **Studio viewer** (a human in the browser) | Watches progress, approves training and real-robot stages, and reviews diagnoses. |

## 3. The pipeline

```
 0 INTAKE ─► 1 SIM BUILD ─► 2 SIM STAGE ─► 3 REAL ROUNDS ──────────────────► 4 IMPROVE ─► 5 DELIVER
 agent       scene from     Astra cyclic    ┌► K real cyclic demos ─┐         more demos,   policy +
 describes   the spec,      demos with      │                       ▼         RL-distilled  model card
 arm, items, feasibility    domain random-  │  fine-tune ◄──── short real      demos, or
 task;       + reach        ization, ACT    └── until target ◄── eval          augmentation
 calibrates  check          pretrain, eval
 cameras
```

Every stage is a step of the Autopilot state machine. It can be started, stopped and resumed, and it
writes its evidence to the lineage graph.

### 3.0 Intake (MCP-first)

The calling agent does not need to know Cortex's schema, and most agents do not know sizes, extrinsics
or a home pose. Intake is built around that:

- **One idempotent tool.** `update_intake(project_id, patch)` merges a partial spec. Every response
  returns the merged spec plus typed **questions** `{path, type, options, default, why}`, so the agent
  only answers what is still open.
- **Names, not numbers.** The agent can say "water bottle". Cortex maps it to primitive defaults from a
  catalogue (shape, size, mass, colour), marks those values `assumed`, and asks the agent to confirm them.
- **Look at the table.** `capture_frame(camera)` returns an image. In sim that is a render; on the real
  arm it is the configured camera. The agent can then propose objects and zones from what it sees, and
  Cortex segments the image to suggest object positions.
- **Read the arm, don't ask.** Joint limits, home pose and gripper range come from the arm model and,
  for a real arm, from its LeRobot calibration file. Those values can only be narrowed, never widened.

| Spec | Fields |
|---|---|
| `RobotProfile` | Arm model (`so101` today; registry for more), connection port, calibration file, joint limits, **home pose** ("position 0"), gripper range, max real-world speed |
| `CameraSpec[]` | Role (`overhead`, `front`, `wrist`), device or URL, resolution, extrinsics (from calibration, section 3.1) |
| `Workspace` | Table bounds and forbidden zones; defaults come from the arm's reach |
| `ObjectSpec[]` | Name, primitive (`box`, `cylinder`, `sphere`), size, mass, colour; each value tagged `given` or `assumed` |
| `PlaceZone[]` | Name, centre and size on the table |
| `TaskSpec v2` | Pick object; place zone(s); **reset policy** (`return_to_origin`, `random_zone` or `cycle_zones`); success targets (sim and real, configurable); demo and round sizes; money and GPU limits |

**Feasibility is checked, not asked.** Cortex measures each object against the arm model:
- **Payload:** the static gravity torque at the worst carry pose must stay under 70% of each actuator's
  force limit. The SO-101 sim actuators are limited to ±2.94 N·m.
- **Grasp:** the object's grasp width must be smaller than the measured jaw opening, with a margin.
- **Stability:** a tall object (height over 2.5× its base) gets an upright check on placement and needs
  extra zone clearance.

An infeasible object is **refused** with the reason and the nearest feasible alternative ("a 500 ml
bottle, 0.5 kg, 65 mm wide, exceeds the jaw opening; a 250 ml bottle fits"). Cortex never asks the
agent to "continue anyway".

### 3.1 Sim build and calibration

- **Scene.** Cortex generates a MuJoCo scene from the spec: the pinned EnvHub table, room, cameras and
  arm, plus the agent's objects as primitives and the place zones as flat markers. The upstream env takes
  a `scene` path, so Cortex never forks it.
- **Reach check.** IK to every object start region and every zone, at grasp and carry height. The result
  is reported per object and per zone, with the nearest reachable alternative for anything that fails.
- **Camera calibration (real arms).** A board seen by the overhead camera gives its extrinsics (an
  ArUco board is the default; checkerboard also works). The sim camera is then placed at the calibrated
  pose, so sim images look like real ones. Astra's pixel pointing and the real verifier both use the
  same calibration.
- **Arm calibration.** Joint offsets and the gripper range from the LeRobot calibration file are applied
  to the sim robot.

### 3.2 The cyclic demonstration protocol (sim and real)

A *cycle* is a demo segment followed by a reset segment.

```
 DEMO  (segment=demo)    home ─► observe ─► grasp object ─► place in target zone ─► home
 RESET (segment=reset)   home ─► grasp object ─► place at origin or next start ─► home
 RECOVER (on failure)    locate object by perception ─► grasp ─► place at a valid start ─► home
```

- **Who drives.** By default GPT-6 Astra drives both segments through the validated tool API: it picks
  targets and the local IK controller executes them. On hardware, the safety controller also vets every
  control tick. The other demo sources are:
  - **leader-arm teleoperation** (a labelled fallback when Astra is unavailable or keeps failing);
  - the **scripted oracle** (sim only).

  Every episode carries its source.
- **Home to home.** Every recorded segment starts and ends at the home pose. The policy therefore always
  starts and finishes in the state it will be run from.
- **Reset segments are data too.** With `random_zone` or `cycle_zones`, a reset is itself a pick-and-place
  from a new start. Successful reset segments are saved as episodes tagged `segment=reset`, and each
  dataset version chooses whether to include them. This nearly doubles the real data.
- **Success comes from a verifier, never from the model's claim.**
  - **In sim:** the simulator's object pose.
  - **On a real arm:** the *perception verifier*. It segments the object in the calibrated overhead camera
    and checks that the object:
    - is inside the target zone;
    - is upright (for tall objects);
    - is no longer held: gripper present-load is near zero and the commanded and measured positions agree;
    - and that the arm reached home.

    The verifier's verdicts are `success`, `object_not_found`, `tipped`, `still_held`, `outside_zone`
    or `unreachable`. Humans can spot-check any verdict, and a confirmed disagreement is stored as a label.
- **Recovery.** After a failed segment the object is somewhere unknown. Cortex runs a RECOVER segment. If
  recovery fails twice, the loop pauses and asks the operator to reset by hand.
- **Measured starts.** The verifier records the object's measured start pose for every demo and every
  evaluation trial. Coverage and per-zone statistics use that measured pose, not the intended one.
- **Evaluation trials** reuse RESET segments between trials, so real-world evaluation needs no human
  reset either.

### 3.3 Stages and gates

| Stage | Loop | Done when | Gate to enter |
|---|---|---|---|
| Sim stage | Collect sim cycles with domain randomization (lighting, colour, camera jitter around the calibrated pose, object size ±10%, mass ±20%), train ACT, evaluate | Sim success ≥ `sim_target` (default 0.7, configurable) on held-out seeds, or the sim budget is spent | Spec feasible, reach check passed |
| Real rounds | K real cycles (default 10), fine-tune from the latest checkpoint, short real eval (default 5 trials), repeat | Real success ≥ `real_target`, or the round budget is spent | Hardware armed, live deadman, and a **human** approved the real stage (never a standing policy) |
| Improve | Section 3.4 | Target met, or budget spent | v1 training approval rules still apply |

Sim demos stay in every dataset, labelled by source and segment. The lineage graph shows exactly which
demos trained which checkpoint.

### 3.4 Improve: more demos, RL-distilled demos, or augmentation

After each evaluation, Cortex picks one action from the evidence and records why:

| Evidence | Action |
|---|---|
| Failures cluster in a start region or zone that has few demos | **Targeted real demos** there |
| Failures are spread out, near misses (grasp offset, early release), demos are plentiful | **RL refinement in sim, distilled.** PPO trains a *state-based* expert in sim with privileged object pose, under the domain randomization of the agent's scene. That expert then generates many extra sim demos, aimed at the failing conditions, which go into the next ACT fine-tune. Nothing trained by RL is deployed directly, so the image sim-to-real gap does not apply to the RL policy. |
| Lighting or appearance failures | Augmentation (v1) |
| Target met, or budget spent | Stop and deliver |

There is an optional second mode, a **proprioceptive residual** on top of ACT (inputs: joint state plus
the ACT action). It is proposed only when distillation stalls. RL never runs on the real arm, and RL
results count only once they are verified on it.

### 3.5 Deliver

`get_policy` returns:
- the checkpoint (LeRobot `pretrained_model/`);
- a **model card**: robot profile, task, scene, data provenance by source and segment, training runs,
  sim and real success with Wilson CIs, and known weak conditions;
- the `lerobot` command to run the policy on the agent's own arm.

## 3.6 The studio is an agentic chat

People use Cortex through a chat. It drives the same tools an MCP agent uses:
- The person types what the robot should learn.
- Intake questions appear as cards with one-click answers (their ready-made patches) or free text.
- The chat shows the sim scene from `capture_frame` and the feasibility report, with suggested
  alternatives for anything that fails.
- Collection, training and evaluation run from the chat, with live progress from the event stream.
- Training approval is an inline card: a human click. Agents still cannot approve.
- A **guided planner** walks the pipeline step by step when no model key is configured. With a key, the
  LLM agent can do the same through tool calls.

## 4. MCP surface added in v2

| Tool | Purpose |
|---|---|
| `start_intake(prompt?)` | Creates a draft project; returns `{project_id, spec, questions[]}` |
| `update_intake(project_id, patch)` | Merges a partial spec; returns `{spec, questions[], feasibility, complete}` |
| `capture_frame(project_id, camera)` | Image for the agent: a render in sim, a real camera frame otherwise |
| `build_simulation(project_id)` | Scene generation plus feasibility and reach report, with a preview image |
| `calibrate_cameras(project_id)` | Real arms: computes extrinsics from a calibration board |
| `start_cyclic_collection(project_id, backend, cycles, driver)` | Starts the cycle protocol (3.2); `backend` is `sim` or `real`; `driver` is `astra`, `teleop` or `oracle` |
| `start_real_rounds(project_id)` | Real-round loop (3.3); returns `approval_required` until a human approves |
| `start_rl_refinement(project_id, checkpoint_id, budget)` | Proposes RL distillation (3.4); approval rules apply |
| `get_policy(project_id)` | Best policy, model card, run command |

All v1 tools stay. Real-robot tools are always refused for remote callers.

## 5. Safety invariants

1. Astra never emits motor commands. Every target goes through validation and IK, and on hardware the
   per-tick safety controller checks it: joint limits, velocity, the workspace with the table as a hard
   floor, the watchdog, and the e-stop.
2. Real motion requires all of the following:
   - `CORTEX_ENABLE_HARDWARE=1`;
   - a local caller;
   - a calibration file;
   - a human-approved real stage;
   - a **live deadman**: an operator heartbeat from the studio or a pedal. If it lapses, the arm stops
     within one control tick and holds.
3. The operator re-confirms every N cycles (default 10). The real-world joint speed cap is lower than
   the sim's.
4. The watchdog runs on its own thread and trips on a stale command stream. The loop it checks never
   feeds it.
5. Training, real-robot stages and paid jobs need human approval. A standing Autopilot policy can
   approve free training. It can never approve real motion.

## 6. Prerequisites and risks

| Risk | Mitigation |
|---|---|
| GPT-6 Astra is blocked (the InsForge plan) and its accuracy is unmeasured | Measure Astra's success rate and cost per demo in sim **before** the real stage is built on it (roadmap P6b). Keep teleop as the labelled fallback. |
| Sim-to-real gap for image policies | Calibrated cameras, domain randomization, real fine-tune rounds, RL used only through distillation |
| The cylinder grasp is harder than a cube | The `sim_target` is configurable, and the payload and grasp checks run at intake |

## 7. Non-goals for v2

Arms other than SO-101 (the `RobotProfile` registry is the extension point), mesh import for objects,
deformable objects, and RL on hardware.
