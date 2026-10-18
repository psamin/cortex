# Roadmap to product spec v2

Each package is one issue and one PR, kept under about 400 changed lines. The issue lists every file it
touches, so parallel agents can avoid each other's files. The PR that changes a package updates its
status here.

## MVP: everything in physics sim

| # | Package | Main files | Depends on | Status |
|---|---|---|---|---|
| P0 | Spec and roadmap | `docs/PRODUCT_SPEC.md`, `docs/ROADMAP.md` | – | merged (#46) |
| P1 | Windows: sim workers start, and the full EnvHub snapshot is fetched | `backend/cortex/sim/so101.py` | – | merged (#45) |
| P2 | Intake models (arm, cameras, objects, zones, environment), object catalogue, typed questions | `backend/cortex/intake/*`, tests | P0 | merged (#47) |
| P3 | Intake tools in the registry, REST and MCP (`start_intake`, `update_intake`, `capture_frame`) | `agent_orchestrator.py`, `api/app.py`, `mcp/src/tools.ts` | P2 | merged (#51) |
| P4 | Scene generator; feasibility checks (payload, jaw, stability); reach check | `backend/cortex/sim/scene.py`, `sim/feasibility.py`, tests | P2 | merged (#50) |
| P5 | Generic objects and zones in the manipulation tools; `go_home` | `backend/cortex/sim/manipulation.py`, `agentic/astra.py`, tests | P4 | planned |
| P6 | Cycle protocol in sim: demo, reset and recover segments, with segment tags | `workers/agent_worker.py`, `services/agent_collection_service.py`, tests | P5 | planned |
| P5b | `RobotBackend` interface (twin implements it; hardware later swaps in) | `backend/cortex/robot/backend.py` | P5 | planned |
| P6a | Sim twin (structural gap: latency, gains, noise; randomization sets A/B/C) and image-only verifier scored against simulator ground truth | `backend/cortex/sim/twin.py`, `sim/verifier.py`, tests | P5b | planned |
| P6c | Camera-mode scripted oracle with pointing noise (Astra stand-in, labelled) | `backend/cortex/sim/camera_oracle.py` | P5 | planned |
| P6b | Measure Astra in the twin: success rate, calls and cost per cycle | `scripts/measure_astra.py` | P6 | planned (needs gateway key) |
| P10 | Improve: decision rule | `services/improve.py` | P6 | planned |
| P11 | RL distillation: PPO state expert in sim, then demo generation | `backend/cortex/rl/*`, `infra/futurama/train_rl.sbatch` | P5 | planned |
| P12 | Autopilot v2 stages (sim pretrain, twin rounds, improve) | `services/autopilot.py` | P6a, P10, P11 | planned |
| P13 | Deliver: `get_policy` and model card | `services/policy_card.py`, tools | P12 | planned |
| P14 | **Agentic chat studio**: a clean chat-first UI where a person teaches and trains a policy end to end (intake question cards, scene previews, feasibility, collection, training approval, evaluation, live progress). Works with the LLM agent or, without a model key, a guided planner. | `web/src/**`, `backend/cortex/services/agent.py`, `backend/cortex/services/guide.py` | P3 | done upstream by @psamin: chat-first studio #22 / #38 / #48 / #59; Kinetic restyle #52 in progress |

## Already built upstream (by @psamin): build on these, don't duplicate them

- **Robot sessions:** the user's own agent drives the arm through validated tools (#14). MCP robot tools return camera images (#21). This covers the session half of P5.
- **Batch collection** with Claude Code as the planner (#24), and **demo import** from the Hub or an upload (#33, #48).
- **Real SO-101 track** (#25): the driver (#36), setup and arming (#37), calibration (#39), and the control loop with deadman, watchdog thread and table floor (#58). This covers P7 and P9c.

## Post-MVP: hardware track (separate modules; the MVP never imports them)

| # | Package | Main files | Depends on |
|---|---|---|---|
| P7 | Hardware `RobotBackend`, watchdog thread, deadman | `backend/cortex/hardware/*` | P5 |
| P7b | Camera calibration (ArUco), feeding the sim camera pose | `backend/cortex/hardware/calibration.py` | P4, P7 |
| P8 | Real perception verifier (reuses the P6a verifier with real frames) | `backend/cortex/hardware/verifier.py` | P6a, P7b |
| P9a | Real cycle collection (operator-gated) | worker, service, API gates | P6, P7, P8 |
| P9b | Real evaluation trials with Astra resets | `services/simulation_runner.py` | P9a |
| P9c | Leader-arm teleop demo source | `backend/cortex/hardware/teleop.py` | P7 |

## Claims (in flight)

Check this table before you edit. Add your row when you start, and remove it when your PR merges. The issue
comments (the `Claim:` convention on #20 and the owning issue) remain the source of truth.

| Package | Branch | Driver | Files touched |
|---|---|---|---|
| #52 redesign (Kinetic) | `52-landing`, then `52-*` | Claude Code, driven by @AneeshD04 | `web/src/landing/**`, `web/src/robotics/**`, small route edits in `web/src/App.tsx` and `layout/AppShell.tsx`, visual-only edits in `web/src/**` |
| P5/P6: sessions and collection on intake scenes | not started | Claude Code, driven by @AneeshD04 | will touch `backend/cortex/sim/manipulation.py` and `backend/cortex/services/robot_session_service.py`, coordinated on #16 |
