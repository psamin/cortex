# Roadmap to product spec v2

Each package is one issue and one PR, kept under about 400 changed lines. The issue lists every file it
touches, so parallel agents can avoid each other's files. The PR that changes a package updates its
status here.

## MVP: everything in physics sim

| # | Package | Main files | Depends on | Status |
|---|---|---|---|---|
| P0 | Spec and roadmap | `docs/PRODUCT_SPEC.md`, `docs/ROADMAP.md` | – | in review |
| P1 | Windows: sim workers start, and the full EnvHub snapshot is fetched | `backend/cortex/sim/so101.py` | – | in review |
| P2 | Intake models, object catalogue, typed questions | `backend/cortex/intake/*`, tests | P0 | in review |
| P3 | Intake tools in the registry, REST and MCP (`start_intake`, `update_intake`, `capture_frame`) | `agent_orchestrator.py`, `api/app.py`, `mcp/src/tools.ts` | P2 | planned |
| P4 | Scene generator; feasibility checks (payload, jaw, stability); reach check | `backend/cortex/sim/scene.py`, `sim/feasibility.py`, tests | P2 | planned |
| P5 | Generic objects and zones in the manipulation tools; `go_home` | `backend/cortex/sim/manipulation.py`, `agentic/astra.py`, tests | P4 | planned |
| P6 | Cycle protocol in sim: demo, reset and recover segments, with segment tags | `workers/agent_worker.py`, `services/agent_collection_service.py`, tests | P5 | planned |
| P6a | Sim twin (held-out randomization) and image-only verifier, scored against simulator ground truth | `backend/cortex/sim/twin.py`, `sim/verifier.py`, tests | P5 | planned |
| P6b | Measure Astra in the twin: success rate, calls and cost per cycle | `scripts/measure_astra.py` | P6 | planned (needs gateway key) |
| P10 | Improve: decision rule | `services/improve.py` | P6 | planned |
| P11 | RL distillation: PPO state expert in sim, then demo generation | `backend/cortex/rl/*`, `infra/futurama/train_rl.sbatch` | P5 | planned |
| P12 | Autopilot v2 stages (sim pretrain, twin rounds, improve) | `services/autopilot.py` | P6a, P10, P11 | planned |
| P13 | Deliver: `get_policy` and model card | `services/policy_card.py`, tools | P12 | planned |
| P14 | Studio: intake and rounds views | `web/src/**` | P3, P12 | planned |

## Post-MVP: hardware track (separate modules; the MVP never imports them)

| # | Package | Main files | Depends on |
|---|---|---|---|
| P7 | Robot backend interface, hardware backend, watchdog thread, deadman | `backend/cortex/hardware/*` | P5 |
| P7b | Camera calibration (ArUco), feeding the sim camera pose | `backend/cortex/hardware/calibration.py` | P4, P7 |
| P8 | Real perception verifier (reuses the P6a verifier with real frames) | `backend/cortex/hardware/verifier.py` | P6a, P7b |
| P9a | Real cycle collection (operator-gated) | worker, service, API gates | P6, P7, P8 |
| P9b | Real evaluation trials with Astra resets | `services/simulation_runner.py` | P9a |
| P9c | Leader-arm teleop demo source | `backend/cortex/hardware/teleop.py` | P7 |
