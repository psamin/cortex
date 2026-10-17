# Roadmap to product spec v2

Each package is one issue and one PR, kept under about 400 changed lines. The issue lists every file
the package touches, so parallel agents can avoid each other's files. Statuses are updated in the PR
that changes them.

| # | Package | Files (main) | Depends on | Status |
|---|---|---|---|---|
| P0 | Spec and roadmap | `docs/PRODUCT_SPEC.md`, `docs/ROADMAP.md` | none | in review |
| P1 | Windows: sim workers start on Windows | `backend/cortex/sim/so101.py` | none | planned |
| P2 | Intake models and service | `backend/cortex/models.py`, `backend/cortex/services/intake_service.py`, tests | P0 | planned |
| P3 | Intake tools: registry, REST, MCP | `agent_orchestrator.py`, `api/app.py`, `mcp/src/tools.ts` | P2 | planned |
| P4 | Scene generator and reach check | `backend/cortex/sim/scene.py`, `sim/so101.py`, tests | P2 | planned |
| P5 | Generic objects and zones in manipulation tools; `go_home` | `backend/cortex/sim/manipulation.py`, `agentic/astra.py`, tests | P4 | planned |
| P6 | Cyclic collection protocol (demo and reset segments) | `backend/cortex/workers/agent_worker.py`, `services/agent_collection_service.py`, tests | P5 | planned |
| P7 | Robot backend interface; hardware backend; watchdog fix | `backend/cortex/hardware/*`, tests | P5 | planned |
| P8 | Real-world perception verifier | `backend/cortex/hardware/verifier.py`, tests | P7 | planned |
| P9 | Real stage: collection and evaluation on hardware (operator-gated) | services, worker, API gates | P6, P7, P8 | planned |
| P10 | Improve: decision rule (demos, RL or augmentation) | `services/improve_policy.py`, `autopilot.py` | P6 | planned |
| P11 | RL refinement in sim (residual PPO) | `backend/cortex/rl/*`, `infra/futurama/train_rl.sbatch` | P5 | planned |
| P12 | Autopilot v2 stages | `services/autopilot.py` | P9, P10, P11 | planned |
| P13 | Deliver: `get_policy`, model card | `services/policy_card.py`, tools | P12 | planned |
| P14 | Studio: intake and stage views | `web/src/**` | P3, P12 | planned |
