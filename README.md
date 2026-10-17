# Cortex

An agent-native robot-learning studio on [LeRobot](https://github.com/huggingface/lerobot). Tell it
"I want my SO-101 to pick up a cube and place it in a bowl" and it runs the imitation-learning loop:
collect demonstrations, check their quality, diagnose failures, recommend which demos to collect next,
version and augment datasets, train an ACT policy, evaluate it in parallel MuJoCo conditions, compare
results, and recommend the next step. **Autopilot** runs that loop unattended within the approvals
you grant up front.

Demonstrations are generated autonomously: an **agent planner**, by default our own self-hosted
**Qwen3-VL** (or GPT-6 Astra via OpenRouter), looks at the rendered MuJoCo scene and picks actions
through validated tools (point at the cube, move, grip, verify, place), and a local IK controller
executes them. Only simulator-verified successes become LeRobot episodes. The existing
scripted IK oracle stays available as a labelled baseline for A/B comparison.

Hugging Face's LeLab is a GUI for LeRobot. Cortex is an experiment orchestrator and data-collection
agent: it decides what to do next and shows the evidence for each decision. People use it through the
React studio; agents use the same project through an MCP server.

## Quick start

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), Node 22, pnpm.

```bash
cp .env.example .env          # optional: every provider falls back to a local equivalent
scripts/dev.sh                # backend :8000, studio :5173, MCP :3000
open http://127.0.0.1:5173
```

Type "Teach a robot to pick and place a cube.", keep Simulation and Autopilot on, and press Start.

Connect Claude Code to the same backend:

```bash
claude mcp add --transport http cortex http://localhost:3000/mcp
```

## Commands

| What | Command |
|---|---|
| Backend only | `cd backend && uv run uvicorn cortex.api.app:app --host 127.0.0.1 --port 8000` |
| Studio only | `cd web && pnpm install && pnpm dev` |
| MCP server only | `cd mcp && npm ci && npm run dev` |
| Unit tests | `cd backend && uv run pytest` |
| End-to-end test (real sim, needs backend running) | `uv --project backend run python scripts/smoke_e2e.py --small` |
| Generate sample demos without the UI | `uv --project backend run python scripts/generate_demos.py --out sample_data --per-condition 2` |
| Expose backend for a hosted MCP server | `scripts/dev.sh --tunnel` |
| Agent-demo walkthrough (live action stream) | `uv --project backend run python scripts/demo_agent.py --demos 3` (`--collector oracle` needs no key) |
| Serve our own planner model on Futurama | `scripts/vlm_futurama.sh up` (then restart the backend); `status`, `down` |
| A/B: scripted oracle vs agent planner | `uv --project backend run python scripts/ab_oracle_vs_astra.py collect --n 10` then `train`, `evaluate` |

## Repository

| Path | Contents |
|---|---|
| [backend/cortex/](backend/cortex/) | FastAPI app, domain services, sim, workers |
| [backend/migrations/](backend/migrations/) | SQL migrations (applied to InsForge Postgres or the SQLite fallback) |
| [web/](web/) | React + TypeScript + Vite + Tailwind studio |
| [mcp/](mcp/) | mcp-use v2 MCP server (deployable on Manufact) |
| [infra/futurama/](infra/futurama/) | Slurm job for ACT training on the GT Futurama cluster |
| [scripts/](scripts/) | dev launcher, e2e test, deterministic demo generator |
| [docs/](docs/) | [API contract](docs/API.md), [provider integrations and limitations](docs/INTEGRATIONS.md) |

Read next: [ARCHITECTURE.md](ARCHITECTURE.md), then [DEMO.md](DEMO.md) for the two-minute judge walkthrough.

## What is real

- **Agent demos.** GPT-6 Astra demonstrations are real MuJoCo executions of validated IK trajectories,
  recorded step by step. Astra never emits motor commands, and success is the simulator's verdict, not
  the model's. Failed attempts are kept as diagnostics and never enter a dataset. If no Astra route is
  configured, Cortex says so and does not substitute another model.
- **Physics.** Demonstrations and evaluations are MuJoCo rollouts in the official EnvHub environment
  `lerobot/so101-tabletop-mujoco`, pinned to commit `12e504e8`. They are never replayed or mocked.
- **Scripted oracle.** Oracle demos come from a scripted IK expert that reads simulator state. Every
  episode carries provenance (`gpt-6-astra` or `scripted_ik_oracle`), and the oracle is never presented
  as a learned policy.
- **Datasets.** Every dataset is a real LeRobotDataset v3 (lerobot 0.6.1) with a lineage file.
- **Training.** Training is real `lerobot-train` ACT on a GPU cluster. Cortex never reports a
  training result without a checkpoint on disk.
- **Labelling.** VLM diagnoses are stored as unverified hypotheses. Measured outcomes come only
  from simulator telemetry.

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for what each provider integration does today and
what is still unverified.
