# Cortex

**The robot-training MCP agents want.** Cortex lets any AI agent train its own physical robot, with no
human in the training loop.

Agents can write software, run experiments and ship code. They still can't teach a robot. Robot
learning today needs people at every step:
- someone teleoperates the demonstrations;
- someone labels and cleans the data;
- someone launches training and babysits the runs;
- someone watches rollouts to decide what failed and what to collect next.

That human loop is why robot policies are slow and expensive to make. It's also why most factories,
warehouses, labs, kitchens and farms still can't retrain a robot when the task or the layout changes.

Cortex turns each of those steps into an MCP tool with measured results, so the agent runs the whole
loop itself:

1. **Describe.** The agent tells Cortex the arm, cameras, objects and workspace layout, in plain
   language over MCP. Any custom environment, not a fixed demo scene.
2. **Simulate.** Cortex builds that layout as a MuJoCo scene and measures whether the arm can do the
   job: jaw opening, holding torque, reach.
3. **Demonstrate.** The agent drives the arm itself through validated tools: home → pick → place →
   home, then it resets and goes again. Only simulator-verified successes become LeRobot episodes.
4. **Augment.** Clones of the demos get adversarial changes such as lighting and camera, so the
   policy is robust to more than one scene.
5. **Train.** An ACT policy trains on the agent's own demonstrations.
6. **Evaluate.** Seeded test runs measure success with confidence intervals. The agent reads the
   numbers, not a person.
7. **Improve.** The agent finds the weakest condition, collects targeted demos or runs RL sandboxes
   for it, and retrains until the numbers hold.
8. **Deploy.** The policy goes to the physical arm to keep training on real demonstrations and then
   run the task.

**Humans out of the loop, not out of control.** People don't do the training work. They only
approve the two things an agent should never decide alone:
- spending money, such as a GPU training job;
- moving real hardware, which happens only with an operator present, a live deadman and an e-stop.

An agent can never approve its own training job or drive a real arm through the MCP.

Built on [LeRobot](https://github.com/huggingface/lerobot) 0.6.1, MuJoCo, ACT and the SO-101 arm.
The MVP runs the whole loop in physics simulation. The physical-arm path is supervised and early.
See [What is real](#what-is-real).

## For agents

If you are an agent, start with [AGENTS.md](AGENTS.md). It has the exact tool sequence from "the
user named a task" to "a measured policy". Connect once:

```bash
claude mcp add --transport http cortex http://localhost:3000/mcp
```

Then the user only has to say what the robot should learn. The MCP server's own `instructions`
carry the same step-by-step path, so any MCP client gets it on connect. Agent-readable indexes:
[llms.txt](web/public/llms.txt) and [.well-known/mcp.json](web/public/.well-known/mcp.json).

## For the humans watching

The studio (React, http://127.0.0.1:5173) is where people watch what their agents are doing:
- each agent session's live timeline of tool calls, results and camera frames (`/studio/agents`);
- the demos and evaluation videos;
- the approval buttons.

It also runs its own studio agent, so you can drive the same loop by chat.

## Quick start

Requirements: macOS, Linux or Windows, [uv](https://docs.astral.sh/uv/), Node 22, pnpm. On Windows, run
`scripts/dev.sh` from Git Bash, or start the backend, studio and MCP server separately (see Commands).

```bash
cp .env.example .env          # optional: every provider falls back to a local equivalent
scripts/dev.sh                # backend :8000, studio :5173, MCP :3000
claude mcp add --transport http cortex http://localhost:3000/mcp
```

Then tell your agent: *"Train my SO-101 to pick up the water bottle and put it on the mat."* Watch it
in the studio at http://127.0.0.1:5173.

How the demonstrations are made:
- The default demonstrator is an **agent planner**: our self-hosted **Qwen3-VL**, or GPT-6 Astra via OpenRouter. It looks at the rendered scene and picks actions through validated tools, and a local IK controller executes them.
- Your own agent can also drive the arm directly with the `robot_*` tools.
- The scripted IK oracle stays available as a labelled baseline.

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

Read next: [AGENTS.md](AGENTS.md) if you are an agent, [ARCHITECTURE.md](ARCHITECTURE.md), then [DEMO.md](DEMO.md) for the two-minute judge walkthrough.

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
