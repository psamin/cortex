# Two-minute judge walkthrough

**The pitch in one line:** an agent connects to one MCP server and trains a physical robot itself. Nobody teleoperates, labels or babysits the runs; a person only approves the spend. Everything below shows the agent doing the work.

## Before the session (not shown live)

1. Put the InsForge key in `.env`, run `scripts/dev.sh`, and open http://127.0.0.1:5173. The header pills
   should show InsForge for db, storage and VLM, and `GET /api/planner` should report GPT-6 Astra via the
   InsForge Model Gateway. If anything shows FALLBACK, say so on stage.
2. GT VPN on. Prepare a **prior project** an hour ahead so a real trained policy exists. In it, Autopilot
   collects Astra demos, trains ACT on Futurama, evaluates it on three conditions, and starts the targeted
   round. Present its checkpoint and evaluation as **prior results**, with their timestamps. Never present
   them as live.
3. Run `claude mcp add --transport http cortex http://localhost:3000/mcp`.

## Live (about 2 min)

| Time | Do | Say |
|---|---|---|
| 0:00 | Type "Train my SO-101 to pick up a cube." Leave Simulation and Agent Demonstrations on. Start. | "One sentence in. No teleoperation: Cortex generates its own demonstrations." |
| 0:10 | Collect → Agent Demonstrations. Left: the live MuJoCo viewport. Centre: the action timeline (Observing scene → Selecting grasp → Approaching → Closing gripper → Lifting → Moving toward bowl → Releasing → Verifying success). | "GPT-6 Astra looks at the rendered scene and points at the cube. A local IK controller validates every command and executes it. The arm you see is the actual simulator run." |
| 0:35 | Point at a rejected command in the timeline, if one occurred (red `unreachable` / `outside_workspace`), and at the replan after it. | "Astra never sends motor commands. Invalid plans come back as structured errors, and it replans." |
| 0:45 | Right panel: successes vs failed attempts, model calls and $, coverage. Click an episode to see its provenance badge (GPT-6 Astra · camera) and full action trace. | "Only simulator-verified successes become LeRobot episodes. Failures are kept as diagnostics. Every action and its cost is stored in InsForge." |
| 1:00 | Lineage graph: the new episodes flowing into a dataset version. Click a node to see its InsForge table/row and storage object. | "The dataset lives in InsForge storage, with lineage back to every attempt." |
| 1:10 | Switch to the prior project → Evaluations: ACT checkpoint on baseline / shifted object / dim light, with CIs. | "This is a real ACT policy trained earlier on agent demos and evaluated on held-out seeds. These are prior results." |
| 1:25 | Autopilot timeline: NEXT_EXPERIMENT_SELECTED: "ACT failed 4 of 6 shifted-object trials → generate 5 agent demos in left/normal". The viewport shows those new demos being generated. | "Cortex diagnosed the weakest condition and is collecting targeted demos for it now. Retraining needs approval, so nothing here claims the policy already improved." |
| 1:45 | Terminal: `claude -p "Use cortex to show collection progress and demonstration results for <pid>"`. Then the Activity tab shows the MCP calls. | "Agents get the same backend through Manufact's MCP. They can't bypass the budget, GPU or hardware approvals." |

## If something breaks

- **No Astra key:** the Collect screen says so and offers the scripted oracle collector. Run
  `scripts/demo_agent.py --collector oracle` and say plainly that it's the oracle.
- **Cluster unreachable:** show the prior project's checkpoint. Don't start new training on stage.
- **Studio down:** `uv --project backend run python scripts/demo_agent.py --demos 2` prints the agent's
  executed actions live from the same backend.
