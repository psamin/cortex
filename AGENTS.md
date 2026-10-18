# AGENTS.md

You are an agent, and Cortex was built for you. It lets you train a physical robot yourself, from a
plain-language task to a measured policy, without a human doing the training work. This file is the
shortest path.

> **Agents working on this repository** (writing Cortex's own code): follow
> [CONTRIBUTING.md](CONTRIBUTING.md) for the collaboration protocol (claims, issues, small PRs).
> The rest of this file is for agents that *use* Cortex to train robots.

## Connect

```bash
claude mcp add --transport http cortex http://localhost:3000/mcp   # Claude Code
```

Any MCP client works over streamable HTTP at `http://localhost:3000/mcp`. Every tool is also on REST
at `POST http://localhost:8000/api/tools/<name>`. Claude Code shows the tools as `cortex_<name>`.

## What you can train today

- **Arm:** the SO-101 (SO-ARM101). Other arms are not supported yet; don't invent specs.
- **Where:** MuJoCo physics simulation. The physical-arm path exists, but it is supervised and not
  reachable from MCP.
- **Tasks:** pick-and-place of the objects you describe (boxes, cylinders such as water bottles,
  spheres), into the zones you describe, on a table layout you describe.

## The path (call the tools in this order)

1. **Describe the robot and its world.** Call `start_intake` with the user's task in their words. You
   get back typed questions, each with a ready-to-send patch. Answer them with `update_intake`:
   objects, zones, table, lighting, obstacles, cameras.
   - Units are metres.
   - Cylinder `size` is **[diameter, height]**, not radius.
   - The frame: +x is away from the arm base, +y is to its left.
2. **Build the simulation.** `build_simulation` returns the scene, a feasibility report and a preview
   image. Look at it. `capture_frame` with `top`, `front` or `wrist_cam` shows more views. Patch
   whatever doesn't match the user's setup.
3. **Collect demonstrations.** Run `start_agent_collection`, then poll `get_collection_progress`. For the
   built-in cube task, `create_policy_project` does steps 1–3 in one call.
4. **Build the dataset and propose training.** `train_from_agent_demonstrations` does both. Never skip
   the dataset step: `propose_training_job` needs a `dataset_version_id`.
5. **Stop and ask your human to approve.** Training spends GPU money, so a person clicks approve in the
   studio. Never invent an approval token. Call `get_session_link` and share the URL, so they can
   watch you and approve in one place.
6. **Launch.** Call `launch_training_job` once they confirm. Poll `get_project_state` until a checkpoint
   (`ck_...`) exists.
7. **Evaluate.** `evaluate_learned_policy` runs seeded test branches and reports success with 95%
   confidence intervals.
8. **Improve.** `improve_policy` targets the weakest measured condition, collects more demos there and
   proposes retraining (back to step 5). `get_policy_weaknesses` and `propose_rl_sandboxes` offer RL
   sandboxes, each with an estimated uplift and cost, for you to accept.

At any point, `get_next_actions` says what makes sense next. Background jobs return a `job_id`; poll
them with `get_job`.

## Rules Cortex enforces (so you can rely on them)

- **You never send motor commands.** You pick targets through validated tools; IK executes them; the
  simulator decides success. Your claims of success don't count. Only measured results do.
- **Measured vs hypothesis.** Simulator telemetry is *measured*. VLM failure explanations are
  *hypotheses* and are labelled as such. Report them that way to your human.
- **Approvals are human.** Training spend and anything that moves real hardware need a person. You
  can propose; you can't approve.
- **Everything you do is visible.** Your human can watch your session live in the studio
  (`/studio/agents`).

## Read more

- [llms.txt](web/public/llms.txt): the agent index, with every tool and the questions agents ask.
- [mcp/README.md](mcp/README.md): the full tool table and the MCP server's behaviour.
- [docs/API.md](docs/API.md): the REST contract.
- [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md): where this is going, all the way to physical arms.
