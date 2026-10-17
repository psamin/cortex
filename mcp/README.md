# Cortex MCP server

An MCP server, built with [mcp-use](https://docs.mcp-use.com) v2 (TypeScript), that lets agents such as Claude Code drive Cortex. It is a thin layer over the backend tool registry. Each `cortex.<name>` tool forwards its arguments to `POST {CORTEX_API_URL}/tools/<name>` and returns:

- `structuredContent`: the backend's JSON (`summary`, `ref_ids`, plus the data).
- `content`: a short text line, the `summary` and the `ref_ids`.

Media is never streamed. Episodes and evaluations come back as ids and storage keys, for example `video_key: projects/<pid>/evaluations/<ev>/<br>/seed1000.mp4`. The local backend serves those keys at `http://127.0.0.1:8000/files/<key>`.

| Tool | Backend tool | Notes |
|---|---|---|
| `cortex.create_project` | `create_project` | returns `project_id` and the demo plan |
| `cortex.get_project_state` | `get_project_state` | read-only; also how you poll evaluations |
| `cortex.list_episodes` | `list_episodes` | read-only |
| `cortex.record_demonstrations` | `record_demonstrations` | sim only; background job, `wait_s` |
| `cortex.analyze_dataset` | `analyze_dataset` | background job, `wait_s` |
| `cortex.get_failure_clusters` | `get_failure_clusters` | read-only; `source="vlm"` clusters are unverified hypotheses |
| `cortex.recommend_demonstrations` | `recommend_demonstrations` | replaces older proposed recommendations |
| `cortex.create_dataset_version` | `create_dataset_version` | background job, `wait_s`; the `ds_...` id is the job's `result_ref` |
| `cortex.propose_training_job` | `propose_training_job` | does not start training |
| `cortex.launch_training_job` | `launch_training_job` | needs human approval in the studio (see below) |
| `cortex.launch_simulation_evaluation` | `launch_simulation_evaluation` | `wait_s`; returns an `ev_...` id, not a job id |
| `cortex.compare_experiments` | `compare_experiments` | read-only |
| `cortex.get_next_actions` | `get_next_actions` | read-only |
| `cortex.get_job` | `get_job` | read-only; poll background jobs |

There is also one prompt, `cortex.teach_task(task, project_id?)`. It walks an agent through the whole loop: create the project, record demonstrations, analyze, recommend, build a dataset version, propose training, ask the human for approval, evaluate in simulation, compare, and get the next actions.

**Approval.** `cortex.launch_training_job` only runs a job that a human has approved in the studio. If the job is not approved, the backend answers `403 {approval_required: true}`. The server turns that into a normal, non-error result that says `HUMAN APPROVAL REQUIRED`. The result includes `next_step` and `studio_url` (from `CORTEX_STUDIO_URL`), so the agent asks the human to approve instead of retrying. Other errors (404, 422, unreachable backend, remote access refused) come back as `isError: true`, with the backend's `detail`.

**Long jobs.** The backend's `wait_s` blocks for up to 600 s (900 s for evaluations), but many MCP clients give up on a tool call after 60 s. The official Inspector CLI timed out at exactly 60 s. The tool descriptions therefore tell agents to use `wait_s <= 45` and then poll `cortex.get_job`. Times measured on a laptop:

| Job | Time |
|---|---|
| Recording | about 5 s per demo |
| Analysis without the VLM | about 1 s |
| Dataset of 11 episodes with `lighting_robustness` | about 20 s |
| Scripted-expert evaluation, 2 episodes per branch | about 30 s |
| Scripted-expert evaluation, 10 episodes per branch | about 2 min |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CORTEX_API_URL` | `http://127.0.0.1:8000/api` | Backend API base. It must be a **public** URL when the server is hosted. |
| `CORTEX_MCP_KEY` | unset | Sent as `Authorization: Bearer ...`. Required whenever the backend is reached through a tunnel. |
| `CORTEX_STUDIO_URL` | `http://localhost:5173` | Studio link included in approval messages. |
| `PORT` | `3000` | Listen port for `npm start`. |

Every request also sends `X-Cortex-Actor: mcp`, so the backend audit log attributes the calls to MCP.

## Run locally

Requires Node 22. mcp-use declares `>=22.22.2`, and 22.19 also works but prints an `EBADENGINE` warning. The backend must be running on `:8000`.

```bash
cd mcp
npm ci            # or npm install
npm run dev       # http://localhost:3000/mcp, Inspector UI at http://localhost:3000/mcp/inspector (hot reload)
# production-style:
npm run build     # -> .mcp-use/build/index.js
npm start         # serves the build on :3000 (PORT / --port to change)
npm run typecheck
```

`scripts/dev.sh` at the repo root already starts this server with the backend and studio.

Smoke test:

```bash
H='-H Content-Type:application/json -H Accept:application/json,text/event-stream'
curl -s $H localhost:3000/mcp -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
npx -y @modelcontextprotocol/inspector --cli http://localhost:3000/mcp --transport http \
  --method tools/call --tool-name cortex.get_next_actions --tool-arg project_id=<prj_...>
```

## Add to Claude Code

```bash
claude mcp add --transport http cortex http://localhost:3000/mcp
```

Claude Code lists the tools as `mcp__cortex__cortex_<name>`, because it turns the dot into `_`. To allow them all in headless runs:

```bash
claude -p "Use the cortex MCP tools to list next actions for project <prj_...>" --allowedTools "mcp__cortex__*"
```

To share the server with teammates, add `-s project`. That writes a `.mcp.json` at the repo root, which is equivalent to:

```json
{ "mcpServers": { "cortex": { "type": "http", "url": "http://localhost:3000/mcp" } } }
```

For a hosted server, use its URL instead: `claude mcp add --transport http cortex https://<slug>.run.mcp-use.com/mcp`.

## Deploy to Manufact (mcp-use cloud)

A hosted MCP server **cannot reach `localhost:8000`**, because inside the Manufact container `localhost` is the container itself. The backend therefore has to be exposed first.

1. **Pick an MCP key and give it to the backend.** Add `CORTEX_MCP_KEY=<long random string>` to `backend/.env`, or export it, then restart the backend. The backend's guard treats any request carrying `cf-connecting-ip` or `x-forwarded-for` as remote. A remote request can only reach `/api/health`, and `/api/tools*` with `Authorization: Bearer <CORTEX_MCP_KEY>`; everything else gets a 403. Without a key, remote tool calls are always refused.
2. **Expose the backend with a tunnel.** A Cloudflare quick tunnel needs no account:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000     # prints https://<random>.trycloudflare.com
   curl https://<random>.trycloudflare.com/api/health  # should answer
   ```
   The quick-tunnel URL changes every time `cloudflared` restarts. When it changes, update `CORTEX_API_URL` on Manufact (step 4).
3. **Log in and deploy** from this directory:
   ```bash
   cd mcp
   npx mcp-use login
   npx mcp-use deploy --no-github --name cortex \
     --env CORTEX_API_URL=https://<random>.trycloudflare.com/api \
     --env CORTEX_MCP_KEY=<same key as the backend> \
     --env CORTEX_STUDIO_URL=http://localhost:5173
   ```
   - `--no-github` uploads this folder as a tarball. `node_modules`, `.env*` and git-ignored paths are left out.
   - Manufact detects Node from `package.json` and `package-lock.json`, runs `npm run build`, then `npm start` on port 3000.
   - To deploy from GitHub instead, run `npx mcp-use deploy --root-dir mcp --name cortex --env ...` from a pushed branch. Do not pass `-y` unless you want the CLI to `git add`, commit and push for you.
   - The CLI prints the server URL, `https://<slug>.run.mcp-use.com/mcp`. Follow the build with `npx mcp-use deployments logs <id> --build --follow`.
4. **Change env vars later** with `npx mcp-use servers env add CORTEX_API_URL=<new url> --server <id>` or in the dashboard, then `npx mcp-use deployments restart <id>`. Env changes only apply after a restart or a redeploy.
5. **Connect:** `claude mcp add --transport http cortex https://<slug>.run.mcp-use.com/mcp`.

Notes:

- The Manufact free plan scales to zero, so expect a cold start on the first call.
- **The deployed `/mcp` endpoint has no auth of its own.** Anyone with the URL can call the `cortex.*` tools through it: create projects, record simulations, run evaluations. Training still cannot launch without a human approval in the studio. Treat the URL as a secret, and stop the tunnel when you are done.
- An alternative with no deploy is `npm run dev -- --tunnel` (`mcp-use start --tunnel` also works). This exposes the local MCP server at `https://<name>.tunnel.mcp-use.run/mcp`, and because the server runs locally it still reaches `localhost:8000` with no backend tunnel. This was not tested here.

## Layout

```
index.ts         MCPServer setup (name "cortex"), registers tools and the prompt
src/backend.ts   POST /tools/{name}: headers, error mapping, approval handling
src/tools.ts     zod input schemas mirroring GET /api/tools, descriptions, annotations
src/prompts.ts   cortex.teach_task
```

The backend registry, `GET /api/tools`, is the source of truth. If a backend tool's arguments change, update `src/tools.ts` to match. The schemas there mirror the registry's, except that the enums the domain model enforces are spelled out: `cube_region`, `lighting`, `policy_type`, `executor` and `augmentation_preset`.
