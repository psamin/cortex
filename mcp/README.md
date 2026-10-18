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
| `cortex.start_intake` | `start_intake` | spec v2: creates a draft project; returns typed questions, each with a ready-to-send `patch` |
| `cortex.update_intake` | `update_intake` | merges a partial spec; lists merge by `name`; unknown keys are rejected |
| `cortex.get_intake_status` | `get_intake_status` | read-only |
| `cortex.build_simulation` | `build_simulation` | builds the MuJoCo scene, checks feasibility; returns a preview image |
| `cortex.capture_frame` | `capture_frame` | read-only; returns an image of `top`, `front` or `wrist_cam` |
| `cortex.get_session_link` | `get_session_link` | read-only; the studio URL where the human watches this session live |

## Views in the agent's chat (MCP Apps)

Three tools render interactive views inside MCP Apps hosts (Claude, ChatGPT, VS Code, the mcp-use Inspector), built with mcp-use views under `views/<name>/view.tsx`. Hosts without MCP Apps get the same text summary as before.

| Tool | View | Shows |
|---|---|---|
| `cortex.show_collection` | `collection` | The latest MuJoCo frame, successes vs attempts, model calls, spend and recent attempts. Refreshes every 2 s while the collection runs. |
| `cortex.show_evaluation` | `evaluation` | Success per condition with its Wilson 95% interval, measured failure modes, and the weakest condition (only when conditions differ). Refreshes while branches run. |
| `cortex.show_training_review` | `approval` | A proposed training job (GPU hours, cost, dataset, rationale) with an **Approve** button. |

The `cortex.view_*` tools have `visibility: "app"`: the views call them to refresh or approve, and hosts hide them from the model.

**Only a human can approve through the view.** `show_training_review` gets a single-use approval nonce from the backend (10-minute expiry, bound to that training job) and puts it only in the result's `_meta`, which MCP Apps hosts give to the view and keep out of the model's context. It is never in `content` or `structuredContent`. The Approve button sends it back through `cortex.view_approve_training`, and the backend records the approval as "human click in the MCP App view". A wrong, reused or expired nonce is refused. The model never sees the nonce, so `cortex.launch_training_job` still answers HUMAN APPROVAL REQUIRED until someone clicks. This relies on the host keeping `_meta` from the model, as the MCP Apps spec requires; the studio and the operator token remain the other ways to approve.

Try them in the Inspector (`npm run dev`, then http://localhost:3000/mcp/inspector): run `cortex.show_collection` with a project id and switch the response to **Component (MCP Apps)**. In dev mode each view compiles on first use, which takes a few seconds.

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
| `CORTEX_API_URL` | `http://127.0.0.1:8000/api` | Backend API base for direct mode. It is used when no runner is connected. A hosted server can leave it unset and use the runner. |
| `CORTEX_MCP_KEY` | unset | Shared secret. It authenticates the runner link (`/relay/*`), which is disabled while the key is unset, and it is sent as `Authorization: Bearer ...` in direct mode. |
| `CORTEX_STUDIO_URL` | `http://localhost:5173` | Studio link included in approval messages. |
| `CORTEX_AGENT_KEY` | unset | Agent key (`ck_...`) from the studio's **Agents → Settings · Agent keys**. Used when the client sends none of its own. |
| `PORT` | `3000` | Listen port for `npm start`. |

Every request also sends `X-Cortex-Actor: mcp`, so the backend audit log attributes the calls to MCP.

## Agent observability: watch your agent in the studio

Every backend call also carries who is calling, so the studio's **Agents** view (`/studio/agents`) shows each connection as a session with a live timeline of its tool calls (arguments, result summary, returned images, duration, status, approvals):

| Header (MCP server → backend) | Value |
|---|---|
| `X-Cortex-Agent-Key` | the agent key: the client's own `X-Cortex-Agent-Key` header (or `Authorization: Bearer ck_...`) to this server, else `CORTEX_AGENT_KEY`. It ties the calls to the key's owner. |
| `X-Cortex-Agent-Session` | the connection's MCP session id |
| `X-Cortex-Client` | the client's `clientInfo` from `initialize`, as `name/version` (e.g. `claude-code/2.1.278`) |

`Authorization: Bearer CORTEX_MCP_KEY` (server → backend) is still only the tunnel guard; the agent key never travels in it.

mcp-use v2 serves every request statelessly and assigns no session id, so `index.ts` installs a small middleware (`trackClients` in `src/backend.ts`) that gives each `initialize` an `Mcp-Session-Id: cx-<uuid>` response header and remembers that connection's `clientInfo`. Streamable-HTTP clients send the id back on every call. Clients on the 2026 per-request protocol carry no session; their calls are grouped per client and key.

Without a key, calls belong to the backend's implicit local owner (local mode is single-user). An unknown or revoked key is refused (`isError`, "Cortex refused this agent key"). Keys only attribute calls: they can never approve training or drive the real arm.

Connect with a key (the studio shows these snippets with your key filled in):

```bash
claude mcp add --transport http cortex http://localhost:3000/mcp --header "X-Cortex-Agent-Key: ck_..."
```

```json
{ "mcpServers": { "cortex": { "type": "http", "url": "http://localhost:3000/mcp", "headers": { "X-Cortex-Agent-Key": "ck_..." } } } }
```

Then ask the agent to call `cortex.get_session_link` and open the URL it returns.

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

A hosted MCP server **cannot reach `localhost:8000`**, because inside the Manufact container `localhost` is the container itself. There are three ways to connect them. The runner is the recommended one.

| Mode | MCP server URL | Backend exposed? | URL stable across restarts? |
|---|---|---|---|
| **Runner** (recommended) | `https://<slug>.run.mcp-use.com/mcp` | no, the runner dials out | yes |
| **mcp-use tunnel** (no deploy) | `https://<name>.tunnel.mcp-use.run/mcp` | no, the MCP server runs locally | yes (verified across a restart) |
| Cloudflare quick tunnel (legacy) | `https://<slug>.run.mcp-use.com/mcp` | yes, `/api/tools` with the key | no, update `CORTEX_API_URL` every time |

### Runner: hosted server, local backend, no tunnel

The local runner (`backend/cortex/mcp_runner.py`) opens an outbound HTTPS link to the hosted server and long-polls it. The server hands each `cortex.*` call to the runner (`src/relay.ts`). The runner POSTs it to the local backend's `/api/tools/<name>` and sends the response back, so results, error mapping and approval handling are identical to direct mode. Nothing listens on a public port.

1. **Pick a shared key.** Run `openssl rand -hex 24`, then put `CORTEX_MCP_KEY=<key>` in the repo-root `.env`. The runner reads it from there.
2. **Deploy once** from this directory:
   ```bash
   cd mcp
   npx mcp-use login
   npx mcp-use deploy --no-github --name cortex \
     --env CORTEX_MCP_KEY=<same key> \
     --env CORTEX_STUDIO_URL=http://localhost:5173
   ```
   - `--no-github` uploads this folder as a tarball. `node_modules`, `.env*` and git-ignored paths are left out.
   - Manufact detects Node from `package.json` and `package-lock.json`, runs `npm run build`, then `npm start` on port 3000.
   - To deploy from GitHub instead, run `npx mcp-use deploy --root-dir mcp --name cortex --env ...` from a pushed branch. Do not pass `-y` unless you want the CLI to `git add`, commit and push for you.
   - The CLI prints the server URL, `https://<slug>.run.mcp-use.com/mcp`. Follow the build with `npx mcp-use deployments logs <id> --build --follow`.
   - Leave `CORTEX_API_URL` unset. Whenever a runner is connected, every call goes through it.
3. **Start the stack with the runner:**
   ```bash
   scripts/dev.sh --runner https://<slug>.run.mcp-use.com
   # or just the runner, next to a backend that is already running:
   cd backend && uv run python -m cortex.mcp_runner --server https://<slug>.run.mcp-use.com
   ```
   It logs `connected to ...`, then one line per tool call. `--api` points it at a backend other than `http://127.0.0.1:8000/api`.
4. **Connect:** `claude mcp add --transport http cortex https://<slug>.run.mcp-use.com/mcp`.

How transport selection works (`src/backend.ts`): when a runner has an open poll (or polled within the last 5 s), calls go through it. Otherwise the server POSTs to `CORTEX_API_URL` directly, which is how local dev and the Cloudflare mode keep working unchanged. If neither works, the tool returns `isError: true` with `Cannot reach the Cortex backend: no local runner is connected and <url> is unreachable ... Start the local runner ...`. A call that no runner picks up within 30 s fails with a message saying the runner seems to have disconnected.

The link's security:

- `POST /relay/poll` and `POST /relay/result` need `Authorization: Bearer <CORTEX_MCP_KEY>`. The key is compared in constant time.
- A wrong key gets a 401, and the runner exits with code 2 instead of retrying.
- Without `CORTEX_MCP_KEY` on the server, the relay is disabled (503).
- The runner forwards only `Content-Type`, `Accept` and `X-Cortex-*` headers to the backend.
- The runner reaches the backend over `127.0.0.1`, so the backend's remote guard never sees it as remote.

### mcp-use tunnel: no deploy at all

`npm run dev -- --tunnel`, or `scripts/dev.sh --mcp-tunnel`, runs the MCP server locally and exposes it at `https://<name>.tunnel.mcp-use.run/mcp` through mcp-use's relay. You need no account, and nothing is deployed. The MCP server still reaches `localhost:8000` directly. The reservation is saved in `mcp/.mcp-use/state/tunnel.json` (git-ignored), so a restart reattaches to the **same** URL. We checked this on 2026-10-18: a tool call worked through the tunnel, and after a kill and restart the server came back at the same subdomain. The URL is only up while your laptop is running the server, and the endpoint has no auth (see below).

### Cloudflare quick tunnel (legacy)

`scripts/dev.sh --tunnel` exposes the backend with `cloudflared tunnel --url http://127.0.0.1:8000` and prints `https://<random>.trycloudflare.com/api`. Deploy with `--env CORTEX_API_URL=<that>/api --env CORTEX_MCP_KEY=<key>`. Remote requests can only reach `/api/health`, and `/api/tools*` with `Authorization: Bearer <CORTEX_MCP_KEY>`. The URL changes every time `cloudflared` restarts. When it does, run `npx mcp-use servers env add CORTEX_API_URL=<new url> --server <id>` and then `npx mcp-use deployments restart <id>`. Env changes only apply after a restart or a redeploy.

Notes:

- The Manufact free plan scales to zero, so expect a cold start on the first call. A runner that is polling keeps the server warm.
- **The deployed `/mcp` endpoint has no auth of its own.** Anyone with the URL can call the `cortex.*` tools through it, and with a runner connected those calls reach your laptop's backend: they can create projects, record simulations and run evaluations. Training still cannot launch without a human approval in the studio. Treat the URL as a secret, and stop the runner when you are done.
- The relay keeps its queue in memory. Run the hosted server as a single instance. A call that is in flight when the server restarts fails, and the runner reconnects on its own.

## Layout

```
index.ts         MCPServer setup (name "cortex"), registers tools and the prompt
src/backend.ts   POST /tools/{name} (runner or direct): headers (incl. agent identity), error mapping, approval handling; trackClients
src/relay.ts     /relay/poll and /relay/result: the authenticated long-poll link to the local runner
src/tools.ts     zod input schemas mirroring GET /api/tools, descriptions, annotations
src/prompts.ts   cortex.teach_task
```

The backend registry, `GET /api/tools`, is the source of truth. If a backend tool's arguments change, update `src/tools.ts` to match. The schemas there mirror the registry's, except that the enums the domain model enforces are spelled out: `cube_region`, `lighting`, `policy_type`, `executor` and `augmentation_preset`.
