# Provider integrations and limitations

Status key: **verified** = exercised end to end in this repo; **implemented, unverified** = code
written against documented APIs but not yet run with real credentials; **fallback** = what runs
when the provider is not configured. `GET /api/health` reports the active provider for each subsystem.

| Area | Provider | Status | Fallback |
|---|---|---|---|
| Simulation | `lerobot/so101-tabletop-mujoco` @ `12e504e8` (EnvHub, Apache-2.0), MuJoCo 3.16 | **verified** on macOS arm64 | none needed |
| Datasets | LeRobotDataset v3 via `lerobot==0.6.1` | **verified** (written, reloaded by LeRobot) | none needed |
| Metadata DB | InsForge Postgres (REST records API, migrations API) | implemented, unverified | SQLite (**verified**) |
| Asset storage | InsForge storage, private bucket, presigned download URLs | implemented, unverified | local disk (**verified**) |
| VLM diagnostics | InsForge Model Gateway → OpenRouter → `anthropic/claude-sonnet-5.5` | implemented, unverified | VLM step skipped and disclosed; telemetry and quality analysis still run |
| Studio agent | same gateway, OpenAI-compatible tool calling | implemented, unverified | deterministic intent planner (**verified**) |
| Manipulation planner | GPT-6 Astra: InsForge gateway → OpenRouter `openai/gpt-6-astra` (catalog lists image input, tools, structured outputs); fallback OpenAI `gpt-6-astra` | planner loop **verified against a local stub**; no real Astra call yet (no key) | none: refuses with 503, no substitution; scripted oracle is a separate, labelled collector |
| Training | ACT via `lerobot-train` on GT Futurama Slurm over SSH | implemented, unverified (VPN was off) | manual command bundle + checkpoint registration |
| MCP | mcp-use v2 server, Manufact hosting | see [mcp/README.md](../mcp/README.md) | local MCP server |
| Evaluation workers | parallel local processes | **verified** | n/a |

## InsForge

- **Database:** `POST /api/database/migrations` applies [backend/migrations](../backend/migrations/)
  idempotently (`IF NOT EXISTS`, history checked first). CRUD goes through
  `/api/database/records/{table}` with PostgREST filters and upserts via
  `Prefer: resolution=merge-duplicates`.
- **Storage:** bucket `cortex-assets` is created private (`isPublic: false`). Objects are uploaded
  with `PUT /api/storage/buckets/{b}/objects/{key}`, and the browser gets presigned URLs from
  `download-strategy`. Self-hosted InsForge on local-disk storage cannot presign; Cortex then serves
  its local working copy instead.
- **Model Gateway:** the OpenRouter key comes from `GET /api/ai/openrouter/api-key` (InsForge Cloud),
  or from `OPENROUTER_API_KEY` when self-hosting. InsForge's own chat proxy is deprecated upstream,
  so Cortex calls OpenRouter directly with the key the gateway provisions.
- **Auth:** the backend uses the admin (`ik_`) key server-side only. The browser never receives it.
- **Limitation:** `project_members` (per-project roles and approval rights) exists in the schema, but
  this build is single-user and does not yet enforce row-level security.

## Training on Futurama

- **Submission:** [infra/futurama/train_act.sbatch](../infra/futurama/train_act.sbatch) runs on the
  `general` partition with 1× L40S for 2 h. Scratch lives in `/dev/shm`, checkpoints are written
  straight to home, and the `/nethome` path is remapped. Inputs are rsynced over the `futurama` SSH
  alias. Cortex polls with `squeue` (the accounting DB is down, so not `sacct`) and fetches
  `checkpoints/last/pretrained_model`.
- **Requirements:** the GT VPN must be connected. Training estimates are labelled estimates until the
  first run reports throughput.
- **Not built:** SmolVLA proposals are generated, but only ACT has the full launch path. HF Jobs is
  not implemented.

## MCP / Manufact

- **Tool surface:** tools are the backend registry (`GET /api/tools`) exposed as `cortex.*` names.
- **Hosted deploys:** a hosted server cannot reach localhost. Run `scripts/dev.sh --tunnel` (a
  Cloudflare quick tunnel) and deploy with `CORTEX_API_URL` and `CORTEX_MCP_KEY`.
- **Remote access:** remote callers can only reach `/api/tools`.

## GPT-6 Astra

- **Model availability:** checked on 2026-10-17. The OpenRouter catalog lists `openai/gpt-6-astra` with image,
  tools and structured outputs, at $10 / $50 per 1M input / output tokens. OpenAI's model page lists Chat
  Completions and Responses with image input and function calling.
- **No pointing format is documented**, so Cortex defines one: pixel coordinates on a labelled grid in a
  480×480 overhead image.
- **Recorded per attempt:** the provider route, the requested model, the model the API reports, and the
  endpoint actually called. Tests use a local stub endpoint, and those attempts record the stub's URL.
- **Not measured yet:** Astra's real pointing accuracy, demo success rate, calls and cost per demo. The
  tool layer was verified with simulated pointing noise of ±12 px (9/9 successes across all regions and
  lighting) and with an image-only stub planner.

## Not in this build

- **Physical SO-101:** [hardware/so101_bridge.py](../backend/cortex/hardware/so101_bridge.py) is written
  and unit-tested: safety limits, watchdog, operator/e-stop/calibration gate, and a refusal of unattended
  mode. It has never been connected to a robot, and the API and Autopilot refuse real-robot motion.
- **RL:** there is no RL experiment. The env has a reward, but nothing here claims RL fine-tunes a
  policy.
- **Blaxel:** removed. Its sandboxes are CPU-only, and local worker processes cover evaluation.
- **Human teleoperation in sim:** not built. Sim demos come from GPT-6 Astra (agent) or the scripted oracle.
