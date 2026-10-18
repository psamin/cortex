# Provider integrations and limitations

Status key: **verified** = exercised end to end in this repo; **implemented, unverified** = code
written against documented APIs but not yet run with real credentials; **fallback** = what runs
when the provider is not configured. `GET /api/health` reports the active provider for each subsystem.

| Area | Provider | Status | Fallback |
|---|---|---|---|
| Simulation | `lerobot/so101-tabletop-mujoco` @ `12e504e8` (EnvHub, Apache-2.0), MuJoCo 3.16 | **verified** on macOS arm64 | none needed |
| Datasets | LeRobotDataset v3 via `lerobot==0.6.1` | **verified** (written, reloaded by LeRobot) | none needed |
| Metadata DB | InsForge Postgres (REST records API, migrations API) | **verified** on InsForge Cloud 2.3.4 (us-west), 2026-10-17: migrations 001–003 applied, full e2e smoke 18/18 | SQLite (**verified**) |
| Asset storage | InsForge storage, private bucket, presigned download URLs | **verified**: private `cortex-assets` bucket, videos/npz/LeRobot files uploaded, signed `cdn.insforge.dev` URLs play without auth | local disk (**verified**) |
| VLM diagnostics | InsForge Model Gateway → OpenRouter → `anthropic/claude-sonnet-5.5` | **blocked by plan**: the project's org is on a free plan; `/api/ai/*` returns 403 "AI is only available on paid plans" | VLM step skipped and disclosed; telemetry and quality analysis still run; or the MCP client's own model (see MCP below) |
| Studio agent | same gateway, OpenAI-compatible tool calling | implemented, unverified | deterministic intent planner (**verified**) |
| Manipulation planner | **Default: our own model.** `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0), served by vLLM 0.31 on a Futurama L40S ([serve_vlm.sbatch](../infra/futurama/serve_vlm.sbatch), [vlm_futurama.sh](../scripts/vlm_futurama.sh)); no per-token cost. Alternative: GPT-6 Astra via OpenRouter (InsForge gateway or our own key) | planner loop **verified against a local stub**; no real Astra call yet: blocked until the InsForge org is on a paid plan | none: refuses with 503, no substitution; scripted oracle is a separate, labelled collector |
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
- **Row-level security:** tables created by raw-SQL migrations start with RLS off on InsForge. On
  2026-10-17 we verified that the public anon key could read `episodes`. Migration
  [004_enable_rls.sql](../backend/migrations/004_enable_rls.sql) enables RLS on all 19 tables with no
  policies. It was verified live: anon reads return 0 rows and anon writes return 401, while the
  backend's admin key, which bypasses RLS, works as before. `project_members` (per-project roles) exists
  in the schema but isn't enforced: this build is single-user.
- **Migrations:** InsForge rejects any migration version that isn't newer than the latest applied one.
  Cortex now skips a migration only if its name is already in the history; any other rejection fails loudly.
- **AI requires a paid org plan.** The free plan returns 403 `AI_UPSTREAM_UNAVAILABLE`. The self-serve paid
  plan is Pro, at $25/month or $240/year, upgraded via the dashboard or
  `npx @insforge/cli billing upgrade pro`.
  - **Included credit:** sources disagree on whether Pro includes any AI credit, so budget for
    pay-as-you-go.
  - **Spend caps:** paid orgs get uncapped gateway keys, so Cortex's own per-episode and per-project
    caps are the only spend limits.
  - **Hackathon credits:** InsForge has granted credits at past events, but nothing is published. Ask the
    sponsor team or their Discord.
  - **Current route:** until the org is upgraded, Cortex uses the team's own `OPENROUTER_API_KEY`, and
    attempts are labelled with that route.
- **Storage limits:** REST uploads cap at 200 MB per file (default 50 MB, buffered in server RAM). Files
  above `INSFORGE_REST_MAX_MB` (default 50) go through InsForge's S3-compatible gateway (`/storage/v1/s3`,
  path-style, SigV4, multipart in 16 MiB parts, aborted on failure) in
  [store/s3.py](../backend/cortex/store/s3.py). Registered ACT checkpoints (about 200 MB) are uploaded
  this way.
  - **Setup:** create project S3 access keys in the dashboard (Storage → Settings → S3 Configuration;
    the secret is shown once) and set `INSFORGE_S3_ACCESS_KEY_ID` / `INSFORGE_S3_SECRET_ACCESS_KEY`. The
    endpoint and region come from `GET /api/storage/s3/config`, or from `INSFORGE_S3_ENDPOINT` /
    `INSFORGE_S3_REGION`.
  - **Without keys:** large uploads fail with that instruction. A checkpoint still registers locally,
    and the audit row records that it wasn't uploaded.
  - **Status:** implemented, unverified. It's tested against AWS's SigV4 examples and a strict fake
    gateway, but no live upload has been made yet, and the shape of `/api/storage/s3/config` is assumed
    (`endpoint`, `region`).
  - **Plans:** free plans get 1 GB of storage and 5 GB/month of egress; Pro gets 100 GB and 250 GB.

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
- **Bring your own model (VLM diagnosis without a Cortex key):** `cortex.diagnose_failures_with_my_model`
  asks the MCP client's own model through sampling (`sampling/createMessage` embedded in an
  `input_required` result, protocol 2026-07-28; one request per keyframe contact sheet). mcp-use 2.10.1
  serves 2025-era requests statelessly, so only 2026-era clients that declare `sampling` can use it;
  others get a refusal before any work starts. Sampling is deprecated in 2026-07-28 (SEP-2577, still
  supported for at least 12 months). Fallback for every client: `cortex.get_diagnosis_inputs` (contact
  sheets + prompt as image content) then `cortex.submit_failure_hypotheses` (the agent's own answers).
  The backend validates every answer against the fixed taxonomy (one bad answer rejects the whole
  submission, nothing stored) and stores accepted ones as unverified hypotheses with `route`
  (`mcp_sampling` / `mcp_agent`) and the client-reported `model`
  ([external_diagnosis.py](../backend/cortex/services/external_diagnosis.py)).

## Self-hosted planner (our own model)

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0, ungated). It is trained for 2D grounding and GUI/tool
  agents and reports points in **relative 0–1000 coordinates**, per Qwen's cookbook
  `cookbooks/2d_grounding.ipynb` ("changed from the absolute coordinates used in Qwen2.5-VL to relative
  coordinates ranging from 0 to 1000").
- **Coordinate handling:** for this route, Cortex describes image points in 0–1000, labels the grid in the
  same units, converts to pixels before validation, and records both the model's point and the executed
  pixels in each trace step.
- **Serving:** vLLM's OpenAI-compatible server with `--enable-auto-tool-choice --tool-call-parser hermes` and
  up to 8 images per prompt. It runs on a Futurama compute node and is reached through an SSH tunnel.
- **Routing:** when `PLANNER_BASE_URL` is set, this route takes precedence over OpenRouter for planning, VLM
  diagnostics and the assistant. Cortex probes `/v1/models` first and refuses with a clear reason if the
  endpoint is down.
- **Unverified:** whether the `hermes` tool parser handles Qwen3-VL's tool-call format in vLLM 0.31, and the
  model's pointing accuracy on our scenes. The planner loop is verified against a local stub that answers in
  0–1000 coordinates.

## GPT-6 Astra (alternative)

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
