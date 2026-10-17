# Cortex Studio (web)

React studio for Cortex: collect demos in simulation, review quality, diagnose failures, version data, train with approval, and evaluate in parallel sim branches. It talks only to the Cortex REST API ([docs/API.md](../docs/API.md)); types mirror [backend/cortex/models.py](../backend/cortex/models.py).

## Commands

```bash
pnpm install          # Node 22+, pnpm 11 (esbuild build script is allowed in pnpm-workspace.yaml)
cp .env.example .env  # optional; defaults to http://127.0.0.1:8000/api
pnpm dev              # http://localhost:5173
pnpm build            # tsc typecheck + vite build -> dist/
pnpm typecheck        # tsc only
pnpm preview          # serve dist/
```

`VITE_API_URL` is the API base, including `/api`. Every request sends `X-Cortex-Actor: gui` so the audit log attributes it. The backend must allow the studio origin through CORS (`cors_origins` in backend settings, default `http://localhost:5173`).

## Layout

| Path | What |
|---|---|
| `src/api/types.ts` | Hand-written TS mirror of the Pydantic models plus `Job`, SSE payloads, `Health` |
| `src/api/client.ts` | Typed fetch client, `ApiError` (`unreachable` when the API is down), POST-SSE parser for the agent stream, media URL resolver |
| `src/api/events.tsx` | One `EventSource` per project (`/projects/{pid}/events`). `job` updates a live job store, `entity` invalidates the matching react-query keys, `log` lines are buffered per `job_id` |
| `src/api/hooks.ts` | react-query hooks; keys are `[resource, pid]` |
| `src/layout/` | App shell (header + `/health` provider pills + theme), project layout (left nav, persistent agent panel) |
| `src/agent/AgentPanel.tsx` | Streaming chat, tool-call/result chips with clickable ref ids, approval cards, Activity tab (`/audit`) |
| `src/pages/` | Home/onboarding, Autopilot overview (`Plan.tsx` + `Autopilot.tsx`), Lineage graph, Collect, Analyze, Improve (Recommendations, Datasets, Training, Evaluations) |
| `src/pages/AgentCollect.tsx` | Collect › Agent demonstrations / Scripted oracle: live MJPEG, live action timeline from SSE `agent_step`, dataset progress, attempts, Astra-vs-oracle comparison |
| `src/components/AgentTrace.tsx` | Trace rows (validator rejections show `error.code`), provenance badges, attempt drawer (video + full trace from `/agent-attempts/{aid}`) |
| `src/pages/Lineage.tsx` | Force-directed lineage graph (`react-force-graph-2d`) from `GET /projects/{pid}/graph`, provenance drawer, live refresh on SSE `entity` |
| `src/components/` | UI primitives, workcell grid (cube region × lighting), health pills |

Routes: `/` (projects or onboarding), `/new`, `/p/:pid` (autopilot overview), `/p/:pid/graph?node=<id>`, `/p/:pid/collect?mode=agent|oracle|manual` (agent is the default only when `/planner` reports available), `/p/:pid/analyze?ep=<id>`, `/p/:pid/improve?tab=recommendations|datasets|training|evaluations`.

## Honesty conventions

- Nothing is rendered unless it came from the API. Empty, loading and error (backend unreachable) states are explicit.
- Provenance marks: measured data uses solid edges; VLM output and hypothesis-backed recommendations use a dashed, hatched edge with an "unverified hypothesis" or "hypothesis" badge; local fallbacks (health pills, lineage provenance) use a dashed amber edge labelled "fallback", driven by the backend's explicit `fallback` flags.
- VLM confidence is shown as "model-reported, not a verified probability".
- The scripted expert is labelled "oracle reference, not a learned policy" everywhere.
- Training never launches without an explicit approval carrying an approver name.
