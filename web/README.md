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

The product is one chat-first screen per project (`/p/:pid`): the agent conversation on the left (35%), the MuJoCo simulation viewer on the right (65%), and a collapsible "Autonomous learning pipeline" panel underneath (stages, metrics, success across iterations, failure analysis). The older pages (Overview, Collect, Analyze, Improve) are secondary, under the header's Details menu.

| Path | What |
|---|---|
| `src/chat/ChatProvider.tsx` | Conversation state at project level (survives navigating to Details): loads `/agent/history`, streams `POST /agent` SSE, queues follow-ups, New chat |
| `src/chat/model.ts` | Pure reducer: SSE + history → items; friendly tool labels; hides Claude Code internals; never shows raw JSON |
| `src/chat/Conversation.tsx` | Messages, tool status rows, inline result + approval cards, pinned pending approvals, composer |
| `src/chat/cards.tsx` | Result cards by id prefix (ac_, ev_, ds_, tj_, an_, rec_, job_, ep_), training/evaluation approvals, budget raise |
| `src/chat/Viewer.tsx` | Simulation viewer: live MJPEG, replays with play/scrub, demo/eval/failure filters, cameras, success-vs-failure compare |
| `src/chat/rollouts.ts` | `/rollouts` + `/iterations` (404-tolerant; falls back to real episodes + evaluation results) |
| `src/chat/Pipeline.tsx` | Pipeline stages from the autopilot state machine + live jobs, measured metrics, iteration chart, failure analysis |
| `src/api/` | Typed client, SSE event store, react-query hooks |
| `src/pages/` | Landing (`Home.tsx`) and the secondary detail pages |

Routes: `/` (landing prompt + recent projects), `/p/:pid` (chat), `/p/:pid/overview|collect|analyze|improve` (details).

## Honesty conventions

- Nothing is rendered unless it came from the API. Empty, loading and error (backend unreachable) states are explicit.
- Provenance marks: measured data uses solid edges; VLM output and hypothesis-backed recommendations are labelled as hypotheses; local fallbacks are listed and labelled in the header's service popover, driven by the backend's explicit `fallback` flags.
- Pipeline metrics show "—" when a value has not been measured; planned iterations stay "(pending)" until a checkpoint exists.
- VLM confidence is shown as "model-reported, not a verified probability".
- The scripted expert is labelled "oracle reference, not a learned policy" everywhere.
- Training never launches without an explicit approval carrying an approver name.
