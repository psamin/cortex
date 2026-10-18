# Cortex REST contract (v1)

Base: `http://localhost:8000/api`. All bodies are JSON and use the Pydantic models in
[backend/cortex/models.py](../backend/cortex/models.py) (the source of truth for field names).
The MCP server calls these same endpoints, so the GUI and agents share one domain layer.

Long-running work returns a `Job` immediately and reports progress on the project event stream.

```ts
type Job = { id: string; kind: string; project_id: string; status: "queued"|"running"|"succeeded"|"failed";
             progress: number /*0..1*/; message: string; result_ref?: string; error?: string }
```

## Health
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{db, storage, vlm, executor, training, lerobot_version, sim_env}` — which provider or local fallback is active |

## Projects
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/projects` | `CreateProjectRequest` | `Project` |
| GET | `/projects` | | `Project[]` |
| GET | `/projects/{pid}/state` | | `ProjectState` (includes `next_actions`) |
| GET | `/projects/{pid}/events` | | **SSE**: `event: job` (Job), `event: entity` (`{table, id}` changed), `event: log` (`{job_id, line}`) |
| GET | `/projects/{pid}/audit` | | audit rows `{actor, tool, args, status, created_at}` |

## Collect
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/projects/{pid}/sim/frame?cube_region=&lighting=&seed=` | | `image/png` — live render of the scene |
| GET | `/projects/{pid}/sim/live` | | `multipart/x-mixed-replace` MJPEG of the currently recording episode (use as `<img src>`) |
| POST | `/projects/{pid}/recordings` | `{condition: Condition, seed?: int}` | `Job` (kind `record`) — records one demo with the scripted expert; streams to `/sim/live` |
| POST | `/projects/{pid}/recordings/batch` | `{conditions: Condition[], per_condition: int, base_seed?: int}` | `Job` |
| POST | `/recordings/{job_id}/stop` | | `Job` (cancelled, nothing saved) |
| GET | `/projects/{pid}/episodes` | | `Episode[]` |
| PATCH | `/episodes/{eid}` | `{status: "accepted"\|"rejected"}` | `Episode` |
| GET | `/episodes/{eid}/media` | | `{video_url, thumb_url, frames: number}` |

## Analyze
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/projects/{pid}/analyses` | `{use_vlm?: bool, max_vlm_episodes?: int}` | `Job` → `Analysis` |
| GET | `/analyses/{aid}` | | `Analysis` |
| GET | `/projects/{pid}/failure-clusters` | | `FailureCluster[]` |
| POST | `/analyses/{aid}/diagnoses/{eid}` | `{status: "human_confirmed"\|"human_rejected", human_label?: string}` | `EpisodeDiagnosis` |

## Improve
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/projects/{pid}/recommendations` | `{budget?: int}` | `Recommendation[]` (ranked) |
| PATCH | `/recommendations/{rid}` | `{status}` | `Recommendation` |
| POST | `/projects/{pid}/datasets` | `CreateDatasetVersionRequest` | `Job` → `DatasetVersion` |
| GET | `/projects/{pid}/datasets` | | `DatasetVersion[]` |
| POST | `/projects/{pid}/training/proposals` | `{dataset_version_id, policy_type?: "act"\|"smolvla", steps?: int}` | `TrainingJob` (status `proposed`, with `estimate`) |
| POST | `/training/{tid}/approve` | `{approver: string, approval_token?: string}` | `TrainingJob` (`approved`) |
| POST | `/training/{tid}/launch` | | `TrainingJob` (`queued`) — refuses unless `approved` |
| GET | `/projects/{pid}/training` | | `TrainingJob[]` |
| GET | `/projects/{pid}/checkpoints` | | `Checkpoint[]` |
| POST | `/projects/{pid}/evaluations` | `LaunchEvaluationRequest` | `EvaluationRun` (branches start in parallel) |
| GET | `/evaluations/{evid}` | | `{run: EvaluationRun, branches: SimulationBranch[]}` |
| GET | `/projects/{pid}/evaluations` | | same shape, list |
| POST | `/projects/{pid}/comparisons` | `{evaluation_ids: string[]}` | `ExperimentComparison` |
| GET | `/projects/{pid}/next-actions` | | `NextAction[]` |

## Agent
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/projects/{pid}/agent` | `{message: string, patch?: object}` | **SSE**, see "Chat events" below. `patch` is a one-click intake answer (a question option's or a feasibility suggestion's patch); it is applied with `update_intake` before the turn is planned. 404 for an unknown project. |
| GET | `/projects/{pid}/agent/history` | | `{role: "user" \| "assistant", content}` and `{role: "event", event, data}` rows, in order (inline images dropped; cards keep their media `url`) |

## Added after v1

Job `status` also includes `cancelled`. SSE adds `event: autopilot` (full `AutopilotRun` on each step).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/projects/{pid}/jobs` · `/jobs/{id}` | | `Job[]` · `Job` |
| POST | `/projects/{pid}/episodes/accept-passing` | | `{accepted}` (successful demos with quality score ≥ 0.6) |
| GET | `/projects/{pid}/analyses/latest` | | `Analysis \| null` |
| GET | `/projects/{pid}/recommendations` | | open `Recommendation[]` |
| GET | `/training/{tid}/command` | | `{command}` (manual training bundle) |
| POST | `/projects/{pid}/checkpoints` | `{path, training_job_id?, step?}` | `Checkpoint` (validates a LeRobot `pretrained_model/` dir) |
| GET | `/projects/{pid}/checkpoints` | | `Checkpoint[]` |
| GET | `/media?key=` | | `{url}` for any storage key (e.g. `EpisodeResult.video_key`) |
| GET | `/projects/{pid}/graph` | | `{nodes[{id,type,label,status,source{backend,table,row_id},meta}], edges[{source,target,kind}], backend}` |
| POST | `/projects/{pid}/autopilot/start` · `/stop` | | `AutopilotRun` |
| GET | `/projects/{pid}/autopilot` | | latest `AutopilotRun \| null` |
| PUT | `/projects/{pid}/autopilot/policy` | partial `AutopilotPolicy` | `AutopilotPolicy` |
| GET | `/tools` | | registry: `{name, description, input_schema, agent_callable}[]` |
| POST | `/tools/{name}` | tool args | `{summary, ref_ids, ...}`; 403 `{approval_required}` when gated |

`POST /projects` also accepts `autopilot` (policy) and `start_autopilot: true`.
`/health` returns `{db, storage, vlm, agent, executor, training}` each as `{provider, fallback}`, plus
`lerobot_version`, `sim_env`, `notes[]`.

## Agent demonstrations

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/planner` | | `{available, model, provider, route, pricing_per_m_tokens}` or `{available: false, reason}` |
| POST | `/projects/{pid}/agent-collections` | `StartAgentCollectionRequest` | `AgentCollection` (503 if Astra is not configured: no substitution; 403 if the planner budget is exhausted) |
| GET | `/projects/{pid}/agent-collections` | | `AgentCollection[]` |
| GET | `/agent-collections/{cid}` | | `{collection, attempts: AgentAttempt[], job}` |
| POST | `/agent-collections/{cid}/stop` | | `AgentCollection` |
| GET | `/projects/{pid}/agent-attempts` · `/agent-attempts/{aid}` | | attempts · `{attempt, trace[], model_summary, media}` |
| GET | `/projects/{pid}/collector-comparison` | | per-collector measured metrics |

SSE adds `agent_step` (`attempt_start` / `agent_step` / `agent_thought`, each carrying `collection_id` and `index`) and `agent_attempt` (a full `AgentAttempt`).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/agent-collections/{cid}/live` | | `{index, start, steps[]}`: the in-progress attempt, for reconnecting viewers |
| GET | `/projects/{pid}/planner-budget` | | `{budget_usd, spent_usd, remaining_usd}` |

Error bodies carry a `code`:
- 403 `{detail, code, approval_required}`, where `code` is one of `planner_budget`, `hardware_requires_operator`, `training_approval`, `approval_required`.
- 503 `{detail, code: "planner_unavailable"}`.

`StartAgentCollectionRequest.task` optionally overrides the planner's task text. Oracle collections are recorded with `mode: "privileged"`.

## Chat events (`POST /projects/{pid}/agent`)

The studio chat streams one turn as SSE frames. Projects with an intake use the guided planner
(`services/guide.py`) when no model is configured; with a model, the LLM gets the same plan in its
system prompt and the same cards are emitted for its tool results. Classic projects keep the old
intent planner.

| Event | Data | Meaning |
|---|---|---|
| `text` | `{delta}` | Assistant prose |
| `tool_call` | `{tool, args}` | A registry tool the assistant is calling |
| `tool_result` | `{tool, summary, ref_ids}` | Its result |
| `step` | `{step, index, label, steps}` | Pipeline position: `intake`, `simulate`, `collect`, `train`, `evaluate` |
| `card` | one of the card objects below | Rich message |
| `approval_required` | `{kind: "training" \| "evaluation", ref_id, summary, args}` | Needs a human click; the studio calls `POST /training/{id}/approve` + `/launch`, or `POST /projects/{pid}/evaluations` |
| `done` | `{}` | End of the turn |

Cards (`card.type`):

| Type | Fields |
|---|---|
| `question` | `path, prompt, why, question_type, options: [{label, patch}], allow_text` |
| `image` | `label` (e.g. `FIG.02 SCENE · TOP CAMERA`), `camera, url, key, scene, image_jpeg_base64?` |
| `feasibility` | `ok, summary, jaw_opening_m, objects: [{name, ok, grasp, payload, tall, grasp_mode}], places: [{name, kind, ok, grasp_err_mm, carry_err_mm}], suggestions: [{label, patch}]` |
| `job` | `job_id, kind, title, job` (a `Job` snapshot; live updates arrive as `event: job` on `/projects/{pid}/events`) |
| `training_manual` | `training_job_id, backend, command, checkpoint_hint, smoke_available, smoke_reason, smoke_steps` (no GPU cluster: run `command`, then send the checkpoint path as a chat message to register it) |
| `results` | `evaluation_id, policy, checkpoint_id, checkpoint_step, smoke, branches: [{name, n, successes, success_rate, ci95}]` |
| `actions` | `actions: [{label, message}]` (quick replies sent as chat messages) |

Free-text replies to an open intake question are mapped onto it (option names, an option number,
"yes"/"confirm", sizes like `4 cm` and masses like `50 g`, catalogue object names); anything else
re-asks with the options. "Run the CPU smoke test" starts a labelled 20-step `lerobot-train
--policy.device=cpu` run (job kind `train_smoke`); its checkpoint carries `metrics.cpu_smoke_run = 1`
and is never attached to the GPU training job.
