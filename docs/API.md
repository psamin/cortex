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
| POST | `/projects/{pid}/agent` | `{message: string}` | **SSE**: `event: text` (`{delta}`), `event: tool_call` (`{tool, args}`), `event: tool_result` (`{tool, summary, ref_ids}`), `event: approval_required` (`{kind, ref_id, summary}`), `event: done` |
| GET | `/projects/{pid}/agent/history` | | `{role, content, tool?}[]` |
| GET | `/projects/{pid}/agent/state` | | `{running, thread, status, resuming, waiting, paused}`: whether a turn really runs in the backend, and a pause that needs Continue |
| POST | `/projects/{pid}/agent/continue` | | `{resumed}`: the Continue button; the autonomous workflow picks up where it stopped |

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

## Cycle protocol (PRODUCT_SPEC 3.2)

A cycle is a **demo** segment (home → pick the object from its start → place it in the task zone → home)
followed by a **reset** segment (home → pick it from the zone → place it at the next start → home). The next
demo starts wherever the reset left the object; only the first placement of a run teleports the object, and
that teleport is labelled as the operator's reset. After a failed segment a **recover** segment puts the object
back at a valid start; if recovery fails twice, the run ends with `stop_reason` set.

Where a reset puts the object comes from the intake's `task.reset_policy`:
- `return_to_origin`: the object's nominal start;
- `cycle_zones`: the extra place zones (`task.place[1:]`) in turn;
- `random_zone`: a seeded random extra zone, or, without extra zones, a seeded random reachable spot 3–5 cm
  from the origin.

Each successful segment is its own episode (the same `episode.npz` layout), with `Episode.segment` set to
`demo`, `reset` or `recover` (existing episodes default to `demo`). Its `provenance` carries `run_id`,
`cycle_id`, `cycle`, `segment_index`, `goal`, `target`, `start_pose`, `end_pose` and `reset_policy`. Failed
segments are kept as diagnostics (video and `result.json`, plus an `AgentAttempt` for collections) and never
become episodes.

`CreateDatasetVersionRequest.segments` picks which segments go into a dataset version. The default is
`["demo", "reset"]`, so recover segments are excluded unless asked for. The choice and the per-segment counts are
written to the dataset lineage (`meta/cortex_lineage.json`, key `segments`).

Starting a run (service level for now; the request models don't carry it yet):
`recording_service.start_recording(..., cycles=N)` for the scripted oracle, and
`agent_collection_service.start_collection(..., cycles=N)` for either collector. Both need the project's intake
scene.

**Worker events** (JSON lines on the worker's stdout, next to the existing `episode_start` / `episode` /
`progress`):

| `type` | Fields |
|---|---|
| `operator_reset` | `reason`, `pose`: the run's first placement (a sim teleport standing in for the operator) |
| `segment_start` | `cycle`, `segment`, `segment_index`, `goal`, `target` (`{zone}` or `{xy, label}`), `total_cycles`, `start_pose` |
| `segment` | the same fields plus `success`, `recorded`, `steps`, `num_frames`, `failure_reason`, `end_pose`, `wall_s`, `out_dir` |
| `operator_reset_required` | `reason`, `pose`: recovery failed twice |
| `cycles_done` | `cycles`, `segments`, `recorded`, `by_segment` (`{demo, reset, recover}` counts), `stop_reason`, `reset_policy` |

A pose is `{xyz, yaw, upright, arm}` (object position in metres, yaw in radians, arm joints in degrees).
Agent collections relay `segment_start` on the project stream as `agent_step` (with `index` = `segment_index`),
and each segment's `AgentAttempt` as `agent_attempt`.
