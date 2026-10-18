import type { MCPServer, ReadResourceResult } from "mcp-use";
import { callTool, getJson } from "./backend.js";

// Read-only MCP resources over the Cortex backend's GET endpoints, so clients can attach project state,
// episodes, dataset lineage, evaluations and a policy model card as context without spending tool calls.
// URIs: cortex://projects[/{project_id}[/episodes|/intake|/datasets/{version}/lineage|/evaluations/{id}|/policy/model-card]]
// and cortex://jobs/{job_id}. Ids complete via completion/complete.

type Json = Record<string, unknown>;
type Project = { id: string; name?: string; mode?: string; task_prompt?: string; created_at?: string };
type Dataset = { id: string; name?: string; parent_id?: string | null; links?: { episode_id: string; split: string; augmentation: unknown }[]; augmentations?: unknown[]; eval_fraction?: number; split_seed?: number; imported_from?: unknown; num_frames?: number; created_at?: string; lerobot_root?: string };
type Episode = { id: string; source?: string; condition?: Json; seed?: number; success?: boolean; status?: string; num_frames?: number };
type Branch = { spec?: { name?: string }; status?: string; seeds?: number[]; result?: { n: number; successes: number; success_rate: number; ci95: [number, number]; mean_reward: number; wall_s?: number; episodes?: unknown[] } | null };
type Evaluation = { run: { id: string; name?: string; policy?: { kind: string; checkpoint_id?: string | null }; policy_label?: string; status?: string; dataset_version_id?: string | null; created_at?: string; job_id?: string | null }; branches: Branch[] };
type TrainingJob = { id: string; dataset_version_id: string; policy_type: string; backend: string; status: string; config?: Json; estimate?: Json; checkpoint_ids?: string[]; metrics?: Json; approved_by?: string | null; created_at?: string };
type Checkpoint = { id: string; training_job_id?: string | null; policy_type: string; step: number; path: string; metrics?: Json; created_at?: string };

const pid = (v: string | string[]) => encodeURIComponent(Array.isArray(v) ? v[0] : v);
const one = (v: string | string[]) => (Array.isArray(v) ? v[0] : v);

const json = (uri: URL, data: unknown): ReadResourceResult => ({
  contents: [{ uri: uri.href, mimeType: "application/json", text: JSON.stringify(data, null, 2) }],
});
const markdown = (uri: URL, text: string): ReadResourceResult => ({
  contents: [{ uri: uri.href, mimeType: "text/markdown", text }],
});

// Completion lookups run on every keystroke in some clients: cache the backend lists briefly.
const cache = new Map<string, { at: number; value: Promise<unknown> }>();
function cached<T>(path: string): Promise<T> {
  const hit = cache.get(path);
  if (hit && Date.now() - hit.at < 5000) return hit.value as Promise<T>;
  const value = getJson<T>(path).catch((e) => {
    cache.delete(path);
    throw e;
  });
  cache.set(path, { at: Date.now(), value });
  return value;
}

function prefix(values: string[], typed: string): string[] {
  const t = typed.toLowerCase();
  return values.filter((v) => v.toLowerCase().startsWith(t)).slice(0, 100);
}

async function completeProject(value: string): Promise<string[]> {
  try {
    return prefix((await cached<Project[]>("/projects")).map((p) => p.id), value);
  } catch {
    return [];
  }
}

async function completeIn(value: string, context: { arguments?: Record<string, string> } | undefined, list: (p: string) => Promise<string[]>) {
  const p = context?.arguments?.project_id;
  if (!p) return [];
  try {
    return prefix(await list(p), value);
  } catch {
    return [];
  }
}

const datasetsOf = async (p: string) => (await cached<Dataset[]>(`/projects/${encodeURIComponent(p)}/datasets`)).map((d) => d.id);
const evaluationsOf = async (p: string) => (await cached<Evaluation[]>(`/projects/${encodeURIComponent(p)}/evaluations`)).map((e) => e.run.id);

function branchSummary(b: Branch) {
  const r = b.result;
  return {
    branch: b.spec?.name,
    status: b.status,
    ...(r ? { successes: r.successes, n: r.n, success_rate: r.success_rate, ci95: r.ci95, mean_reward: r.mean_reward } : {}),
  };
}

async function lineage(project: string, version: string): Promise<Json> {
  const datasets = await getJson<Dataset[]>(`/projects/${encodeURIComponent(project)}/datasets`);
  const byId = new Map(datasets.map((d) => [d.id, d]));
  const ds = byId.get(version) ?? datasets.find((d) => d.name === version);
  if (!ds) throw new Error(`dataset version ${version} not found in project ${project} (have: ${datasets.map((d) => d.id).join(", ") || "none"})`);
  const episodes = new Map((await getJson<Episode[]>(`/projects/${encodeURIComponent(project)}/episodes`)).map((e) => [e.id, e]));
  const ancestors: { id: string; name?: string; created_at?: string }[] = [];
  for (let p = ds.parent_id; p && ancestors.length < 50; p = byId.get(p)?.parent_id) {
    ancestors.push({ id: p, name: byId.get(p)?.name, created_at: byId.get(p)?.created_at });
  }
  const links = (ds.links ?? []).map((l) => {
    const e = episodes.get(l.episode_id);
    return { episode_id: l.episode_id, split: l.split, augmentation: l.augmentation, source: e?.source, condition: e?.condition, seed: e?.seed, success: e?.success };
  });
  const count = (f: (l: (typeof links)[number]) => unknown) =>
    links.reduce<Record<string, number>>((acc, l) => ((acc[String(f(l) ?? "none")] = (acc[String(f(l) ?? "none")] ?? 0) + 1), acc), {});
  return {
    dataset_version_id: ds.id,
    name: ds.name,
    created_at: ds.created_at,
    parents: ancestors,
    imported_from: ds.imported_from ?? null,
    augmentations: ds.augmentations ?? [],
    eval_fraction: ds.eval_fraction,
    split_seed: ds.split_seed,
    num_frames: ds.num_frames,
    lerobot_root: ds.lerobot_root,
    counts: { by_split: count((l) => l.split), by_source: count((l) => l.source), by_augmentation: count((l) => (l.augmentation ? "augmented" : "original")) },
    episodes: links,
  };
}

/** A model card: the backend has none yet, so summarize the best checkpoint and its measured evaluations. */
async function modelCard(project: string): Promise<string> {
  const p = encodeURIComponent(project);
  const [proj, training, checkpoints, evaluations, datasets] = await Promise.all([
    getJson<Project>(`/projects/${p}`),
    getJson<TrainingJob[]>(`/projects/${p}/training`),
    getJson<Checkpoint[]>(`/projects/${p}/checkpoints`),
    getJson<Evaluation[]>(`/projects/${p}/evaluations`),
    getJson<Dataset[]>(`/projects/${p}/datasets`),
  ]);
  const learned = evaluations.filter((e) => e.run.policy?.kind === "checkpoint" && e.run.policy.checkpoint_id);
  const meanRate = (e: Evaluation) => {
    const rs = e.branches.map((b) => b.result).filter((r) => r != null);
    return rs.length ? rs.reduce((s, r) => s + r.successes, 0) / Math.max(1, rs.reduce((s, r) => s + r.n, 0)) : -1;
  };
  const bestEval = [...learned].sort((a, b) => meanRate(b) - meanRate(a))[0];
  const best = bestEval
    ? checkpoints.find((c) => c.id === bestEval.run.policy?.checkpoint_id)
    : [...checkpoints].sort((a, b) => b.step - a.step || String(b.created_at).localeCompare(String(a.created_at)))[0];
  const lines: string[] = [`# Model card: ${proj.name ?? project}`, "", `Project \`${project}\` (${proj.mode ?? "sim"}). Task: ${proj.task_prompt ?? "?"}`, ""];
  lines.push("> Generated by the Cortex MCP server from measured backend records. Cortex has no stored model card yet;", "> this summarizes the best checkpoint and its simulator evaluations. Success rates are simulator results only.", "");
  if (!best) {
    lines.push("## Status", "", "No trained checkpoint yet.", "");
    const pending = training.map((t) => `- \`${t.id}\` ${t.policy_type} on \`${t.dataset_version_id}\`: **${t.status}**${t.status === "proposed" ? " (needs human approval in the studio)" : ""}`);
    lines.push(pending.length ? "Training jobs:" : "No training jobs proposed yet.", ...pending, "");
  } else {
    const tj = training.find((t) => t.id === best.training_job_id);
    const ds = datasets.find((d) => d.id === tj?.dataset_version_id);
    lines.push("## Policy", "", `- Checkpoint: \`${best.id}\` (${best.policy_type}, step ${best.step})`, `- Path: \`${best.path}\``);
    if (tj) lines.push(`- Training job: \`${tj.id}\` on ${tj.backend}, status ${tj.status}${tj.approved_by ? `, approved by ${tj.approved_by}` : ""}`);
    if (tj?.config) lines.push(`- Config: \`${JSON.stringify(tj.config).slice(0, 400)}\``);
    if (best.metrics && Object.keys(best.metrics).length) lines.push(`- Checkpoint metrics: \`${JSON.stringify(best.metrics)}\``);
    lines.push("", "## Training data", "");
    if (ds) {
      const splits = (ds.links ?? []).reduce<Record<string, number>>((a, l) => ((a[l.split] = (a[l.split] ?? 0) + 1), a), {});
      lines.push(`- Dataset version \`${ds.id}\` (${ds.name ?? ""}): ${ds.links?.length ?? 0} episodes ${JSON.stringify(splits)}, ${ds.num_frames ?? "?"} frames`, `- Augmentations: ${JSON.stringify(ds.augmentations ?? [])}`, `- Lineage: cortex://projects/${project}/datasets/${ds.id}/lineage`);
    } else lines.push("- Dataset version unknown.");
    lines.push("");
  }
  const relevant = best ? learned.filter((e) => e.run.policy?.checkpoint_id === best.id) : [];
  const reference = evaluations.filter((e) => e.run.policy?.kind === "scripted_expert");
  lines.push("## Evaluations (MuJoCo, held-out seeds)", "");
  if (!relevant.length) lines.push(best ? "This checkpoint has not been evaluated yet: run cortex.evaluate_learned_policy." : "No learned-policy evaluations yet.");
  for (const e of relevant) lines.push(...evalLines(project, e));
  if (reference.length) {
    lines.push("", "### Scripted-expert reference (oracle, privileged state; not a learned policy)", "");
    for (const e of reference.slice(-2)) lines.push(...evalLines(project, e));
  }
  lines.push("", "## Limitations", "", "- SO-101 arm, simulation only (MuJoCo); no automatic sim-to-real transfer.", "- VLM failure clusters are hypotheses unless verified by a human.");
  return lines.join("\n");
}

function evalLines(project: string, e: Evaluation): string[] {
  const out = [`- \`${e.run.id}\` ${e.run.policy_label ?? ""} (${e.run.status ?? "?"}), cortex://projects/${project}/evaluations/${e.run.id}`];
  for (const b of e.branches) {
    const r = b.result;
    out.push(r ? `  - ${b.spec?.name}: ${r.successes}/${r.n} = ${(r.success_rate * 100).toFixed(0)}% (95% CI ${(r.ci95[0] * 100).toFixed(0)}-${(r.ci95[1] * 100).toFixed(0)}%)` : `  - ${b.spec?.name}: ${b.status}`);
  }
  return out;
}

export function projectUris(project: string): string[] {
  const base = `cortex://projects/${project}`;
  return [base, `${base}/episodes`, `${base}/intake`, `${base}/policy/model-card`];
}

/** Tell subscribed clients that a project's resources changed. Never throws. */
export function touchProject(server: MCPServer, project: string, created = false) {
  for (const k of [...cache.keys()]) if (k === "/projects" || k.includes(project)) cache.delete(k);
  const fire = async () => {
    if (created) await server.notifyResourcesChanged();
    for (const uri of projectUris(project)) await server.notifyResourceUpdated(uri);
  };
  void fire().catch(() => undefined);
}

/** Registers the cortex:// resources and resource templates. */
export function registerResources(server: MCPServer) {
  server.resource(
    { name: "projects", uri: "cortex://projects", title: "Cortex projects", description: "Every project: id, name, mode, task prompt. Newest first.", mimeType: "application/json" },
    async (uri) => json(uri, (await getJson<Project[]>("/projects")).map(({ id, name, mode, task_prompt, created_at }) => ({ id, name, mode, task_prompt, created_at, uri: `cortex://projects/${id}` }))),
  );

  server.resourceTemplate(
    {
      name: "project_state",
      uriTemplate: "cortex://projects/{project_id}",
      title: "Project state",
      description: "Same snapshot as cortex.get_project_state: plan, episode counts, datasets, training jobs, evaluations, recommendations, next actions.",
      mimeType: "application/json",
      complete: { project_id: completeProject },
    },
    async (uri, { project_id }) => json(uri, await getJson(`/projects/${pid(project_id)}/state`)),
  );

  server.resourceTemplate(
    {
      name: "project_episodes",
      uriTemplate: "cortex://projects/{project_id}/episodes",
      title: "Episodes",
      description: "Recorded demonstrations: id, condition, seed, success, status, frames, quality score and flags (no video data).",
      mimeType: "application/json",
      complete: { project_id: completeProject },
    },
    async (uri, { project_id }) => {
      const r = await callTool("list_episodes", { project_id: one(project_id) });
      if (r.isError) throw new Error((r.content?.[0] as { text?: string })?.text ?? "list_episodes failed");
      return json(uri, (r.structuredContent as Json).episodes);
    },
  );

  server.resourceTemplate(
    {
      name: "project_intake",
      uriTemplate: "cortex://projects/{project_id}/intake",
      title: "Intake spec",
      description: "The custom-scene intake: spec, open questions, completeness, whether the sim scene is built.",
      mimeType: "application/json",
      complete: { project_id: completeProject },
    },
    async (uri, { project_id }) => json(uri, await getJson(`/projects/${pid(project_id)}/intake`)),
  );

  server.resourceTemplate(
    {
      name: "dataset_lineage",
      uriTemplate: "cortex://projects/{project_id}/datasets/{version}/lineage",
      title: "Dataset version lineage",
      description: "Where a dataset version came from: parent versions, import source, augmentations, split, and every episode with its source, condition and seed.",
      mimeType: "application/json",
      complete: { project_id: completeProject, version: (v, c) => completeIn(v, c, datasetsOf) },
    },
    async (uri, { project_id, version }) => json(uri, await lineage(one(project_id), one(version))),
  );

  server.resourceTemplate(
    {
      name: "evaluation",
      uriTemplate: "cortex://projects/{project_id}/evaluations/{evaluation_id}",
      title: "Evaluation run",
      description: "One simulator evaluation: policy, status and per-branch successes with 95% CIs (per-episode results included).",
      mimeType: "application/json",
      complete: { project_id: completeProject, evaluation_id: (v, c) => completeIn(v, c, evaluationsOf) },
    },
    async (uri, { evaluation_id }) => {
      const e = await getJson<Evaluation>(`/evaluations/${pid(evaluation_id)}`);
      return json(uri, { run: e.run, summary: e.branches.map(branchSummary), branches: e.branches });
    },
  );

  server.resourceTemplate(
    {
      name: "policy_model_card",
      uriTemplate: "cortex://projects/{project_id}/policy/model-card",
      title: "Policy model card",
      description: "Markdown model card for the project's best checkpoint: training config, dataset lineage, measured simulator evaluations, limitations.",
      mimeType: "text/markdown",
      complete: { project_id: completeProject },
    },
    async (uri, { project_id }) => markdown(uri, await modelCard(one(project_id))),
  );

  server.resourceTemplate(
    {
      name: "job",
      uriTemplate: "cortex://jobs/{job_id}",
      title: "Background job",
      description: "Status, progress (0-1), message and result_ref of a background job.",
      mimeType: "application/json",
    },
    async (uri, { job_id }) => json(uri, await getJson(`/jobs/${pid(job_id)}`)),
  );
}
