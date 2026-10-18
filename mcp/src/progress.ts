import type { CallToolResult, RequestContext } from "mcp-use";
import { z } from "zod";
import { API_URL, backendHeaders, callTool, getJson, identityOf } from "./backend.js";

// Long-running Cortex tools (recording, analysis, dataset builds, evaluations, agent collections) start a
// background job. When the MCP request carries a progressToken, the tool call stays open, follows the job
// (the backend's per-project SSE stream, with get_job polling as a fallback) and sends
// notifications/progress until it finishes, then returns the final result. Without a progressToken
// (or with follow: false) the tool behaves exactly as before: it returns the job id after wait_s.

type Json = Record<string, unknown>;
type Ctx = Pick<RequestContext, "reportProgress" | "signal" | "request" | "client">;
type Job = { id: string; kind?: string; project_id?: string; status: string; progress?: number; message?: string; result_ref?: string | null; error?: string | null };

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
/** Upper bound for one followed call; past it the call returns the job id like the no-progress path. */
const FOLLOW_MAX_S = Number(process.env.CORTEX_MCP_FOLLOW_MAX_S ?? 1800);
/** Poll get_job at least this often, even while the SSE stream is healthy. */
const POLL_S = 5;

type Finish = (input: Json, data: Json, job: Job) => [string, Json] | undefined;
type LongJob = {
  /** The backend tool blocks for wait_s; following forces it to 0 so progress starts at once. */
  waitArg?: boolean;
  /** No job id: the backend call itself is slow, so only heartbeat progress while it runs. */
  heartbeatOnly?: boolean;
  /** Read tool whose result is appended once the job is done (default: none, the final job is enough). */
  finish?: Finish;
};

const collection: Finish = (_i, d) => (typeof d.collection_id === "string" ? ["get_collection_progress", { collection_id: d.collection_id }] : undefined);
const evaluation: Finish = (i, d) => {
  const pid = (i.project_id ?? d.project_id) as string | undefined;
  return pid ? ["compare_experiments", { project_id: pid }] : undefined;
};

/** Tool name (without the cortex. prefix) -> how to follow it. */
export const LONG_JOBS: Record<string, LongJob> = {
  record_demonstrations: { waitArg: true },
  analyze_dataset: { waitArg: true, finish: (i) => ["get_failure_clusters", { project_id: i.project_id }] },
  create_dataset_version: { waitArg: true },
  launch_simulation_evaluation: { waitArg: true, finish: evaluation },
  evaluate_learned_policy: { finish: evaluation },
  create_policy_project: { finish: collection },
  start_agent_collection: { finish: collection },
  improve_policy: { finish: collection },
  train_from_agent_demonstrations: {},
  import_demonstrations: {},
  build_simulation: { heartbeatOnly: true },
};

export const FOLLOW_NOTE =
  "\nPROGRESS: when your client sends a progressToken, this call stays open, streams notifications/progress " +
  `and returns only when the job has finished (up to ${Math.round(FOLLOW_MAX_S / 60)} min), so you do not need wait_s or ` +
  "cortex_get_job polling. Without a progressToken it returns the job id at once (or after wait_s): then poll cortex_get_job. " +
  "Pass follow: false if your client times out long calls despite progress.";

/** Adds the `follow` flag to a long tool's input schema. */
export function withFollow(schema: z.ZodObject): z.ZodObject {
  return schema.extend({
    follow: z.boolean().optional().describe(
      "Default true: if the client sent a progressToken, keep this call open and stream progress until the job finishes. " +
        "false = return the job id immediately (old behaviour).",
    ),
  });
}

function jobIdOf(data: Json): string | undefined {
  if (typeof data.job_id === "string") return data.job_id;
  const job = data.job as Json | undefined;
  if (job && typeof job.id === "string") return job.id;
  const refs = Array.isArray(data.ref_ids) ? data.ref_ids.map(String) : [];
  return refs.find((r) => r.startsWith("job_"));
}

function textOf(r: CallToolResult): string {
  return (r.content ?? []).map((c) => (c.type === "text" ? c.text : "")).filter(Boolean).join("\n");
}

/** Monotonic progress reporter (the spec requires progress to increase with every notification). */
function reporter(ctx: Ctx) {
  let last = 0;
  return async (fraction: number, message: string, done = false) => {
    const v = done ? 100 : Math.min(99.9, Math.max(last + 0.1, fraction * 100));
    last = v;
    await ctx.reportProgress(Math.round(v * 10) / 10, 100, message).catch(() => false);
  };
}

/** Subscribe to GET /projects/{pid}/events and call onJob for every `job` event. Never throws. */
function watchProject(projectId: string, signal: AbortSignal, onJob: (j: Job) => void): { healthy: () => boolean } {
  let lastEvent = 0;
  (async () => {
    try {
      const res = await fetch(`${API_URL}/projects/${encodeURIComponent(projectId)}/events`, {
        headers: backendHeaders("text/event-stream"),
        signal,
      });
      if (!res.ok || !res.body) return;
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) return;
        lastEvent = Date.now();
        buf += value;
        let i: number;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, i);
          buf = buf.slice(i + 2);
          const event = /^event: (.*)$/m.exec(frame)?.[1];
          const data = /^data: (.*)$/m.exec(frame)?.[1];
          if (event === "job" && data) {
            try {
              onJob(JSON.parse(data) as Job);
            } catch {
              /* ignore a malformed frame */
            }
          }
        }
      }
    } catch {
      /* aborted or unreachable: get_job polling covers it */
    }
  })();
  // The backend pings every 15 s, so 20 s of silence means the stream is gone.
  return { healthy: () => Date.now() - lastEvent < 20_000 };
}

async function getJob(id: string, signal: AbortSignal): Promise<Job | undefined> {
  try {
    return await getJson<Job>(`/jobs/${encodeURIComponent(id)}`, signal);
  } catch {
    return undefined;
  }
}

function describe(job: Job): string {
  const pct = typeof job.progress === "number" ? ` ${Math.round(job.progress * 100)}%` : "";
  return `${job.kind ?? "job"} ${job.id} ${job.status}${pct}${job.message ? `: ${job.message}` : ""}`;
}

/** Follow a job until it reaches a terminal status, the deadline passes or the request is cancelled. */
async function follow(jobId: string, projectId: string | undefined, ctx: Ctx, report: ReturnType<typeof reporter>): Promise<Job | undefined> {
  const stop = new AbortController();
  const abort = () => stop.abort();
  ctx.signal.addEventListener("abort", abort, { once: true });
  const deadline = Date.now() + FOLLOW_MAX_S * 1000;
  let job = await getJob(jobId, stop.signal);
  let wake: (() => void) | undefined;
  const sse = projectId
    ? watchProject(projectId, stop.signal, (j) => {
        if (j.id === jobId) {
          job = j;
          wake?.();
        }
      })
    : undefined;
  try {
    let lastPoll = Date.now();
    while (!(job && TERMINAL.has(job.status)) && Date.now() < deadline && !ctx.signal.aborted) {
      if (job) await report(job.progress ?? 0, describe(job));
      await new Promise<void>((resolve) => {
        wake = resolve;
        setTimeout(resolve, (sse?.healthy() ? POLL_S : 2) * 1000);
      });
      wake = undefined;
      if (!sse?.healthy() || Date.now() - lastPoll >= POLL_S * 1000) {
        job = (await getJob(jobId, stop.signal)) ?? job;
        lastPoll = Date.now();
      }
    }
    return job;
  } finally {
    ctx.signal.removeEventListener("abort", abort);
    stop.abort();
  }
}

/** Keep sending progress while a slow synchronous backend call runs. */
async function withHeartbeat<T>(p: Promise<T>, label: string, report: ReturnType<typeof reporter>): Promise<T> {
  const started = Date.now();
  let n = 0;
  const timer = setInterval(() => {
    n += 1;
    // Asymptotic toward 95%: there is no real progress signal for synchronous calls.
    void report(0.95 * (1 - Math.exp(-n / 20)), `${label}: running ${Math.round((Date.now() - started) / 1000)} s`);
  }, 3000);
  try {
    return await p;
  } finally {
    clearInterval(timer);
  }
}

/**
 * Run a long tool. `onDone(projectId)` fires after the job finished (resource update notifications).
 * Returns the plain backend result when the client sent no progressToken or passed follow: false.
 */
export async function runLong(
  name: string,
  backendName: string,
  input: Json,
  ctx: Ctx,
  onDone?: (projectId: string) => void,
): Promise<CallToolResult> {
  const spec = LONG_JOBS[name];
  const { follow: wantFollow, ...args } = input;
  const report = reporter(ctx);
  // reportProgress resolves false when the request carried no progressToken: keep the old behaviour.
  const tracking = spec !== undefined && wantFollow !== false && (await ctx.reportProgress(0, 100, `cortex.${name}: starting`));
  if (!tracking) return callTool(backendName, args, ctx.signal, identityOf(ctx));

  if (spec.heartbeatOnly) {
    const r = await withHeartbeat(callTool(backendName, args, ctx.signal, identityOf(ctx)), `cortex.${name}`, report);
    await report(1, `cortex.${name}: done`, true);
    return r;
  }

  const started = Date.now();
  const first = await callTool(backendName, spec.waitArg ? { ...args, wait_s: 0 } : args, ctx.signal, identityOf(ctx));
  const data = (first.structuredContent ?? {}) as Json;
  const jobId = first.isError ? undefined : jobIdOf(data);
  if (!jobId || data.approval_required === true) return first;

  const projectId = (args.project_id ?? data.project_id) as string | undefined;
  const job = await follow(jobId, projectId, ctx, report);
  const waited = Math.round((Date.now() - started) / 1000);
  const done = job !== undefined && TERMINAL.has(job.status);
  if (done) await report(1, describe(job), true);

  let finalText: string | undefined;
  let finalData: Json | undefined;
  const fin = done && job.status === "succeeded" ? spec.finish?.(args, data, job) : undefined;
  if (fin) {
    const r = await callTool(fin[0], fin[1], ctx.signal, identityOf(ctx));
    if (!r.isError) {
      finalText = `cortex.${fin[0]}: ${textOf(r)}`;
      finalData = { tool: `cortex.${fin[0]}`, ...((r.structuredContent ?? {}) as Json) };
    }
  }
  if (done && projectId) onDone?.(projectId);

  const headline = done
    ? `Job ${jobId} ${job.status} after ${waited} s${job.message ? `: ${job.message}` : ""}${job.error ? ` (error: ${job.error})` : ""}`
    : `Job ${jobId} still ${job?.status ?? "unknown"} after ${waited} s (follow limit or cancel): poll cortex_get_job with job_id ${jobId}.`;
  const refs = [...(Array.isArray(data.ref_ids) ? data.ref_ids.map(String) : [])];
  // result_ref is one id, or a comma-joined list (recordings return their episode ids).
  for (const r of job?.result_ref?.split(",") ?? []) if (r && !refs.includes(r)) refs.push(r);
  return {
    content: [{ type: "text", text: [textOf(first), headline, job?.result_ref && `result_ref: ${job.result_ref}`, finalText].filter(Boolean).join("\n") }],
    structuredContent: {
      ...data,
      summary: `${String(data.summary ?? `cortex.${name}`)} | ${headline}`,
      ref_ids: refs,
      job: job ?? data.job ?? null,
      followed: { job_id: jobId, finished: done, waited_s: waited, ...(finalData ? { final: finalData } : {}) },
    },
  };
}
