import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { resolveMediaUrl } from "../api/client";
import { Badge, Select, type Tone } from "../components/ui";
import { cn } from "../lib/format";
import type { AgentCall, CallStatus } from "./api";

const TONE: Record<CallStatus, Tone> = { running: "servo", ok: "pass", error: "fault", approval_required: "caution" };
const LABEL: Record<CallStatus, string> = { running: "running", ok: "ok", error: "error", approval_required: "approval required" };

function ms(n?: number) {
  if (n === undefined) return "";
  return n < 1000 ? `${n} ms` : `${(n / 1000).toFixed(n < 10000 ? 2 : 1)} s`;
}

function time(iso: string) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Where a human acts on an approval: training jobs live in Improve, the rest in the project chat. */
function approvalHref(c: AgentCall) {
  if (!c.project_id) return null;
  return "training_job_id" in c.args || /training/.test(c.tool) ? `/p/${c.project_id}/improve` : `/p/${c.project_id}`;
}

function CallCard({ c }: { c: AgentCall }) {
  const [open, setOpen] = useState(false);
  const approval = c.status === "approval_required" || c.needs_approval;
  const href = approval ? approvalHref(c) : null;
  const args = JSON.stringify(c.args ?? {}, null, 2);
  return (
    <li className="sk-depth relative rounded border border-line-strong bg-plate">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line px-3 py-2">
        <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", c.status === "running" ? "animate-pulse bg-servo" : c.status === "ok" ? "bg-pass" : c.status === "error" ? "bg-fault" : "bg-caution")} />
        <span className="font-mono text-[12.5px] text-ink">cortex.{c.tool}</span>
        <Badge tone={TONE[c.status]}>{LABEL[c.status]}</Badge>
        {c.needs_approval && c.status === "ok" && <Badge tone="caution">needs approval</Badge>}
        <span className="ml-auto flex items-center gap-3 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-3">
          {c.project_id && (
            <Link to={`/p/${c.project_id}`} className="normal-case tracking-normal text-ink-2 underline-offset-2 hover:text-ink hover:underline">
              {c.project_id}
            </Link>
          )}
          <span>{c.status === "running" ? "…" : ms(c.duration_ms)}</span>
          <span>{time(c.started_at)}</span>
        </span>
      </div>
      <div className="space-y-2 px-3 py-2.5">
        {(c.summary || c.error) && <p className={cn("text-[13px] leading-relaxed", c.error ? "text-fault" : "text-ink-2")}>{c.error ?? c.summary}</p>}
        {href && (
          <Link to={href} className="sk-pill sk-pill-primary inline-flex h-7 items-center px-3 text-xs font-medium">
            Review the approval
          </Link>
        )}
        {(c.image_urls ?? []).length > 0 && (
          <div className="flex flex-wrap gap-2">
            {c.image_urls!.map((u) => {
              const src = resolveMediaUrl(u) ?? u;
              return (
                <a key={u} href={src} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-sm border border-line-strong">
                  <img src={src} alt={`${c.tool} image`} className="h-28 w-28 object-cover" loading="lazy" />
                </a>
              );
            })}
          </div>
        )}
        <button type="button" onClick={() => setOpen((o) => !o)} className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-3 hover:text-ink">
          {open ? "Hide" : "Show"} arguments{c.result_preview ? " and result" : ""}
        </button>
        {open && (
          <div className="grid gap-2 md:grid-cols-2">
            <pre className="scroll-thin max-h-64 overflow-auto rounded-sm bg-rail p-2 font-mono text-[11.5px] leading-relaxed text-ink-2">{args}</pre>
            {c.result_preview && (
              <pre className="scroll-thin max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-sm bg-rail p-2 font-mono text-[11.5px] leading-relaxed text-ink-2">
                {pretty(c.result_preview)}
              </pre>
            )}
          </div>
        )}
      </div>
    </li>
  );
}

function pretty(s: string) {
  try {
    return JSON.stringify(JSON.parse(s), null, 2);
  } catch {
    return s; // truncated preview: show as is
  }
}

/** One card per tool call, newest first, with project / tool / errors-only filters. */
export function Timeline({ calls }: { calls: AgentCall[] }) {
  const [project, setProject] = useState("");
  const [tool, setTool] = useState("");
  const [errors, setErrors] = useState(false);
  const projects = useMemo(() => [...new Set(calls.map((c) => c.project_id).filter(Boolean) as string[])], [calls]);
  const tools = useMemo(() => [...new Set(calls.map((c) => c.tool))].sort(), [calls]);
  const shown = calls
    .filter((c) => (!project || c.project_id === project) && (!tool || c.tool === tool) && (!errors || c.status === "error" || c.status === "approval_required"))
    .slice()
    .reverse();
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Select aria-label="Filter by project" value={project} onChange={(e) => setProject(e.target.value)} className="h-8 w-auto min-w-[150px] text-xs">
          <option value="">All projects</option>
          {projects.map((p) => (
            <option key={p}>{p}</option>
          ))}
        </Select>
        <Select aria-label="Filter by tool" value={tool} onChange={(e) => setTool(e.target.value)} className="h-8 w-auto min-w-[150px] text-xs">
          <option value="">All tools</option>
          {tools.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </Select>
        <label className="inline-flex items-center gap-1.5 text-xs text-ink-2">
          <input type="checkbox" checked={errors} onChange={(e) => setErrors(e.target.checked)} /> Errors and approvals only
        </label>
        <span className="ml-auto font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-3">
          {shown.length} of {calls.length} calls
        </span>
      </div>
      {shown.length === 0 ? (
        <p className="py-8 text-center text-sm text-ink-3">{calls.length ? "No calls match these filters." : "No calls in this session yet."}</p>
      ) : (
        <ol className="space-y-3">
          {shown.map((c) => (
            <CallCard key={c.id} c={c} />
          ))}
        </ol>
      )}
    </div>
  );
}
