import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Cpu, Rocket, ShieldAlert, Terminal } from "lucide-react";
import { api } from "../../api/client";
import { useLogs } from "../../api/events";
import { useCheckpoints, useDatasets, useTraining } from "../../api/hooks";
import type { PolicyType, TrainingJob } from "../../api/types";
import { useApprover } from "../../components/refs";
import { ManualTraining } from "../Autopilot";
import { Badge, Button, EmptyState, ErrorState, Field, Input, Panel, QueryState, Segmented, Select, SkeletonRows, StatusBadge, InlineError } from "../../components/ui";
import { ago, cn, evidenceValue, usd } from "../../lib/format";

export function TrainingTab() {
  const { pid = "" } = useParams();
  const training = useTraining(pid);
  return (
    <div className="space-y-4">
      <ProposeForm pid={pid} />
      <QueryState q={training} skeleton={<SkeletonRows rows={3} />}>
        {(jobs) =>
          jobs.length === 0 ? (
            <Panel>
              <EmptyState icon={<Cpu className="h-5 w-5" />} title="No training runs yet">
                Nothing launches without a named approver.
              </EmptyState>
            </Panel>
          ) : (
            <div className="space-y-3">
              {[...jobs]
                .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
                .map((t) => (
                  <TrainingCard key={t.id} t={t} pid={pid} />
                ))}
            </div>
          )
        }
      </QueryState>
      <Checkpoints pid={pid} />
    </div>
  );
}

function ProposeForm({ pid }: { pid: string }) {
  const qc = useQueryClient();
  const datasets = useDatasets(pid);
  const [ds, setDs] = useState("");
  const [policy, setPolicy] = useState<PolicyType>("act");
  const [steps, setSteps] = useState("");
  const latest = datasets.data?.length ? [...datasets.data].sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0].id : "";
  const chosen = ds || latest;
  const m = useMutation({
    mutationFn: () => api.proposeTraining(pid, chosen, policy, steps === "" ? undefined : Number(steps)),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["training", pid] }),
  });
  return (
    <Panel title="Propose training" meta="estimate shown first">
      {datasets.data && datasets.data.length === 0 ? (
        <p className="text-xs text-ink-3">Create a dataset first (Datasets tab).</p>
      ) : (
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-[240px]">
            <Field label="Dataset version">
              <Select value={chosen} onChange={(e) => setDs(e.target.value)} disabled={!datasets.data}>
                {(datasets.data ?? []).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} · {d.id}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Field label="Policy">
            <Segmented
              value={policy}
              onChange={setPolicy}
              ariaLabel="Policy type"
              options={[
                { value: "act", label: "ACT" },
                { value: "smolvla", label: "SmolVLA" },
              ]}
            />
          </Field>
          <div className="w-[120px]">
            <Field label="Steps">
              <Input type="number" min={100} value={steps} onChange={(e) => setSteps(e.target.value)} placeholder="auto" className="num font-mono" />
            </Field>
          </div>
          <Button variant="primary" disabled={!chosen} loading={m.isPending} onClick={() => m.mutate()}>
            Propose
          </Button>
        </div>
      )}
      {m.error && (
        <div className="mt-3">
          <ErrorState error={m.error} compact />
        </div>
      )}
    </Panel>
  );
}

function TrainingCard({ t, pid }: { t: TrainingJob; pid: string }) {
  const qc = useQueryClient();
  const [approver, setApprover] = useApprover();
  const onDone = () => qc.invalidateQueries({ queryKey: ["training", pid] });
  const approve = useMutation({ mutationFn: () => api.approveTraining(t.id, approver.trim()), onSuccess: onDone });
  const launch = useMutation({ mutationFn: () => api.launchTraining(t.id), onSuccess: onDone });
  const live = t.status === "queued" || t.status === "running" || t.status === "succeeded" || t.status === "failed";
  const e = t.estimate;

  return (
    <article className="rounded-md bg-plate border border-line">
      <header className="flex flex-wrap items-center gap-2 border-b border-line-soft px-3.5 py-2.5">
        <span className="text-sm font-medium uppercase">{t.policy_type}</span>
        <span className="font-mono text-2xs text-ink-3">{t.id}</span>
        <StatusBadge status={t.status} />
        <span className="text-2xs text-ink-3">
          on <span className="font-mono">{t.dataset_version_id}</span> · {t.backend.replace("_", " ")}
        </span>
        <span className="ml-auto text-2xs text-ink-4">{ago(t.updated_at)}</span>
      </header>

      <div className="grid grid-cols-1 gap-4 p-3.5 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="min-w-0 space-y-3">
          {t.rationale && <p className="text-sm text-ink-2">{t.rationale}</p>}
          <details>
            <summary className="cursor-pointer text-2xs font-medium uppercase tracking-[0.06em] text-ink-3 hover:text-ink-2">Config</summary>
            <dl className="mt-1.5 grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-x-4 gap-y-1 font-mono text-2xs">
              {Object.entries(t.config).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2 border-b border-line-soft py-0.5">
                  <dt className="truncate text-ink-3">{k}</dt>
                  <dd className="num truncate text-ink" title={evidenceValue(v)}>
                    {evidenceValue(v)}
                  </dd>
                </div>
              ))}
            </dl>
          </details>
          {t.error && <InlineError error={t.error} fallback="Training stopped with an error." />}
          {t.backend === "manual" && t.status !== "proposed" && t.status !== "succeeded" && (
            <div>
              <div className="text-2xs font-medium uppercase tracking-[0.06em] text-ink-3">Manual · no GPU backend</div>
              <ManualTraining pid={pid} tid={t.id} />
            </div>
          )}
          {t.checkpoint_ids.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {t.checkpoint_ids.map((c) => (
                <Badge key={c}>
                  <span className="font-mono">{c}</span>
                </Badge>
              ))}
            </div>
          )}
        </div>

        {/* estimate + approval gate */}
        <div className={cn("rounded p-3", e.paid ? "bg-caution/5 border border-caution/35" : "bg-well border border-line")}>
          <div className="flex items-center justify-between">
            <span className="text-2xs font-medium uppercase tracking-[0.06em] text-ink-3">Estimate</span>
            {e.paid ? <Badge tone="caution">paid</Badge> : <Badge>no charge</Badge>}
          </div>
          <div className="mt-2 flex items-baseline gap-1">
            <span className="num kn-display text-[30px]">{usd(e.est_cost_usd)}</span>
            <span className="text-xs text-ink-3">est.</span>
          </div>
          <div className="mt-1 text-xs text-ink-2">
            <span className="font-mono">{e.gpu}</span> · <span className="num">{e.gpu_hours.toFixed(2)}</span> GPU-h
          </div>
          {e.notes && <p className="mt-1.5 text-2xs text-ink-3">{e.notes}</p>}

          <div className="mt-3 border-t border-line-soft pt-3">
            {t.status === "proposed" && (
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-2xs font-medium text-servo">
                  <ShieldAlert className="h-3.5 w-3.5" /> Needs approval
                </div>
                <Input value={approver} onChange={(ev) => setApprover(ev.target.value)} placeholder="Approver name" aria-label="Approver name" />
                <Button
                  variant="primary"
                  size="sm"
                  className="w-full"
                  disabled={!approver.trim()}
                  title={approver.trim() ? "Approve this training run" : "Type your name first; it is recorded as the approver"}
                  loading={approve.isPending}
                  onClick={() => approve.mutate()}
                >
                  <Check className="h-3.5 w-3.5" /> Approve{e.paid ? ` ${usd(e.est_cost_usd)}` : ""}
                </Button>
              </div>
            )}
            {t.status === "approved" && (
              <div className="space-y-2">
                <p className="text-2xs text-ink-3">
                  Approved by <span className="text-ink">{t.approved_by}</span>
                </p>
                <Button variant="primary" size="sm" className="w-full" loading={launch.isPending} onClick={() => launch.mutate()}>
                  <Rocket className="h-3.5 w-3.5" /> Launch
                </Button>
              </div>
            )}
            {t.status !== "proposed" && t.status !== "approved" && (
              <p className="text-2xs text-ink-3">
                {t.approved_by ? (
                  <>
                    Approved by <span className="text-ink">{t.approved_by}</span>
                  </>
                ) : (
                  "No approver recorded"
                )}
                {t.external_id && (
                  <>
                    {" "}
                    · job <span className="font-mono">{t.external_id}</span>
                  </>
                )}
              </p>
            )}
            {(approve.error || launch.error) && <InlineError error={approve.error ?? launch.error} className="mt-2" />}
          </div>
        </div>
      </div>
      {live && <LogView ids={[t.id, t.external_id]} active={t.status === "queued" || t.status === "running"} />}
    </article>
  );
}

function LogView({ ids, active }: { ids: (string | null)[]; active: boolean }) {
  const lines = useLogs(ids);
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 60) el.scrollTop = el.scrollHeight;
  }, [lines.length]);
  return (
    <div className="border-t border-line-soft">
      <div className="flex items-center gap-1.5 px-3.5 py-1.5 text-2xs text-ink-3">
        <Terminal className="h-3 w-3" /> Log
        {active && <span className="h-1.5 w-1.5 rounded-full bg-servo live-dot" />}
        <span className="ml-auto num font-mono">{lines.length} lines</span>
      </div>
      <pre ref={ref} className="max-h-56 overflow-auto bg-well px-3.5 py-2 font-mono text-[11px] leading-[1.55] text-ink-2 scroll-thin">
        {lines.length ? lines.join("\n") : active ? "Waiting for logs…" : "No logs this session."}
      </pre>
    </div>
  );
}

function Checkpoints({ pid }: { pid: string }) {
  const q = useCheckpoints(pid);
  return (
    <Panel title="Checkpoints" bodyClassName="p-0">
      <QueryState q={q} skeleton={<div className="p-3.5"><SkeletonRows rows={2} /></div>}>
        {(cks) =>
          cks.length === 0 ? (
            <EmptyState title="No checkpoints yet">Saved by training runs.</EmptyState>
          ) : (
            <table className="w-full text-left text-sm">
              <thead className="text-2xs uppercase tracking-[0.05em] text-ink-3">
                <tr className="border-b border-line-soft">
                  <th className="px-3.5 py-2 font-medium">Checkpoint</th>
                  <th className="px-2 py-2 font-medium">Policy</th>
                  <th className="px-2 py-2 text-right font-medium">Step</th>
                  <th className="px-2 py-2 font-medium">Metrics</th>
                  <th className="px-3.5 py-2 text-right font-medium">Saved</th>
                </tr>
              </thead>
              <tbody>
                {[...cks]
                  .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
                  .map((c) => (
                    <tr key={c.id} className="border-b border-line-soft last:border-0">
                      <td className="px-3.5 py-2">
                        <div className="font-mono text-xs">{c.id}</div>
                        <div className="max-w-[260px] truncate font-mono text-2xs text-ink-4" title={c.path}>
                          {c.path}
                        </div>
                      </td>
                      <td className="px-2 py-2 text-xs uppercase">{c.policy_type}</td>
                      <td className="num px-2 py-2 text-right font-mono text-xs">{c.step.toLocaleString()}</td>
                      <td className="px-2 py-2 font-mono text-2xs text-ink-2">
                        {Object.entries(c.metrics)
                          .slice(0, 3)
                          .map(([k, v]) => `${k}=${evidenceValue(v)}`)
                          .join("  ") || "—"}
                      </td>
                      <td className="px-3.5 py-2 text-right text-2xs text-ink-3">{ago(c.created_at)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )
        }
      </QueryState>
    </Panel>
  );
}
