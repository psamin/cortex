import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import type { Infra, InfraChoice, InfraOption } from "../api/types";
import { Button, ErrorState, Field, InlineError, Input, Panel, Select, Skeleton } from "../components/ui";
import { cn } from "../lib/format";

/** Per-project infrastructure: who plans, where training runs, which GPU and how long. */
export function SetupPage() {
  const { pid = "" } = useParams();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["infra", pid], queryFn: () => api.infra(pid) });
  const [draft, setDraft] = useState<InfraChoice | null>(null);
  useEffect(() => {
    if (q.data) setDraft(q.data.choice);
  }, [q.data]);
  const save = useMutation({
    mutationFn: () => api.saveInfra(pid, draft!),
    onSuccess: (d: Infra) => qc.setQueryData(["infra", pid], d),
  });
  const set = <K extends keyof InfraChoice>(k: K, v: InfraChoice[K]) => {
    setDraft((d) => (d ? { ...d, [k]: v } : d));
    save.reset();
  };
  const d = q.data;
  const dirty = !!d && !!draft && JSON.stringify(draft) !== JSON.stringify(d.choice);

  return (
    <div className="max-w-[880px] space-y-4 px-5 py-5">
      <div>
        <h1 className="kn-display text-[36px] leading-tight">Setup</h1>
        <p className="mt-0.5 text-sm text-ink-3">Where this project runs.</p>
      </div>
      {q.isLoading || (d && !draft) ? (
        <Skeleton className="h-64 w-full" />
      ) : q.error || !d || !draft ? (
        <ErrorState error={q.error} onRetry={() => q.refetch()} />
      ) : (
        <>
          <Panel title="Planner" meta="drives agent demos">
            <Choices options={d.planner} value={draft.planner} onChange={(v) => set("planner", v)} name="planner" />
          </Panel>
          <Panel title="Training" meta="lerobot-train">
            <Choices options={d.training_backend} value={draft.training_backend} onChange={(v) => set("training_backend", v)} name="training" />
            <div className="mt-3 grid grid-cols-[1fr_180px] gap-3">
              <Field label="GPU">
                <Select value={draft.training_gres} onChange={(e) => set("training_gres", e.target.value)}>
                  {d.gpus.map((g) => (
                    <option key={g.id} value={g.id} disabled={!g.available}>
                      {g.label}
                      {!g.available && g.reason ? ` (${g.reason})` : ""}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Walltime (min)" hint={d.walltime_min.note}>
                <Input
                  type="number"
                  min={d.walltime_min.min}
                  max={d.walltime_min.max}
                  value={draft.training_walltime_min}
                  onChange={(e) => set("training_walltime_min", Math.min(d.walltime_min.max, Math.max(d.walltime_min.min, Number(e.target.value) || d.walltime_min.min)))}
                  className="num font-mono"
                />
              </Field>
            </div>
          </Panel>
          <div className="flex items-center gap-3">
            <Button variant="primary" disabled={!dirty} title={dirty ? undefined : "Nothing changed"} loading={save.isPending} onClick={() => save.mutate()}>
              Save
            </Button>
            {save.isSuccess && !dirty && <span className="text-sm text-pass">Saved</span>}
            {save.error && <InlineError error={save.error} />}
          </div>
          <Panel title="Evaluation" meta="read-only">
            <Choices options={d.evaluation} readOnly />
          </Panel>
          <Panel title="Robot" meta="read-only">
            <Choices options={d.robot} readOnly />
            <Link to={`/p/${pid}/robot`} className="mt-2 inline-block text-sm text-servo hover:underline">
              Robot page
            </Link>
          </Panel>
        </>
      )}
    </div>
  );
}

/** Radio list; unavailable options are disabled and show the backend's reason as text. */
function Choices({ options, value, onChange, name, readOnly }: { options: InfraOption[]; value?: string; onChange?: (v: string) => void; name?: string; readOnly?: boolean }) {
  return (
    <ul className="space-y-1.5">
      {options.map((o) => (
        <li key={o.id}>
          <label className={cn("flex items-start gap-2.5 rounded px-2 py-1.5", !readOnly && o.available && "cursor-pointer hover:bg-rail/60")}>
            {readOnly ? (
              <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", o.available ? "bg-pass" : "bg-ink-4")} />
            ) : (
              <input type="radio" name={name} checked={value === o.id} disabled={!o.available} onChange={() => onChange?.(o.id)} className="mt-1 accent-[rgb(var(--servo))]" />
            )}
            <span className="min-w-0">
              <span className={cn("block text-sm", o.available ? "text-ink" : "text-ink-3")}>{o.label}</span>
              {!o.available && o.reason && <span className="block text-xs text-ink-3">{o.reason}</span>}
            </span>
          </label>
        </li>
      ))}
    </ul>
  );
}
