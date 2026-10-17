import { useHealth } from "../api/hooks";
import type { Health, ProviderInfo } from "../api/types";
import { cn } from "../lib/format";

const SUBSYSTEMS = ["db", "storage", "vlm", "agent", "executor", "training"] as const satisfies readonly (keyof Health)[];

export function describeProvider(p: ProviderInfo | string | null | undefined): { name: string; fallback: boolean; model?: string } {
  if (!p) return { name: "unknown", fallback: true };
  if (typeof p === "string") return { name: p, fallback: /fallback|local|disabled/i.test(p) };
  return { name: p.provider, fallback: !!p.fallback, model: p.model ?? undefined };
}

export function HealthPills() {
  const q = useHealth();

  if (q.isLoading)
    return <div className="flex gap-1">{SUBSYSTEMS.map((s) => <span key={s} className="skeleton h-5 w-16 rounded-sm" />)}</div>;

  if (q.error || !q.data)
    return (
      <span
        className="inline-flex h-5 items-center gap-1.5 rounded-sm bg-fault/10 px-2 text-2xs font-medium text-fault shadow-[0_0_0_1px_rgb(var(--fault)/.3)]"
        title="GET /health failed"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-fault" />
        API offline
      </span>
    );

  const h = q.data;
  const notes = h.notes?.length ? `\n\n${h.notes.join("\n")}` : "";
  return (
    <div className="flex items-center gap-1" aria-label="Provider status">
      {SUBSYSTEMS.map((s) => {
        const d = describeProvider(h[s]);
        return (
          <span
            key={s}
            title={`${s}: ${d.name}${d.model ? ` · ${d.model}` : ""}${d.fallback ? " — local fallback" : ""}${notes}`}
            className={cn(
              "inline-flex h-5 items-center gap-1 rounded-sm px-1.5 text-2xs",
              d.fallback ? "prov-fallback text-ink-2" : "bg-rail text-ink-2 shadow-[0_0_0_1px_var(--line)]",
            )}
          >
            <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", d.fallback ? "bg-caution" : "bg-pass")} />
            <span className="text-ink-3">{s}</span>
            <span className="hidden max-w-[96px] truncate font-mono text-[10.5px] min-[1500px]:inline">{d.name}</span>
            {d.fallback && <span className="text-[10px] font-medium uppercase tracking-wide text-caution">fallback</span>}
          </span>
        );
      })}
    </div>
  );
}
