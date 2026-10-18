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

  if (q.isLoading) return <span className="label hidden lg:inline">Checking systems…</span>;

  if (q.error || !q.data)
    return (
      <span className="label inline-flex items-center gap-1.5 !text-fault" title="GET /health failed">
        <span className="h-1.5 w-1.5 rounded-full bg-fault" />
        API offline
      </span>
    );

  const h = q.data;
  const notes = h.notes?.length ? "\n\n" + h.notes.join("\n") : "";
  return (
    <div className="hidden items-center gap-3.5 lg:flex" aria-label="Provider status">
      {SUBSYSTEMS.map((s) => {
        const d = describeProvider(h[s]);
        return (
          <span
            key={s}
            title={`${s}: ${d.name}${d.model ? ` · ${d.model}` : ""}${d.fallback ? " (local fallback)" : ""}${notes}`}
            className="label inline-flex items-center gap-1.5 !tracking-[0.14em]"
          >
            <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", d.fallback ? "border border-caution" : "bg-servo")} />
            {s}
          </span>
        );
      })}
    </div>
  );
}
