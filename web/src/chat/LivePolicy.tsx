import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, CircleDashed, Cpu, Square } from "lucide-react";
import { ApiError, api } from "../api/client";
import { cn } from "../lib/format";

/** Trained policies running on a Futurama GPU over held-out episodes they never trained on (#157).
 *  Opening the tab requests a GPU when none is allocated, and narrates each Slurm step until it's live. */
export function GpuStage() {
  const qc = useQueryClient();
  const st = useQuery({
    queryKey: ["live-policy"],
    queryFn: api.livePolicy,
    refetchInterval: (q) => (q.state.data?.available ? 30_000 : 3_000),
  });
  const refresh = () => qc.invalidateQueries({ queryKey: ["live-policy"] });
  const start = useMutation({ mutationFn: api.startLivePolicy, onSettled: refresh });
  const stop = useMutation({ mutationFn: api.stopLivePolicy, onSettled: refresh });
  const [pick, setPick] = useState<{ task: string; ep: number } | null>(null);
  const s = st.data;
  const policies = Object.entries(s?.policies ?? {});
  const err = (start.error ?? stop.error) as ApiError | null;

  // opening the tab is the request: allocate a GPU unless one is up, coming up, or just failed
  const asked = useRef(false);
  useEffect(() => {
    if (s && !s.available && s.phase === "stopped" && !s.error && !asked.current) {
      asked.current = true;
      start.mutate();
    }
  }, [s, start]);

  if (!s?.available) {
    const order = Object.keys(s?.steps ?? {});
    const at = s?.step ? order.indexOf(s.step) : start.isPending ? 0 : -1;
    const busy = s?.phase === "starting" || start.isPending;
    return (
      <div className="relative flex h-full flex-col items-center justify-center gap-4 px-8">
        {busy ? <CircleDashed className="h-6 w-6 animate-spin text-ink-3" /> : <Cpu className="h-6 w-6 text-ink-3" />}
        <p className="max-w-md text-center text-xs leading-5 text-ink-2">
          {start.isPending && !s?.step ? "Requesting a GPU on Georgia Tech's Futurama cluster…" : s?.reason ?? "Checking the GPU server…"}
        </p>
        {at >= 0 && (
          <ol className="space-y-1 text-xs">
            {order.slice(0, -1).map((k, i) => (
              <li key={k} className={cn("flex items-center gap-2", i < at ? "text-ink-3" : i === at ? "text-ink" : "text-ink-4")}>
                {i < at ? <Check className="h-3 w-3" /> : i === at ? <CircleDashed className="h-3 w-3 animate-spin" /> : <span className="h-3 w-3" />}
                {s?.steps?.[k]}
              </li>
            ))}
          </ol>
        )}
        {!busy && (
          <button type="button" onClick={() => start.mutate()}
            className="h-8 rounded-full bg-ink px-4 text-xs font-medium text-bench">
            {s?.error ? "Try again" : "Request a GPU"}
          </button>
        )}
        {err && <p className="max-w-sm text-center text-xs text-fault">{err.message}</p>}
      </div>
    );
  }

  const cur = pick ?? (policies[0] ? { task: policies[0][0], ep: policies[0][1].held_out[0] } : null);
  return (
    <div className="relative flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 px-4 pt-3 text-xs">
        <span className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-3">{s.gpu}</span>
        {policies.map(([task, p]) =>
          p.held_out.map((ep) => (
            <button key={`${task}-${ep}`} type="button" onClick={() => setPick({ task, ep })}
              title={`${p.checkpoint} on held-out episode ${ep}`}
              className={cn("h-6 rounded-full border px-2.5", cur?.task === task && cur.ep === ep ? "border-ink bg-ink text-bench" : "border-line-strong text-ink-2 hover:text-ink")}>
              {task} · ep {ep}
            </button>
          )),
        )}
        <button type="button" onClick={() => stop.mutate()} disabled={stop.isPending}
          className="ml-auto inline-flex h-6 items-center gap-1 rounded-full border border-line-strong px-2.5 text-ink-3 hover:text-ink">
          <Square className="h-3 w-3" /> Stop GPU
        </button>
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center p-4">
        {cur && (
          <img key={`${cur.task}-${cur.ep}`} src={api.livePolicyStreamUrl(cur.task, cur.ep)}
            alt={`${cur.task} policy running live on the GPU`} className="block max-h-full w-auto max-w-full border border-line-strong object-contain" />
        )}
      </div>
    </div>
  );
}
