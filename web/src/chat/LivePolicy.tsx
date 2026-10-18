import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, CircleDashed, Cpu, Square } from "lucide-react";
import { ApiError, api } from "../api/client";
import { cn } from "../lib/format";

/** Trained policies running on a Futurama GPU over held-out episodes they never trained on (#157).
 *  Opening the tab requests a GPU when none is allocated, and narrates each Slurm step until it's live. */
const SIM_TITLE: Record<string, string> = { jenga: "Jenga", unscrew: "Screwdriver", suture: "Suture" };
const SIM_MATCH: Record<string, RegExp> = { jenga: /jenga/i, unscrew: /screw|bolt|factory/i, suture: /sutur/i };
type Pick = { kind: "sim"; task: string; seed: number } | { kind: "replay"; task: string; ep: number };

/** `hint`: the project's goal, to open on its own task (e.g. "Jenga (sim)" opens the Jenga policy). */
export function GpuStage({ hint = "" }: { hint?: string }) {
  const qc = useQueryClient();
  const st = useQuery({
    queryKey: ["live-policy"],
    queryFn: api.livePolicy,
    refetchInterval: (q) => (q.state.data?.available ? 30_000 : 3_000),
  });
  const refresh = () => qc.invalidateQueries({ queryKey: ["live-policy"] });
  const start = useMutation({ mutationFn: api.startLivePolicy, onSettled: refresh });
  const stop = useMutation({ mutationFn: api.stopLivePolicy, onSettled: refresh });
  const [pick, setPick] = useState<Pick | null>(null);
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

  const sims = Object.entries(s.sim ?? {});
  const run = (task: string) => setPick({ kind: "sim", task, seed: 100_000 + Math.floor(Math.random() * 900_000) });
  const own = sims.find(([t]) => SIM_MATCH[t]?.test(hint))?.[0];
  const cur: Pick | null =
    pick ?? (own ? { kind: "sim", task: own, seed: 100_000 } : sims[0] ? { kind: "sim", task: sims[0][0], seed: 100_000 }
      : policies[0] ? { kind: "replay", task: policies[0][0], ep: policies[0][1].held_out[0] } : null);
  const curSim = cur?.kind === "sim" ? s.sim?.[cur.task] : undefined;
  return (
    <div className="relative flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 px-4 pt-3 text-xs">
        <span className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-3">{s.gpu}</span>
        {sims.map(([task, p]) => (
          <button key={task} type="button" onClick={() => run(task)}
            title={`${p.variant.toUpperCase()} step ${p.step}: ${Math.round(p.success_rate * 100)}% on ${p.eval_episodes} unseen layouts`}
            className={cn("h-6 rounded-full border px-2.5", cur?.kind === "sim" && cur.task === task ? "border-ink bg-ink text-bench" : "border-line-strong text-ink-2 hover:text-ink")}>
            {SIM_TITLE[task] ?? task} · sim
          </button>
        ))}
        {policies.map(([task, p]) =>
          p.held_out.map((ep) => (
            <button key={`${task}-${ep}`} type="button" onClick={() => setPick({ kind: "replay", task, ep })}
              title={`${p.checkpoint} on held-out real episode ${ep}`}
              className={cn("h-6 rounded-full border px-2.5", cur?.kind === "replay" && cur.task === task && cur.ep === ep ? "border-ink bg-ink text-bench" : "border-line-strong text-ink-3 hover:text-ink")}>
              {task} · real ep {ep}
            </button>
          )),
        )}
        <button type="button" onClick={() => stop.mutate()} disabled={stop.isPending}
          className="ml-auto inline-flex h-6 items-center gap-1 rounded-full border border-line-strong px-2.5 text-ink-3 hover:text-ink">
          <Square className="h-3 w-3" /> Stop GPU
        </button>
      </div>
      {curSim && cur?.kind === "sim" && (
        <div className="flex items-center gap-3 px-4 pt-2 text-xs text-ink-2">
          <span>{curSim.instruction}</span>
          <span className="text-ink-3">{curSim.variant.toUpperCase()} · {Math.round(curSim.success_rate * 100)}% success on {curSim.eval_episodes} unseen layouts</span>
          <button type="button" onClick={() => run(cur.task)} className="ml-auto h-7 rounded-full bg-ink px-3 font-medium text-bench">
            Run on a new layout
          </button>
        </div>
      )}
      <div className="flex min-h-0 flex-1 items-center justify-center p-4">
        {cur && (
          <img key={cur.kind === "sim" ? `${cur.task}-${cur.seed}` : `${cur.task}-${cur.ep}`}
            src={cur.kind === "sim" ? api.livePolicySimUrl(cur.task, cur.seed) : api.livePolicyStreamUrl(cur.task, cur.ep)}
            alt={`${cur.task} policy running live on the GPU`} className="block max-h-full w-auto max-w-full border border-line-strong object-contain" />
        )}
      </div>
    </div>
  );
}
