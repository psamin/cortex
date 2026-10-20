import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Undo2, X } from "lucide-react";
import { api } from "../../api/client";
import { useCheckpoints, useRealTrials, useSimReal } from "../../api/hooks";
import { CUBE_REGIONS, LIGHTINGS, type CubeRegion, type Lighting, type SimRealRow } from "../../api/types";
import { useApprover } from "../../components/refs";
import { Badge, Button, Input, InlineError, Panel, Select } from "../../components/ui";
import { cn } from "../../lib/format";
import { agreementWord, gapText, rateText } from "../../lib/simReal";

/**
 * Real arm (docs/STACK_SPEC.md 2.2): log each rollout of a checkpoint on the physical arm, as the person watching
 * judged it, and see the real rate next to the sim rate for the same checkpoint.
 */
export function RealTrialsPanel({ pid }: { pid: string }) {
  const cks = useCheckpoints(pid);
  const report = useSimReal(pid);
  const trials = useRealTrials(pid);
  const qc = useQueryClient();
  const [judge, setJudge] = useApprover();
  const [ck, setCk] = useState("");
  const [region, setRegion] = useState<CubeRegion>("center");
  const [light, setLight] = useState<Lighting>("normal");
  const [why, setWhy] = useState("");
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["real-trials", pid] });
    qc.invalidateQueries({ queryKey: ["sim-real", pid] });
  };
  const checkpoint = ck || cks.data?.[cks.data.length - 1]?.id || "";
  const log = useMutation({
    mutationFn: (success: boolean) =>
      api.logRealTrials(pid, judge.trim(), [
        { checkpoint_id: checkpoint, success, condition: { cube_region: region, lighting: light }, failure_mode: success ? null : why.trim() || null },
      ]),
    onSuccess: () => {
      setWhy("");
      refresh();
    },
  });
  const undo = useMutation({ mutationFn: (tid: string) => api.voidRealTrial(tid), onSuccess: refresh });

  if (!cks.data?.length) return null; // nothing trained yet: nothing to run on the arm
  const r = report.data;
  const recent = [...(trials.data ?? [])].reverse().slice(0, 3);
  const ready = !!checkpoint && !!judge.trim();

  return (
    <Panel title="Real arm" meta={r ? <span className="text-2xs text-ink-3">{metaLine(trials.data?.length ?? 0, r.checkpoints.length)}</span> : undefined}>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0">
          {r && r.checkpoints.length > 0 ? (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-2xs uppercase tracking-[0.06em] text-ink-3">
                  <th className="pb-1.5 font-medium">Checkpoint</th>
                  <th className="pb-1.5 font-medium">Real</th>
                  <th className="pb-1.5 font-medium">Sim</th>
                  <th className="pb-1.5 font-medium">Gap</th>
                </tr>
              </thead>
              <tbody>
                {r.checkpoints.map((row) => (
                  <CheckpointRow key={row.checkpoint_id} row={row} />
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-ink-3">Run a checkpoint on the arm and log each rollout here. Sim results become trustworthy once they match real ones.</p>
          )}
          {r && r.checkpoints.length > 0 && (
            <p className="mt-2 text-2xs text-ink-3">
              Sim ranks checkpoints like the real arm:{" "}
              <span className="text-ink-2">
                {r.spearman != null ? agreementWord(r.spearman) : `not yet (${r.checkpoints_compared}/${MIN_COMPARED} checkpoints with both)`}
              </span>
            </p>
          )}
        </div>

        <div className="space-y-2 rounded border border-line bg-well p-3">
          <div className="grid grid-cols-2 gap-2">
            <Select aria-label="Checkpoint" value={checkpoint} onChange={(e) => setCk(e.target.value)} className="col-span-2">
              {[...cks.data].reverse().map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} · step {c.step.toLocaleString()}
                </option>
              ))}
            </Select>
            <Select aria-label="Object position" value={region} onChange={(e) => setRegion(e.target.value as CubeRegion)}>
              {CUBE_REGIONS.map((x) => (
                <option key={x}>{x}</option>
              ))}
            </Select>
            <Select aria-label="Lighting" value={light} onChange={(e) => setLight(e.target.value as Lighting)}>
              {LIGHTINGS.map((x) => (
                <option key={x}>{x}</option>
              ))}
            </Select>
            <Input aria-label="Judged by" placeholder="Your name" value={judge} onChange={(e) => setJudge(e.target.value)} />
            <Input aria-label="If it failed, how" placeholder="If failed: how" value={why} onChange={(e) => setWhy(e.target.value)} />
          </div>
          <div className="flex gap-2">
            <Button size="sm" className="flex-1" disabled={!ready} loading={log.isPending && log.variables} onClick={() => log.mutate(true)} title={ready ? "This rollout succeeded" : "Type your name first"}>
              <Check className="h-3.5 w-3.5 text-pass" /> Success
            </Button>
            <Button size="sm" className="flex-1" disabled={!ready} loading={log.isPending && !log.variables} onClick={() => log.mutate(false)} title={ready ? "This rollout failed" : "Type your name first"}>
              <X className="h-3.5 w-3.5 text-fault" /> Fail
            </Button>
          </div>
          {(log.error || undo.error) && <InlineError error={log.error ?? undo.error} />}
          {recent.length > 0 && (
            <ul className="space-y-1 border-t border-line-soft pt-2">
              {recent.map((t) => (
                <li key={t.id} className="flex items-center gap-2 text-2xs text-ink-3">
                  <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", t.success ? "bg-pass" : "bg-fault")} />
                  <span className="truncate">
                    <span className="font-mono">{t.checkpoint_id}</span> · {t.condition.cube_region}/{t.condition.lighting}
                    {t.failure_mode ? ` · ${t.failure_mode}` : ""}
                  </span>
                  {t.source === "agent" && <Badge>agent</Badge>}
                  <button type="button" aria-label="Undo" title="Undo this trial" className="ml-auto text-ink-4 hover:text-ink" onClick={() => undo.mutate(t.id)}>
                    <Undo2 className="h-3 w-3" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Panel>
  );
}

const MIN_COMPARED = 3; // backend real_trials.MIN_CHECKPOINTS

const metaLine = (n: number, cks: number) => (n ? `${n} trial${n === 1 ? "" : "s"} · ${cks} checkpoint${cks === 1 ? "" : "s"}` : "");

function CheckpointRow({ row }: { row: SimRealRow }) {
  return (
    <tr className="border-t border-line-soft">
      <td className="py-1.5 font-mono text-2xs">{row.checkpoint_id}</td>
      <td className="py-1.5 text-ink">{rateText(row.real)}</td>
      <td className="py-1.5 text-ink-2">{row.sim ? rateText(row.sim) : <span className="text-ink-4">not evaluated</span>}</td>
      <td className={cn("py-1.5", row.gap != null && row.gap < -0.15 ? "text-caution" : "text-ink-2")}>{gapText(row.gap)}</td>
    </tr>
  );
}
