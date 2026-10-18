import { useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { CornerDownRight, Database, Plus } from "lucide-react";
import { api } from "../../api/client";
import { isJobActive, useJob, useTrackJob } from "../../api/events";
import { useDatasets, useEpisodes } from "../../api/hooks";
import type { Augmentation, AugmentationType, DatasetVersion, Episode, Job } from "../../api/types";
import { Badge, Button, Dialog, EmptyState, ErrorState, Field, Input, Meter, Panel, QueryState, Select, SkeletonRows, Toggle, InlineError } from "../../components/ui";
import { ago, cn } from "../../lib/format";
import { AUG_NOTE, augWords, datasetBreakdown } from "../../lib/augment";

function stats(d: DatasetVersion) {
  const sources = new Set(d.links.map((l) => l.episode_id));
  const aug = d.links.filter((l) => l.augmentation).length;
  const train = d.links.filter((l) => l.split === "train").length;
  const evalN = d.links.length - train;
  return { sources: sources.size, original: d.links.length - aug, aug, train, eval: evalN };
}

/** Order versions depth-first by parent so lineage reads top-down. */
function lineage(ds: DatasetVersion[]): { d: DatasetVersion; depth: number }[] {
  const byParent = new Map<string | null, DatasetVersion[]>();
  const ids = new Set(ds.map((d) => d.id));
  for (const d of ds) {
    const p = d.parent_id && ids.has(d.parent_id) ? d.parent_id : null;
    byParent.set(p, [...(byParent.get(p) ?? []), d]);
  }
  const out: { d: DatasetVersion; depth: number }[] = [];
  const walk = (p: string | null, depth: number) => {
    for (const d of (byParent.get(p) ?? []).sort((a, b) => (a.created_at < b.created_at ? -1 : 1))) {
      out.push({ d, depth });
      walk(d.id, depth + 1);
    }
  };
  walk(null, 0);
  return out;
}

export function DatasetsTab() {
  const { pid = "" } = useParams();
  const q = useDatasets(pid);
  const episodes = useEpisodes(pid);
  const epById = new Map((episodes.data ?? []).map((e) => [e.id, e]));
  const [open, setOpen] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  const track = useTrackJob();

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <Button variant="primary" size="sm" onClick={() => setOpen(true)}>
          <Plus className="h-3.5 w-3.5" /> New version
        </Button>
      </div>
      {job && (isJobActive(job) || job.status === "failed") && (
        <div className="flex items-center gap-3 rounded bg-plate px-3 py-2 text-xs border border-line">
          <span className="text-ink-2">Building dataset…</span>
          <Meter value={job.progress} className="max-w-[200px]" />
          <span className="truncate text-ink-3">{job.status === "failed" ? <InlineError error={job.error ?? job.message} detail={job.error_detail} /> : job.message}</span>
        </div>
      )}
      <QueryState q={q} skeleton={<SkeletonRows rows={3} />}>
        {(ds) =>
          ds.length === 0 ? (
            <Panel>
              <EmptyState icon={<Database className="h-5 w-5" />} title="No dataset versions yet" action={<Button size="sm" onClick={() => setOpen(true)}>Create the first version</Button>}>
                A version snapshots your accepted episodes (optionally with augmentations) for training.
              </EmptyState>
            </Panel>
          ) : (
            <div className="space-y-2">
              {lineage(ds).map(({ d, depth }) => (
                <VersionRow key={d.id} d={d} depth={depth} parentName={ds.find((x) => x.id === d.parent_id)?.name} eps={epById} />
              ))}
            </div>
          )
        }
      </QueryState>
      <CreateVersionDialog
        open={open}
        onClose={() => setOpen(false)}
        pid={pid}
        versions={q.data ?? []}
        onStarted={(j) => {
          track(j);
          setJobId(j.id);
          setOpen(false);
        }}
      />
    </div>
  );
}

function VersionRow({ d, depth, parentName, eps }: { d: DatasetVersion; depth: number; parentName?: string; eps: Map<string, Episode> }) {
  const s = stats(d);
  const breakdown = datasetBreakdown(d, eps);
  const total = Math.max(1, s.original + s.aug);
  return (
    <div className="flex gap-2" style={{ paddingLeft: Math.min(depth, 4) * 24 }}>
      {depth > 0 && <CornerDownRight className="mt-3 h-4 w-4 shrink-0 text-ink-4" />}
      <div className="min-w-0 flex-1 rounded-md bg-plate p-3 border border-line">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-sm font-medium">{d.name}</span>
          <span className="font-mono text-2xs text-ink-3">{d.id}</span>
          {parentName && <span className="text-2xs text-ink-3">from {parentName}</span>}
          <span className="ml-auto text-2xs text-ink-4">{ago(d.created_at)}</span>
        </div>
        <div className="mt-2.5 grid grid-cols-2 gap-4 lg:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <div className="mb-1 flex justify-between text-2xs text-ink-3">
              <span className="text-ink-2">{breakdown.text}</span>
            </div>
            <div className="flex h-1.5 overflow-hidden rounded-full bg-rail">
              <div className="bg-servo" style={{ width: `${(s.original / total) * 100}%` }} />
              <div className="bg-servo/35" style={{ width: `${(s.aug / total) * 100}%` }} />
            </div>
          </div>
          <div className="text-2xs text-ink-3">
            held out {Math.round(d.eval_fraction * 100)}% of source demos · seed <span className="font-mono">{d.split_seed}</span>
          </div>
          <div className="text-2xs text-ink-3">
            <span className="num font-mono text-ink">{d.num_frames.toLocaleString()}</span> frames
            {d.lerobot_root && (
              <div className="mt-0.5 truncate font-mono" title={d.lerobot_root}>
                {d.lerobot_root}
              </div>
            )}
          </div>
        </div>
        {d.augmentations.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-1">
            {d.augmentations.map((a, i) => (
              <Badge key={i}>
                <span className="font-mono">{augWords(a)}</span>
              </Badge>
            ))}
          </div>
        )}
        {breakdown.hasAug && <p className="mt-1.5 text-2xs text-ink-3">{AUG_NOTE}</p>}
      </div>
    </div>
  );
}

const AUG_OPTIONS: { key: string; type: AugmentationType; param: string; value: number; label: string; hint: string }[] = [
  { key: "dim", type: "brightness", param: "factor", value: 0.55, label: "Brightness — darker", hint: "factor < 1 darkens" },
  { key: "bright", type: "brightness", param: "factor", value: 1.35, label: "Brightness — brighter", hint: "factor > 1 brightens" },
  { key: "contrast", type: "contrast", param: "factor", value: 1.3, label: "Contrast", hint: "factor around mean" },
  { key: "temp", type: "color_temperature", param: "shift", value: 0.6, label: "Color temperature", hint: "−1 cool … +1 warm" },
  { key: "shadow", type: "shadow", param: "strength", value: 0.45, label: "Cast shadow", hint: "0 none … 1 black" },
];

function CreateVersionDialog({
  open,
  onClose,
  pid,
  versions,
  onStarted,
}: {
  open: boolean;
  onClose: () => void;
  pid: string;
  versions: DatasetVersion[];
  onStarted: (j: Job) => void;
}) {
  const episodes = useEpisodes(pid);
  const accepted = episodes.data?.filter((e) => e.status === "accepted").length;
  const [name, setName] = useState("");
  const [parent, setParent] = useState("");
  const [evalFrac, setEvalFrac] = useState(0.2);
  const [seed, setSeed] = useState(0);
  const [on, setOn] = useState<Record<string, boolean>>({});
  const [vals, setVals] = useState<Record<string, number>>(Object.fromEntries(AUG_OPTIONS.map((o) => [o.key, o.value])));

  const augmentations: Augmentation[] = AUG_OPTIONS.filter((o) => on[o.key]).map((o) => ({
    type: o.type,
    params: { [o.param]: vals[o.key] },
  }));
  const m = useMutation({
    mutationFn: () =>
      api.createDataset(pid, {
        name: name.trim() || null,
        parent_id: parent || null,
        augmentations,
        eval_fraction: evalFrac,
        split_seed: seed,
      }),
    onSuccess: onStarted,
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="New dataset version"
      width={560}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" loading={m.isPending} onClick={() => m.mutate()}>
            Create version
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Name" hint="Optional">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="auto" />
          </Field>
          <Field label="Parent version">
            <Select value={parent} onChange={(e) => setParent(e.target.value)}>
              <option value="">None (new root)</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        <p className="text-2xs text-ink-3">
          Episodes: all accepted episodes{accepted !== undefined ? ` (${accepted} right now)` : ""}.
        </p>
        <div>
          <div className="mb-1 text-xs font-medium text-ink-2">Augmentations</div>
          <p className="mb-2 text-2xs text-ink-3">Each enabled transform adds one augmented copy of every training episode.</p>
          <div className="divide-y divide-[var(--line-soft)] rounded bg-well px-3 border border-line">
            {AUG_OPTIONS.map((o) => (
              <div key={o.key} className="flex items-center gap-3 py-1.5">
                <div className="flex-1">
                  <Toggle checked={!!on[o.key]} onChange={(v) => setOn((s) => ({ ...s, [o.key]: v }))} label={o.label} hint={o.hint} />
                </div>
                <label className={cn("flex items-center gap-1.5 font-mono text-2xs text-ink-3", !on[o.key] && "opacity-40")}>
                  {o.param}
                  <Input
                    type="number"
                    step={0.05}
                    disabled={!on[o.key]}
                    value={vals[o.key]}
                    onChange={(e) => setVals((s) => ({ ...s, [o.key]: Number(e.target.value) }))}
                    className="num h-7 w-20 font-mono"
                  />
                </label>
              </div>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Eval fraction" hint="Held out by source episode">
            <Input type="number" min={0} max={0.9} step={0.05} value={evalFrac} onChange={(e) => setEvalFrac(Number(e.target.value))} className="num font-mono" />
          </Field>
          <Field label="Split seed">
            <Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} className="num font-mono" />
          </Field>
        </div>
        {m.error && <ErrorState error={m.error} compact />}
      </div>
    </Dialog>
  );
}
