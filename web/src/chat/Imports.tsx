import { useState } from "react";
import { FileArchive, Upload, X } from "lucide-react";
import { api, uploadDemos } from "../api/client";
import { useTrackJob } from "../api/events";
import type { Job } from "../api/types";
import { InlineError, splitError } from "../components/ui";
import { IsoIcon } from "../sketch";
import { ResultCard } from "./cards";
import { useChat } from "./ChatProvider";
import type { ChatItem } from "./model";
import { GlowEdge } from "../components/ui/spotlight-card";

const MB = 1024 * 1024;
const fmtSize = (b: number) => (b >= 1024 * MB ? `${(b / (1024 * MB)).toFixed(2)} GB` : `${(b / MB).toFixed(1)} MB`);
const clampEpisodes = (n: number) => Math.min(50, Math.max(1, Math.round(n) || 1));

/** Imports are idempotent: re-importing the same dataset returns the earlier, already finished job. */
const isEarlierJob = (job: Job, startedAt: number) =>
  ["succeeded", "failed", "cancelled"].includes(job.status) && !!job.created_at && Date.parse(job.created_at) < startedAt - 2000;

/** Start imports from the studio: each becomes one chat row (upload progress → live job card). */
export function useImporter(pid: string) {
  const chat = useChat();
  const track = useTrackJob();
  const upload = (file: File, maxEpisodes: number) => {
    const n = clampEpisodes(maxEpisodes);
    const id = chat.addImport({
      title: `Importing ${file.name}`,
      detail: `${fmtSize(file.size)} · up to ${n} episode${n === 1 ? "" : "s"}`,
      status: "uploading",
      loaded: 0,
      total: file.size,
    });
    const startedAt = Date.now();
    uploadDemos(pid, file, n, (loaded, total) => chat.patchImport(id, { loaded, total }))
      .promise.then((job) => {
        track(job);
        const again = isEarlierJob(job, startedAt);
        chat.patchImport(id, {
          status: "started",
          jobId: job.id,
          loaded: file.size,
          ...(again ? { title: `Already imported: ${file.name}`, detail: "Imported earlier with the same settings; nothing added." } : {}),
        });
      })
      .catch((e) => {
        const err = splitError(e, undefined, "The import could not start.");
        chat.patchImport(id, { status: "error", error: err.text, errorDetail: err.detail });
      });
  };
  const fromHub = (repoId: string, maxEpisodes: number) => {
    const n = clampEpisodes(maxEpisodes);
    const id = chat.addImport({ title: `Importing ${repoId} from Hugging Face`, detail: `up to ${n} episode${n === 1 ? "" : "s"}`, status: "uploading", loaded: 0, total: 0 });
    const startedAt = Date.now();
    api
      .importHub(pid, repoId, n)
      .then((job) => {
        track(job);
        const again = isEarlierJob(job, startedAt);
        chat.patchImport(id, {
          status: "started",
          jobId: job.id,
          ...(again ? { title: `Already imported: ${repoId}`, detail: "Imported earlier with the same settings; nothing added." } : {}),
        });
      })
      .catch((e) => {
        const err = splitError(e, undefined, "The import could not start.");
        chat.patchImport(id, { status: "error", error: err.text, errorDetail: err.detail });
      });
  };
  return { upload, fromHub };
}

export function ImportRow({ it, pid }: { it: Extract<ChatItem, { kind: "import" }>; pid: string }) {
  const pctDone = it.total ? Math.min(1, it.loaded / it.total) : 0;
  return (
    <div className="enter space-y-2.5">
      <div className="sk-depth relative flex items-start gap-2.5 rounded border border-line-strong bg-plate px-3 py-2.5">
        <GlowEdge />
        <IsoIcon name="dataset" size={20} accent={it.status === "uploading"} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm text-ink">{it.title}</p>
          <p className="text-xs text-ink-3">{it.detail}</p>
          {it.status === "uploading" && it.total > 0 && (
            <div className="mt-1.5 flex items-center gap-2">
              <div className="h-1 flex-1 overflow-hidden bg-[var(--line)]">
                <div className="h-full bg-servo transition-[width] duration-200" style={{ width: `${pctDone * 100}%` }} />
              </div>
              <span className="num w-28 text-right font-mono text-[11px] text-ink-3">
                {fmtSize(it.loaded)} · {Math.round(pctDone * 100)}%
              </span>
            </div>
          )}
          {it.status === "uploading" && it.total === 0 && <p className="mt-1 text-xs text-ink-3">Starting…</p>}
          {it.status === "error" && <InlineError error={it.error} detail={it.errorDetail} className="mt-1.5 rounded-sm border border-fault/30 px-2.5 py-1.5" />}
        </div>
      </div>
      {it.status === "started" && it.jobId && <ResultCard pid={pid} refId={it.jobId} />}
    </div>
  );
}

/** Inline sheet above the composer: confirm a dropped/attached archive, or pick a Hugging Face repo. */
export function ImportSheet({
  mode,
  file,
  onClose,
  onUpload,
  onHub,
}: {
  mode: "file" | "hub";
  file: File | null;
  onClose: () => void;
  onUpload: (file: File, maxEpisodes: number) => void;
  onHub: (repoId: string, maxEpisodes: number) => void;
}) {
  const [episodes, setEpisodes] = useState(5);
  const [repo, setRepo] = useState("");
  const submit = () => {
    if (mode === "file" && file) onUpload(file, episodes);
    else if (mode === "hub") onHub(repo.trim() || "lerobot/svla_so101_pickplace", episodes);
    onClose();
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="enter sk-depth relative mb-3 rounded border border-line-strong bg-plate p-3.5"
    >
      <GlowEdge />
      <div className="mb-2 flex items-center justify-between">
        <span className="flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-[0.16em] text-ink"><IsoIcon name="dataset" size={18} />{mode === "file" ? "Import demonstrations" : "Import from Hugging Face"}</span>
        <button type="button" onClick={onClose} className="rounded-full p-1 text-ink-3 hover:bg-rail hover:text-ink" aria-label="Cancel import">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {mode === "file" && file && (
        <div className="mb-2.5 flex items-center gap-2 text-sm">
          <FileArchive className="h-4 w-4 shrink-0 text-ink-3" />
          <span className="min-w-0 flex-1 truncate text-ink-2">{file.name}</span>
          <span className="shrink-0 text-xs text-ink-3">{fmtSize(file.size)}</span>
        </div>
      )}
      {mode === "hub" && (
        <label className="mb-2.5 block">
          <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">Dataset repo id</span>
          <input
            autoFocus
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            placeholder="lerobot/svla_so101_pickplace"
            className="h-8 w-full rounded-sm border border-line-strong bg-plate px-2.5 font-mono text-sm text-ink placeholder:text-ink-4 focus:border-servo focus:outline-none"
          />
        </label>
      )}
      <div className="flex items-end gap-2">
        <label className="block">
          <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">Max episodes (1–50)</span>
          <input
            type="number"
            min={1}
            max={50}
            value={episodes}
            onChange={(e) => setEpisodes(clampEpisodes(Number(e.target.value)))}
            className="num h-8 w-24 rounded-sm border border-line-strong bg-plate px-2.5 font-mono text-sm text-ink focus:border-servo focus:outline-none"
            aria-label="Max episodes"
          />
        </label>
        <p className="min-w-0 flex-1 pb-1.5 text-[11px] leading-snug text-ink-4">
          {mode === "file" ? ".zip / .tar.gz · ≤ 2 GB" : "Public LeRobot dataset"} · imported, not sim-verified
        </p>
        <button type="submit" className="sk-pill sk-pill-primary inline-flex h-8 shrink-0 items-center gap-1.5 px-4 text-[13px] font-medium">
          <Upload className="h-3.5 w-3.5" /> Import
        </button>
      </div>
    </form>
  );
}
