import { useEffect, useState } from "react";
import { ThemeProvider, useDynamicTool, useToolContext } from "mcp-use/react";
import type { CollectionOut } from "../../src/views.js";
import { Notice, Shell, Stat } from "../kit.js";

type Snapshot = CollectionOut & { live_frame_b64: string | null; live_frame_age_s: number | null };
const REFRESH_MS = 2000;

export default function CollectionView() {
  const view = useToolContext();
  const refresh = useDynamicTool<{ project_id: string; collection_id?: string }, Snapshot>("cortex.view_collection_snapshot");
  const [snap, setSnap] = useState<Snapshot | null>(null);

  const first = view.status === "ready" ? (view.toolOutput as unknown as CollectionOut) : undefined;
  const firstFrame = view.status === "ready" ? (view.meta?.["cortex/live_frame_b64"] as string | undefined) : undefined;
  const live = snap ?? (first ? { ...first, live_frame_b64: firstFrame ?? null, live_frame_age_s: null } : null);

  useEffect(() => {
    if (!live?.running) return;
    const id = setTimeout(() => {
      refresh
        .callTool({ project_id: live.project_id, collection_id: live.collection_id })
        .then((r) => setSnap(r.structuredContent))
        .catch(() => undefined); // keep the last good snapshot; the next tick retries
    }, REFRESH_MS);
    return () => clearTimeout(id);
  }, [live?.running, live?.project_id, live?.collection_id, snap]);

  if (view.status === "error") return <Notice>Could not load the collection: {view.error.message}</Notice>;
  if (!live) return <Notice>Loading the collection…</Notice>;

  const failed = live.attempts - live.successes;
  return (
    <ThemeProvider>
      <Shell
        eyebrow={`Agent demonstrations · ${live.collector}${live.model ? ` · ${live.model}` : ""}`}
        title={
          <>
            {live.running ? "Collecting demonstrations" : `Collection ${live.status}`}{" "}
            <span className="pill mono">{live.collection_id}</span>
          </>
        }
      >
        {live.error && !live.running ? (
          <p className="bad" style={{ marginTop: -4 }}>
            {live.error}
          </p>
        ) : null}
        <div className="row" style={{ alignItems: "flex-start" }}>
          <div style={{ flex: "1 1 240px" }}>
            {live.live_frame_b64 ? (
              <img className="frame" alt="Latest MuJoCo frame" src={`data:image/jpeg;base64,${live.live_frame_b64}`} />
            ) : (
              <div className="frame" style={{ display: "grid", placeItems: "center" }}>
                <span className="muted">No live frame yet</span>
              </div>
            )}
            <div className="muted mono" style={{ fontSize: 11, marginTop: 4 }}>
              MuJoCo · SO-101 {live.running ? "· live" : ""}
            </div>
          </div>
          <div style={{ flex: "1 1 260px" }}>
            <div className="row" style={{ marginBottom: 12 }}>
              <Stat label="successes" value={<span className="ok">{live.successes}</span>} />
              <Stat label="failed" value={<span className={failed ? "bad" : ""}>{failed}</span>} />
              <Stat label="planned" value={live.planned} />
              <Stat label="model calls" value={live.model_calls} />
              <Stat label="spend" value={`$${live.cost_usd.toFixed(2)}`} />
            </div>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>condition</th>
                  <th>result</th>
                  <th>calls</th>
                </tr>
              </thead>
              <tbody>
                {live.recent
                  .slice()
                  .reverse()
                  .map((a) => (
                    <tr key={a.index}>
                      <td className="mono">{a.index}</td>
                      <td className="mono">{a.condition}</td>
                      <td className={a.status === "success" ? "ok" : "bad"}>
                        {a.status}
                        {a.stop_reason && a.status !== "success" ? <span className="muted"> · {a.stop_reason}</span> : null}
                      </td>
                      <td className="mono">{a.model_calls ?? "–"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <p className="muted" style={{ fontSize: 12 }}>
              Only simulator-verified successes become LeRobot episodes; failures are kept as diagnostics.{" "}
              <a href={live.studio_url} target="_blank" rel="noreferrer">
                Open in studio
              </a>
            </p>
          </div>
        </div>
      </Shell>
    </ThemeProvider>
  );
}
