import { useEffect, useState } from "react";
import { ThemeProvider, useDynamicTool, useToolContext } from "mcp-use/react";
import type { EvaluationOut } from "../../src/views.js";
import { Notice, pct, Shell, Stat } from "../kit.js";

const REFRESH_MS = 3000;

export default function EvaluationView() {
  const view = useToolContext();
  const refresh = useDynamicTool<{ project_id: string; evaluation_id?: string }, EvaluationOut>("cortex.view_evaluation_report");
  const [snap, setSnap] = useState<EvaluationOut | null>(null);
  const ev = snap ?? (view.status === "ready" ? (view.toolOutput as unknown as EvaluationOut) : null);

  useEffect(() => {
    if (!ev?.running) return;
    const id = setTimeout(() => {
      refresh
        .callTool({ project_id: ev.project_id, evaluation_id: ev.evaluation_id })
        .then((r) => setSnap(r.structuredContent))
        .catch(() => undefined);
    }, REFRESH_MS);
    return () => clearTimeout(id);
  }, [ev?.running, ev?.project_id, ev?.evaluation_id, snap]);

  if (view.status === "error") return <Notice>Could not load the evaluation: {view.error.message}</Notice>;
  if (!ev) return <Notice>Loading the evaluation…</Notice>;

  const done = ev.branches.filter((b) => b.n > 0);
  const n = done.reduce((s, b) => s + b.n, 0);
  const k = done.reduce((s, b) => s + b.successes, 0);
  return (
    <ThemeProvider>
      <Shell
        eyebrow={`Simulation evaluation · ${ev.policy_kind === "checkpoint" ? "learned policy" : "scripted reference"}`}
        title={
          <>
            {ev.policy_label} <span className="pill mono">{ev.evaluation_id}</span>
          </>
        }
      >
        <div className="row" style={{ marginBottom: 12 }}>
          <Stat label="overall" value={n ? pct(k / n) : "–"} />
          <Stat label="trials" value={n} />
          <Stat label="status" value={<span style={{ fontSize: 14 }}>{ev.running ? "running…" : ev.status}</span>} />
        </div>
        <table>
          <thead>
            <tr>
              <th>condition</th>
              <th>success</th>
              <th style={{ width: "34%" }}>95% interval</th>
              <th>failure modes (measured)</th>
            </tr>
          </thead>
          <tbody>
            {ev.branches.map((b) => (
              <tr key={b.name}>
                <td>
                  {b.name}
                  {b.name === ev.weakest ? <span className="pill bad" style={{ marginLeft: 6 }}>weakest</span> : null}
                </td>
                <td className="mono">{b.n ? `${b.successes}/${b.n} · ${pct(b.success_rate)}` : b.status}</td>
                <td>
                  {b.ci95 ? (
                    <div className="bar" title={`${pct(b.ci95[0])}–${pct(b.ci95[1])}`}>
                      <i style={{ left: `${b.ci95[0] * 100}%`, width: `${(b.ci95[1] - b.ci95[0]) * 100}%` }} />
                      <b style={{ left: `calc(${(b.success_rate ?? 0) * 100}% - 1px)` }} />
                    </div>
                  ) : (
                    <span className="muted">–</span>
                  )}
                </td>
                <td className="muted">
                  {Object.entries(b.failure_modes)
                    .map(([m, c]) => `${m.replaceAll("_", " ")} ×${c}`)
                    .join(", ") || "–"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: 12 }}>
          Measured from simulator telemetry on held-out seeds.{" "}
          <a href={ev.studio_url} target="_blank" rel="noreferrer">
            Open in studio
          </a>
        </p>
      </Shell>
    </ThemeProvider>
  );
}
