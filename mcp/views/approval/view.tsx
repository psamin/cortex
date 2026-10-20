import { useState } from "react";
import { ThemeProvider, useDynamicTool, useToolContext } from "mcp-use/react";
import type { ReviewOut } from "../../src/views.js";
import { Notice, Shell, Stat } from "../kit.js";

type ApproveArgs = { project_id: string; training_job_id: string; nonce: string };

export default function ApprovalView() {
  const view = useToolContext();
  const approve = useDynamicTool<ApproveArgs, { training_job_id: string; status: string }>("cortex.view_approve_training");
  const [done, setDone] = useState<string | null>(null);

  if (view.status === "error") return <Notice>Could not load the training job: {view.error.message}</Notice>;
  if (view.status !== "ready") return <Notice>Loading the training proposal…</Notice>;
  const tj = view.toolOutput as unknown as ReviewOut;
  // Single-use nonce in view-only _meta: the model never sees it, so only this button can approve.
  const nonce = view.meta?.["cortex/approval_nonce"] as string | undefined;
  const status = done ?? tj.status;
  const e = tj.estimate;

  async function onApprove() {
    if (!nonce) return;
    try {
      const r = await approve.callTool({ project_id: tj.project_id, training_job_id: tj.training_job_id, nonce });
      setDone(r.structuredContent.status);
    } catch {
      /* approve.error carries the message */
    }
  }

  return (
    <ThemeProvider>
      <Shell
        eyebrow="Human approval required · training"
        title={
          <>
            {tj.policy_type.toUpperCase()} training on {tj.backend.replace("_", " ")} <span className="pill mono">{tj.training_job_id}</span>
          </>
        }
      >
        <div className="row" style={{ marginBottom: 12 }}>
          <Stat label="GPU hours" value={e.gpu_hours} />
          <Stat label="est. cost" value={e.paid ? `$${e.est_cost_usd.toFixed(2)}` : "free"} />
          <Stat label="steps" value={tj.steps ?? "–"} />
          {tj.dataset ? <Stat label="source demos" value={tj.dataset.source_episodes} /> : null}
        </div>
        <table>
          <tbody>
            <tr>
              <th>GPU</th>
              <td>{e.gpu}</td>
            </tr>
            {tj.dataset ? (
              <tr>
                <th>Dataset</th>
                <td>
                  <span className="mono">{tj.dataset.id}</span> {tj.dataset.name} · {tj.dataset.train_episodes} train /{" "}
                  {tj.dataset.eval_episodes} eval episodes · {tj.dataset.frames} frames
                  {tj.dataset.augmentations.length ? ` · augmented: ${tj.dataset.augmentations.join(", ")}` : ""}
                </td>
              </tr>
            ) : null}
            <tr>
              <th>Why</th>
              <td>{tj.rationale}</td>
            </tr>
            <tr>
              <th>Notes</th>
              <td className="muted">{e.notes}</td>
            </tr>
          </tbody>
        </table>
        <div className="row" style={{ marginTop: 14, alignItems: "center" }}>
          {status === "proposed" && nonce ? (
            <>
              <button className="btn" disabled={approve.isPending} onClick={() => void onApprove()}>
                {approve.isPending ? "Approving…" : "Approve training"}
              </button>
              <a className="btn ghost" href={tj.studio_url} target="_blank" rel="noreferrer">
                Review in studio
              </a>
            </>
          ) : status === "proposed" ? (
            <span className="muted">
              This approval link was used or expired. Ask the agent to show the training review again, or approve in the{" "}
              <a href={tj.studio_url} target="_blank" rel="noreferrer">studio</a>.
            </span>
          ) : (
            <span className="ok">
              {status === "queued" || status === "running"
                ? "Approved and started."
                : status === "approved"
                  ? "Approved, but it didn't start yet. Launch it from the studio or ask the agent."
                  : `Status: ${status}`}
              {tj.approved_by && !done ? <span className="muted"> · by {tj.approved_by}</span> : null}
            </span>
          )}
        </div>
        {approve.error ? <p className="bad">{approve.error.message}</p> : null}
        <p className="muted" style={{ fontSize: 12 }}>
          Only you can approve: the agent never sees this button's single-use code.
        </p>
      </Shell>
    </ThemeProvider>
  );
}
