import type { TrainingJob } from "../api/types";

/**
 * Why a training job failed, in plain words. The backend classifies its stored error into a `failure_kind`
 * (training_service.FAILURE_KINDS); the raw error stays in the backend logs (#328), the cause is shown here (#348).
 */
const CAUSE: Record<string, { short: string; long: string }> = {
  cluster_unreachable: {
    short: "couldn't reach the cluster",
    long: "Couldn't reach the cluster. Check the VPN and that ssh to the cluster works, then retry.",
  },
  tool_missing: {
    short: "rsync or ssh is missing",
    long: "This machine is missing a tool training needs (ssh, rsync or scp). Install it, restart the backend, then retry.",
  },
  subprocesses_unavailable: {
    short: "backend can't run training",
    long: "The backend can't start child processes. On Windows, restart it without --reload, then retry.",
  },
  interrupted: {
    short: "interrupted by a restart",
    long: "The backend restarted before the cluster accepted the job. Retry to submit it again.",
  },
  cluster_rejected: {
    short: "the cluster rejected the job",
    long: "The cluster rejected the job (partition, GPU or time limit). Check the cluster settings in Setup, then retry.",
  },
  upload_failed: {
    short: "upload to the cluster failed",
    long: "Copying the dataset to the cluster failed. Check the connection and disk space on the cluster, then retry.",
  },
  training_crashed: {
    short: "training crashed on the cluster",
    long: "Training started on the cluster but ended without finishing (for example out of GPU memory or time).",
  },
  checkpoint_missing: {
    short: "no checkpoint came back",
    long: "Training finished but the checkpoint couldn't be fetched from the cluster. Retry, or register it by hand.",
  },
  other: { short: "failed", long: "Training failed. The backend log has the details." },
};

export function failureCause(t: Pick<TrainingJob, "status" | "failure_kind">): { short: string; long: string } | null {
  if (t.status !== "failed") return null;
  return CAUSE[t.failure_kind ?? "other"] ?? CAUSE.other;
}

/** A failed or stopped job that was approved can run again from the studio (POST /launch resets it). */
export const canRetry = (t: Pick<TrainingJob, "status" | "approved_by">) =>
  (t.status === "failed" || t.status === "cancelled") && !!t.approved_by;
