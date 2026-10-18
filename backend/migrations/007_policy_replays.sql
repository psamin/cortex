-- Policy replays (#179). A trained policy run open loop on held-out recorded demos: it sees the recorded
-- camera images and arm state frame by frame and its joint targets are compared with the human's. It
-- measures imitation (joint error), not task success, so it is kept apart from evaluation_runs.
-- `data` is cortex.models.PolicyReplay: kind held_out_replay (a Futurama replay job) or live_inference
-- (a session streamed to the studio's GPU tab), with per-episode joint error, inference time and video keys.
CREATE TABLE IF NOT EXISTS policy_replays (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL, -- complete | partial (a live session the viewer left early)
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS policy_replays_project_idx ON policy_replays(project_id);

ALTER TABLE policy_replays ENABLE ROW LEVEL SECURITY;
