-- Real trials (docs/STACK_SPEC.md 2.2): one rollout of a checkpoint on the physical arm, judged by a person.
-- `data` is cortex.services.real_trials.RealTrial: checkpoint_id, condition, success, failure_mode, judged_by,
-- source (studio | agent: an agent recorded what a person reported), voided (withdrawn in the studio).
CREATE TABLE IF NOT EXISTS real_trials (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL, -- success | failure | voided
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS real_trials_project_idx ON real_trials(project_id);

ALTER TABLE real_trials ENABLE ROW LEVEL SECURITY;
