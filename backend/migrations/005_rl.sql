-- RL sandboxes (#80). A sandbox is one proposed/accepted skill-RL run on a weak condition; its
-- `data` is cortex.rl.types.RLSandbox (weakness, env, budget, labelled estimate, progress, measured result).
-- A round links a finished sandbox's measured skill gain to the realised ACT uplift once the
-- retrained checkpoint is re-evaluated (filled in by the closing-the-loop step); it fits the transfer curve.
CREATE TABLE IF NOT EXISTS rl_sandboxes (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rl_sandboxes_project_idx ON rl_sandboxes(project_id);

CREATE TABLE IF NOT EXISTS rl_rounds (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL, -- skill_measured | realised
  data JSONB NOT NULL, -- {sandbox_id, weakness_key, condition, skill_gain, predicted_policy_uplift, realised_uplift}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rl_rounds_project_idx ON rl_rounds(project_id);

ALTER TABLE rl_sandboxes ENABLE ROW LEVEL SECURITY;
ALTER TABLE rl_rounds ENABLE ROW LEVEL SECURITY;
