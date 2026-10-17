-- Autopilot runs: the closed improvement loop (collect -> analyze -> version ->
-- train -> evaluate -> recommend -> repeat) and every step it took.
CREATE TABLE IF NOT EXISTS autopilot_runs (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
