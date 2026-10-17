-- Autonomous (GPT-6 Astra) demonstration collection. Every attempt is stored;
-- only simulator-verified successes become rows in `episodes`.
CREATE TABLE IF NOT EXISTS agent_collections (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  data JSONB NOT NULL, -- {collector, provider, model, mode, request, scenarios, totals}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_attempts (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT NOT NULL, -- success | failed
  data JSONB NOT NULL, -- {collection_id, seed, scenario, usage, recoveries, provider, model, artifacts{trace,video,npz}}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_attempts_project_idx ON agent_attempts(project_id);
