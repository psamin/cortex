-- Lock every Cortex table behind row-level security. With RLS on and no policies,
-- the public anon key and user JWTs can read/write nothing; the backend's admin
-- (ik_) key bypasses RLS, so Cortex itself is unaffected. Tables created with raw
-- SQL migrations have RLS off by default on InsForge (verified 2026-10-17: the anon
-- key could read `episodes`). Postgres-only; the SQLite fallback skips it.
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE robot_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE episode_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE dataset_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE dataset_episode_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE failure_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE evaluation_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_branches ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE autopilot_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_attempts ENABLE ROW LEVEL SECURITY;
