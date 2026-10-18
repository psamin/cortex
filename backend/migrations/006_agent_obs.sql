-- Agent observability (#105): who an MCP agent acts for, its sessions, and a richer call log.
-- Every table keeps the shared (id, project_id, status, data, created_at) shape; the extra
-- columns are the ones the API filters on.

-- An agent key ties an MCP connection to a person. Only the SHA-256 of the key is stored;
-- the key itself is shown once, at creation.
CREATE TABLE IF NOT EXISTS agent_keys (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  status TEXT NOT NULL, -- active | revoked
  owner_id TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  data JSONB NOT NULL, -- {label, prefix, created_at, revoked_at, last_used_at}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_keys_owner_idx ON agent_keys(owner_id);

-- One row per MCP connection (the MCP session id the server forwards).
CREATE TABLE IF NOT EXISTS agent_sessions (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  status TEXT NOT NULL, -- live | idle
  owner_id TEXT NOT NULL,
  data JSONB NOT NULL, -- {key_id, client, client_version, first_seen, last_seen, call_count, projects, last_tool}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_sessions_owner_idx ON agent_sessions(owner_id);

-- Tool calls now carry the agent session; status gains running | approval_required.
ALTER TABLE tool_audit_logs ADD COLUMN session_id TEXT;
CREATE INDEX IF NOT EXISTS audit_session_idx ON tool_audit_logs(session_id);

ALTER TABLE agent_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_sessions ENABLE ROW LEVEL SECURITY;
