"""Two-level execution graph for predictive execution speculation (experiment 2).

Level 1 (workflow graph): nodes are operation classes; in MiniGrid an operation is
"actor decision -> action class" (left/right/forward/...). Edges count observed
transitions op_t -> op_{t+1}, optionally per context.

Level 2 (execution-state graph): one row per concrete operation instance: workflow/env
id, normalized context key, input artifact hash (hash of the exact actor prompt),
operation result, duration, whether it was speculative, reused, validated, and cost.

Storage is a single SQLite file. Only normalized features and hashes are stored, never
raw observations.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from dreamcache.environments.observations import features

SCHEMA = """
CREATE TABLE IF NOT EXISTS exec_state (
  id INTEGER PRIMARY KEY, workflow TEXT, env TEXT, episode TEXT, t INTEGER,
  context TEXT, prev_op TEXT, op TEXT, input_hash TEXT, env_version TEXT,
  duration_s REAL, speculative INTEGER, reused INTEGER, validated INTEGER, cost_json TEXT);
CREATE TABLE IF NOT EXISTS workflow_edge (
  workflow TEXT, env TEXT, src TEXT, dst TEXT, count INTEGER,
  PRIMARY KEY (workflow, env, src, dst));
"""


def goal_bucket(f) -> str:
    if not f.goal_visible:
        return "none"
    ahead, right = f.goal_rel
    if right == 0:
        return "ahead"
    return "right" if right > 0 else "left"


def context_key(obs, history) -> str:
    """Deterministic, observation-only normalization of the decision context.

    Uses only what the deployed agent sees (egocentric view, own last action). Never
    uses the actor's upcoming action or any simulator state.
    """
    f = features(obs)
    last = history[-1] if history else -1
    obst = min(f.nearby_obstacles, 2)
    return f"F={f.front}|L={f.left}|R={f.right}|G={goal_bucket(f)}|O={obst}|last={last}"


def input_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]


@dataclass
class Transition:
    workflow: str
    env: str
    episode: str
    t: int
    context: str
    prev_op: str
    op: str


class ExecGraph:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def add(self, tr: Transition, input_h="", env_version="", duration=0.0, speculative=False,
            reused=False, validated=False, cost=None):
        self.db.execute("INSERT INTO exec_state (workflow, env, episode, t, context, prev_op, op, input_hash,"
                        " env_version, duration_s, speculative, reused, validated, cost_json)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (tr.workflow, tr.env, tr.episode, tr.t, tr.context, tr.prev_op, tr.op, input_h,
                         env_version, duration, int(speculative), int(reused), int(validated),
                         json.dumps(cost or {})))
        self.db.execute("INSERT INTO workflow_edge VALUES (?,?,?,?,1) ON CONFLICT(workflow, env, src, dst)"
                        " DO UPDATE SET count = count + 1", (tr.workflow, tr.env, tr.prev_op, tr.op))

    def edges(self, workflow=None):
        q = "SELECT workflow, env, src, dst, count FROM workflow_edge"
        return self.db.execute(q + (" WHERE workflow=?" if workflow else ""),
                               (workflow,) if workflow else ()).fetchall()

    def commit(self):
        self.db.commit()
