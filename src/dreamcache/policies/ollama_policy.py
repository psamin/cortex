"""Ollama HTTP client and LLM fast policy.

Calls the official /api/chat endpoint, non-streaming, with a JSON-schema `format`
and `think: false` (no chain-of-thought tokens). Responses are validated with
Pydantic. A sqlite response cache keyed on the full request makes deterministic
(temperature 0) calls reproducible across methods; cached calls are still
*charged* their original tokens and server time so compute accounting is honest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from pydantic import BaseModel, ValidationError

from dreamcache.environments.minigrid_adapter import ACTION_NAMES, NAME_TO_ACTION
from dreamcache.environments.observations import describe
from dreamcache.types import Cost, Decision

SYSTEM = ("You control an agent in a 2D grid world. Choose actions that complete the "
          "mission. Respond only with the requested JSON.")
INSTRUCTION = ("Rate how good each available action is right now for completing the "
               "mission safely, from 0 (terrible) to 10 (best), then name the best one.")
RULES = {
    "dynamic": ("Rules: moving forward into a wall (#) or a moving obstacle (O) ends the episode "
                "as a failure. Obstacles move one cell randomly each step. Turning changes only "
                "your facing direction. Reach G to succeed."),
    "doorkey": ("Rules: walls (#) and closed doors block movement. Pick up the key (K) while "
                "facing it, then face the locked door (D) and toggle it to open, then walk to G."),
}
REASON_INSTRUCTION = ("First write a very short reason (at most 25 words) about where the goal "
                      "and dangers are relative to you, then rate each available action from 0 "
                      "(terrible) to 10 (best) and name the best one.")


class LLMReply(BaseModel):
    scores: dict[str, float]
    action: str
    reason: str | None = None


class OllamaClient:
    def __init__(self, host=None, model="qwen3:4b", timeout=300.0, retries=3,
                 num_predict=80, num_ctx=4096, cache_path=None, think=False):
        self.host = host or os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
        self.model, self.retries, self.think = model, retries, think
        self.num_predict, self.num_ctx = num_predict, num_ctx
        self.http = httpx.Client(base_url=self.host, timeout=timeout)
        # cache_path="" disables caching; None falls back to $DC_LLM_CACHE
        self.cache_path = os.environ.get("DC_LLM_CACHE") if cache_path is None else cache_path
        self._db = None
        self._lock = threading.Lock()

    # ---- cache -------------------------------------------------------------
    def _cache(self):
        with self._lock:
            return self._open()

    def _open(self):
        if self.cache_path and self._db is None:
            self._db = sqlite3.connect(self.cache_path, timeout=60, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        return self._db

    def available(self) -> list[str]:
        r = self.http.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def digest(self) -> str | None:
        for m in self.http.get("/api/tags").json().get("models", []):
            if m["name"] == self.model:
                return m.get("digest")
        return None

    def warmup(self):
        self.chat([{"role": "user", "content": "Say ok."}], schema=None, temperature=0, seed=0,
                  use_cache=False)

    def chat(self, messages, schema, temperature, seed, use_cache=True):
        """Returns (content, Cost). Retries on transport errors/timeouts."""
        body = {"model": self.model, "messages": messages, "stream": False,
                "think": self.think, "keep_alive": -1,
                "options": {"temperature": temperature, "seed": seed,
                            "num_predict": self.num_predict, "num_ctx": self.num_ctx}}
        if schema is not None:
            body["format"] = schema
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        db = self._cache() if use_cache else None
        if db is not None:
            with self._lock:
                row = db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
            if row:
                v = json.loads(row[0])
                return v["content"], Cost(llm_calls=1, prompt_tokens=v["pt"], gen_tokens=v["gt"],
                                          llm_seconds=v["s"], cache_hits=1)
        last = None
        for attempt in range(self.retries):
            try:
                t0 = time.perf_counter()
                r = self.http.post("/api/chat", json=body)
                r.raise_for_status()
                j = r.json()
                wall = time.perf_counter() - t0
                content = j["message"]["content"]
                pt, gt = j.get("prompt_eval_count", 0), j.get("eval_count", 0)
                s = j.get("total_duration", wall * 1e9) / 1e9
                if db is not None:
                    with self._lock:
                        db.execute("INSERT OR REPLACE INTO c VALUES (?,?)",
                                   (key, json.dumps({"content": content, "pt": pt, "gt": gt, "s": s})))
                        db.commit()
                return content, Cost(llm_calls=1, prompt_tokens=pt, gen_tokens=gt, llm_seconds=s)
            except (httpx.HTTPError, KeyError, ValueError) as e:  # noqa: PERF203
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed after {self.retries} tries: {last}")

    def close(self):
        self.http.close()
        if self._db is not None:
            self._db.close()


def _schema(legal, reason=False):
    names = [ACTION_NAMES[a] for a in legal]
    props = {"reason": {"type": "string"}} if reason else {}
    return {"type": "object",
            "properties": {**props, "scores": {"type": "object",
                                      "properties": {n: {"type": "number"} for n in names},
                                      "required": names},
                           "action": {"type": "string", "enum": names}},
            "required": (["reason"] if reason else []) + ["scores", "action"]}


def parse_reply(content: str, legal) -> tuple[dict[int, float], int, bool]:
    """-> (raw scores per action, chosen action, malformed?). Recovers embedded JSON."""
    names = {ACTION_NAMES[a]: a for a in legal}
    try:
        rep = LLMReply.model_validate_json(content)
    except ValidationError:
        m = re.search(r"\{.*\}", content, re.S)
        try:
            rep = LLMReply.model_validate_json(m.group(0)) if m else None
        except ValidationError:
            rep = None
    if rep is None:
        return {a: 1.0 for a in legal}, legal[0], True
    scores = {a: max(0.0, float(rep.scores.get(n, 0.0))) for n, a in names.items()}
    act = names.get(rep.action)
    malformed = act is None
    if act is None:
        act = max(scores, key=scores.get)
    return scores, act, malformed


def normalize(scores: dict[int, float]) -> dict[int, float]:
    tot = sum(scores.values())
    if tot <= 0:
        return {a: 1 / len(scores) for a in scores}
    return {a: v / tot for a, v in scores.items()}


class LLMPolicy:
    """Deployable fast policy: sees only the text observation and own history."""

    privileged = False

    def __init__(self, client: OllamaClient, legal, proposal_temperature=1.0, max_workers=4,
                 prompt="v1", family="dynamic"):
        """prompt: v1 = observation + rating instruction; v2 = v1 + environment rules;
        v3 = v2 + a short free-text reason field before the scores."""
        self.client, self.legal, self.prompt, self.family = client, legal, prompt, family
        self.name = f"llm:{client.model}:{prompt}"
        self.schema = _schema(legal, reason=prompt == "v3")
        self.proposal_temperature = proposal_temperature
        self.pool = ThreadPoolExecutor(max_workers)

    def _messages(self, obs, history):
        body = describe(obs, history, self.legal)
        if self.prompt in ("v2", "v3"):
            body = RULES[self.family] + "\n\n" + body
        instr = REASON_INSTRUCTION if self.prompt == "v3" else INSTRUCTION
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": body + "\n" + instr}]

    def decide(self, obs, history) -> Decision:
        t0 = time.perf_counter()
        content, cost = self.client.chat(self._messages(obs, history), self.schema, 0.0, 0)
        raw, act, bad = parse_reply(content, self.legal)
        p = normalize(raw)
        # The chosen action is the argmax of scores; the "action" field breaks ties.
        best = max(p.values())
        tied = [a for a in self.legal if p[a] == best]
        action = act if act in tied else tied[0]
        cost.wall_seconds = time.perf_counter() - t0
        cost.malformed = int(bad)
        return Decision(action=action, probs=p, cost=cost)

    def propose(self, obs, history, n: int, seed: int) -> tuple[list[int], Cost]:
        """n independent stochastic proposals (concurrent requests to one server)."""
        t0 = time.perf_counter()
        msgs = self._messages(obs, history)
        futs = [self.pool.submit(self.client.chat, msgs, self.schema,
                                 self.proposal_temperature, seed * 1000 + i) for i in range(n)]
        acts, total = [], Cost()
        for f in futs:
            content, c = f.result()
            raw, act, bad = parse_reply(content, self.legal)
            c.malformed = int(bad)
            acts.append(act)
            total += c
        total.wall_seconds = time.perf_counter() - t0
        return acts, total

    # ---- draft / verify (speculative-decoding-style methods) -------------------
    def _names(self):
        return [ACTION_NAMES[a] for a in self.legal]

    def _ask(self, obs, history, instr, schema, temperature=0.0, seed=0):
        body = describe(obs, history, self.legal)
        if self.prompt in ("v2", "v3"):
            body = RULES[self.family] + "\n\n" + body
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body + "\n" + instr}]
        t0 = time.perf_counter()
        content, cost = self.client.chat(msgs, schema, temperature, seed)
        cost.wall_seconds = time.perf_counter() - t0
        try:
            return json.loads(content), cost
        except ValueError:
            m = re.search(r"\{.*\}", content, re.S)
            try:
                return (json.loads(m.group(0)) if m else {}), cost
            except ValueError:
                cost.malformed = 1
                return {}, cost

    def verify(self, obs, history, proposed: int):
        """Big-model check of a drafted action: -> (accepted, action_to_take, Cost)."""
        names = self._names()
        schema = {"type": "object", "properties": {"accept": {"type": "boolean"},
                                                   "action": {"type": "string", "enum": names}},
                  "required": ["accept", "action"]}
        instr = (f"A fast assistant proposes the action '{ACTION_NAMES[proposed]}'. If it is a good, "
                 "safe action right now, set accept=true and repeat it. Otherwise set accept=false "
                 "and give the best action.")
        j, cost = self._ask(obs, history, instr, schema)
        act = NAME_TO_ACTION.get(j.get("action"), proposed)
        acc = bool(j.get("accept", True)) and act == proposed
        if act not in self.legal:
            act = proposed
        return acc, (proposed if acc else act), cost

    def draft_plan(self, obs, history, length: int):
        """Small-model draft of the next `length` actions (one call)."""
        names = self._names()
        schema = {"type": "object", "properties": {"plan": {"type": "array", "items": {"type": "string", "enum": names},
                                                            "minItems": 1, "maxItems": length}},
                  "required": ["plan"]}
        instr = (f"Plan your next {length} actions in order (obstacles may move, so plan carefully). "
                 "Return them as a list.")
        j, cost = self._ask(obs, history, instr, schema)
        plan = [NAME_TO_ACTION[n] for n in j.get("plan", []) if n in NAME_TO_ACTION and NAME_TO_ACTION[n] in self.legal]
        return plan[:length] or [self.legal[0]], cost

    def verify_plan(self, obs, history, plan: list[int]):
        """Big-model check of a drafted plan: -> (n_accepted_prefix, correction, Cost)."""
        names = self._names()
        schema = {"type": "object", "properties": {"accepted_steps": {"type": "integer", "minimum": 0, "maximum": len(plan)},
                                                   "next_action": {"type": "string", "enum": names}},
                  "required": ["accepted_steps", "next_action"]}
        listing = ", ".join(f"{i + 1}. {ACTION_NAMES[a]}" for i, a in enumerate(plan))
        instr = (f"A fast assistant drafted this plan: {listing}. How many of the FIRST steps are correct "
                 "to execute starting now (0 if the first step is wrong)? Then give the action that "
                 "should be taken right after the accepted steps (or now, if none are accepted).")
        j, cost = self._ask(obs, history, instr, schema)
        n = int(max(0, min(len(plan), j.get("accepted_steps", 0) or 0)))
        corr = NAME_TO_ACTION.get(j.get("next_action"), plan[0])
        return n, (corr if corr in self.legal else plan[0]), cost
