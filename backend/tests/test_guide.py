"""The guided chat with no model key: from "teach it to pick up the cube" to collection, over REST + SSE."""

import json

import pytest
from fastapi.testclient import TestClient

from cortex.intake import start, status
from cortex.services.guide_answers import answer, match_option
from cortex.sim.so101 import configure_gl

configure_gl()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY", "PLANNER_BASE_URL", "FUTURAMA_SSH_HOST"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app
    from cortex.services import agent

    agent.HISTORY.clear()
    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def say(c, pid, message, patch=None):
    """POST a chat turn; returns the parsed SSE events as (event, data) pairs."""
    body = {"message": message, **({"patch": patch} if patch is not None else {})}
    with c.stream("POST", f"/api/projects/{pid}/agent", json=body) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())
    out = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        ev = next((ln[7:] for ln in frame.splitlines() if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in frame.splitlines() if ln.startswith("data: ")), None)
        if ev:
            out.append((ev, json.loads(data) if data else {}))
    assert out[-1][0] == "done"
    return out


def cards(events, kind=None):
    return [d for e, d in events if e == "card" and (kind is None or d["type"] == kind)]


def texts(events):
    return " ".join(d["delta"] for e, d in events if e == "text")


def test_free_text_answers_map_onto_the_open_question():
    spec = start("pick up the water bottle")
    q = next(q for q in status(spec).questions if q.path.startswith("objects."))
    assert match_option("the small one", q.options) == "small water bottle"
    assert answer(spec, q, "an empty water bottle")["objects"][0]["mass_kg"] == 0.02
    assert answer(spec, q, "2")["objects"][0]["mass_kg"] == 0.02  # option number
    assert answer(spec, q, "purple") is None
    cube = start("pick up the cube")
    cq = next(q for q in status(cube).questions if q.path == "objects.cube")
    assert answer(cube, cq, "yes") == {"objects": [{"name": "cube", "confirm": True}]}
    assert answer(cube, cq, "it's a 4 cm cube, 50 g")["objects"][0] == {
        "name": "cube", "shape": "box", "size": [0.04, 0.04, 0.04], "mass_kg": 0.05}


def test_guided_chat_from_prompt_to_collection(client):
    prompt = "teach it to pick up the cube"
    pid = client.post("/api/intake", json={"prompt": prompt}).json()["project_id"]

    ev = say(client, pid, prompt)  # kickoff: the prompt itself is not parsed as an answer
    assert ("step", {"step": "intake", "index": 1, "label": "Describe",
                     "steps": ["intake", "simulate", "collect", "train", "evaluate"]}) in ev
    q = cards(ev, "question")[0]
    assert q["path"] == "objects.cube" and q["options"][0]["patch"] == {"objects": [{"name": "cube", "confirm": True}]}

    ev = say(client, pid, "purple")  # not an answer: ask again with the options
    assert "couldn't match" in texts(ev) and cards(ev, "question")[0]["path"] == "objects.cube"

    ev = say(client, pid, "yes, that's right")
    assert any(e == "tool_call" and d["tool"] == "update_intake" for e, d in ev)
    assert cards(ev, "question")[0]["path"] == "zones"

    # A one-click answer: the card's patch rides along with the message.
    zq = cards(ev, "question")[0]
    ev = say(client, pid, zq["options"][0]["label"], patch=zq["options"][0]["patch"])
    pq = cards(ev, "question")[0]
    assert pq["path"] == "task.place" and [o["label"] for o in pq["options"]] == ["spot a", "spot b"]

    ev = say(client, pid, "the first one")  # completes the intake, so the scene is built right away
    assert any(e == "tool_call" and d["tool"] == "build_simulation" for e, d in ev)
    img = cards(ev, "image")[0]
    assert img["label"].startswith("FIG.02") and img["image_jpeg_base64"] and img["url"]
    feas = cards(ev, "feasibility")[0]
    assert feas["ok"] and feas["objects"][0]["grasp"]["ok"] and feas["objects"][0]["payload"]["ok"]
    assert ("step", {"step": "simulate", "index": 2, "label": "Simulate",
                     "steps": ["intake", "simulate", "collect", "train", "evaluate"]}) in ev

    ev = say(client, pid, "Collect demonstrations")
    call = next(d for e, d in ev if e == "tool_call" and d["tool"] == "start_agent_collection")
    assert call["args"]["collector"] == "oracle"  # no model key: the labelled scripted oracle, said plainly
    assert "scripted IK oracle" in texts(ev) and "sim_scripted" in texts(ev)
    job = cards(ev, "job")[0]
    assert job["kind"] == "agent_collect" and job["job_id"].startswith("job")

    ev = say(client, pid, "continue")  # while it runs, the guide shows progress instead of starting more
    assert cards(ev, "job") or any(e == "tool_call" for e, _ in ev)

    hist = client.get(f"/api/projects/{pid}/agent/history").json()
    assert any(h["role"] == "event" and h["event"] == "card" for h in hist)
    assert not any("image_jpeg_base64" in h.get("data", {}) for h in hist)  # images are kept by URL only
    client.post(f"/api/agent-collections/{call_collection(client, pid)}/stop")


def call_collection(c, pid):
    return c.get(f"/api/projects/{pid}/agent-collections").json()[0]["id"]


def test_infeasible_object_gets_one_click_suggestions(client):
    pid = client.post("/api/intake", json={"prompt": "pick up the water bottle"}).json()["project_id"]
    say(client, pid, "pick up the water bottle")
    ev = say(client, pid, "the full one", patch={"objects": [{"name": "water_bottle", "confirm": True}]})
    while (q := cards(ev, "question")):
        ev = say(client, pid, "ok", patch=q[0]["options"][0]["patch"])
    feas = cards(ev, "feasibility")[0]
    assert not feas["ok"]
    assert any(s["label"].startswith("Use a") for s in feas["suggestions"])
