"""TEST DOUBLE ONLY: a local OpenAI-compatible chat-completions server that plays
the planner role for integration tests. It is a color-blob heuristic over the
images the worker sends (it never sees simulator state) and is never used to
produce demonstrations outside tests.

Episodes are counted from the "Begin." message; episode numbers listed in
`sabotage` point 70 px away from the cube and give up, producing a genuine failed
attempt.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


def blob_center(img: np.ndarray, color: str) -> tuple[float, float] | None:
    r, g, b = (img[..., i].astype(int) for i in range(3))
    # ratio tests so dim and bright scenes both work
    if color == "red":
        mask = (r > 35) & (r > 2.2 * g) & (r > 2.2 * b)
    else:
        mask = (g > 35) & (g > 1.5 * r) & (g > 1.4 * b)
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) > 20 else None


def latest_overhead(messages: list[dict]) -> np.ndarray | None:
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            text = m["content"][0].get("text", "")
            if "observe_scene" in text:
                url = m["content"][1]["image_url"]["url"]
                return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    return None


class FakePlanner:
    def __init__(self, sabotage: set[int] | None = None):
        self.sabotage = sabotage or set()
        self.episode = -1
        self.calls = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()

    def decide(self, body: dict) -> dict:
        msgs = body["messages"]
        if msgs[-1]["role"] == "user" and msgs[-1]["content"] == "Begin. The arm is at its home pose with the gripper half open.":
            self.episode += 1
        done = [json.loads(m["content"]) | {"_name": n} for m, n in zip(
            [m for m in msgs if m["role"] == "tool"],
            [c["function"]["name"] for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])])]
        names = [d["_name"] for d in done]
        bad = self.episode in self.sabotage
        if not names:
            return self._call("observe_scene", {"camera": "overhead"})
        if "propose_grasp" not in names:
            img = latest_overhead(msgs)
            u, v = blob_center(img, "red") if img is not None else (240, 240)
            if bad:
                u, v = min(u + 70, 470), v
            return self._call("propose_grasp", {"object_id": "red_cube", "image_point": {"u": u, "v": v}})
        g = next(d for d in reversed(done) if d["_name"] == "propose_grasp")
        seq = [("set_gripper", {"state": "open"}), ("move_end_effector", {"target": g["pregrasp"]}),
               ("move_end_effector", {"target": g["grasp"]}), ("set_gripper", {"state": "closed"}),
               ("verify_grasp", {})]
        after_propose = names[names.index("propose_grasp") + 1:]
        if len(after_propose) < len(seq):
            return self._call(*seq[len(after_propose)])
        held = next(d for d in reversed(done) if d["_name"] == "verify_grasp").get("holding_object")
        if not held:
            return self._call("finish_episode", {"believes_success": False, "summary": "grasp missed"})
        rest = [("move_end_effector", {"target": {**g["grasp"], "z": 0.075}}), ("observe_scene", {"camera": "overhead"})]
        if len(after_propose) < len(seq) + len(rest):
            return self._call(*rest[len(after_propose) - len(seq)])
        if "place_object" not in names:
            img = latest_overhead(msgs)
            u, v = blob_center(img, "green")
            return self._call("place_object", {"image_point": {"u": u, "v": v}})
        if names[-1] != "check_task_success":
            return self._call("check_task_success", {})
        return self._call("finish_episode", {"believes_success": bool(done[-1].get("success")), "summary": "ok"})

    def _call(self, name: str, args: dict) -> dict:
        self.calls += 1
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{self.calls}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    def _handler(self):
        planner = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                msg = planner.decide(body)
                out = json.dumps({"model": "test/fake-planner", "choices": [{"message": msg}],
                                  "usage": {"prompt_tokens": 1200, "completion_tokens": 40, "cost": 0.0}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        return H
