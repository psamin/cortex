"""Live GPU sessions are saved to the task's project when the stream ends (#179)."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from cortex.services import live_policy_service as lp
from cortex.services.live_recording import StreamRecorder


def _jpeg(v: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((48, 64, 3), v, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _part(i: int, n: int, jpg: bytes, headers: bool = True) -> bytes:
    if not headers:  # serve_policy.py before #179
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\nX-Frame: {i}/{n}\r\n"
            f"X-Inference-Ms: {40 + i % 3}\r\nX-Joint-Err-Deg: 0.5\r\nX-Gripper-Err: 0.25\r\n"
            f"X-Instruction: pull%20a%20block\r\n\r\n").encode() + jpg + b"\r\n"


def _stream(n: int, total: int | None = None, headers: bool = True) -> bytes:
    return b"".join(_part(i, total or n, _jpeg(10 + i), headers) for i in range(n))


@pytest.mark.parametrize("chunk", [7, 1000, 10**6])
def test_frames_and_measurements_are_split_out_whatever_the_chunking(chunk):
    data, rec = _stream(40), StreamRecorder()
    for k in range(0, len(data), chunk):
        rec.feed(data[k:k + chunk])
    rec.close()
    assert rec.frames == 40 and len(rec.jpegs) == 20  # every other frame is kept for the video
    assert rec.jpegs[0] == _jpeg(10) and rec.jpegs[1] == _jpeg(12)
    assert len(rec.ms) == 40 and rec.err == [0.5] * 40 and rec.grip == [0.25] * 40
    assert rec.complete and rec.total == 40 and rec.instruction == "pull a block"


def test_an_older_server_without_part_headers_still_gives_the_video():
    rec = StreamRecorder()
    rec.feed(_stream(5, headers=False))
    assert rec.frames == 4  # the last part ends only when the stream does
    rec.close()
    assert rec.frames == 5 and rec.ms == [] and not rec.complete


def test_a_session_the_viewer_left_is_partial():
    rec = StreamRecorder()
    rec.feed(_stream(40, total=200))
    assert rec.frames == 40 and not rec.complete


@pytest.fixture
def upstream():
    """A stand-in for serve_policy.py: policies.json and one held-out episode's MJPEG stream."""
    info = {"gpu": "NVIDIA H200", "policies": {"jenga": {"checkpoint": "jenga_010000", "stage": "jenga",
                                                         "held_out": [58]}}}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/policies.json":
                body = json.dumps(info).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.wfile.write(_stream(40))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def client(tmp_path, monkeypatch, upstream):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LIVE_POLICY_URL", upstream)
    monkeypatch.setenv("FUTURAMA_PROJECTS_FILE", str(tmp_path / "projects.json"))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY", "FUTURAMA_SSH_HOST"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None)
    with TestClient(app) as c:
        yield c, tmp_path / "projects.json"
    settings.cache_clear()
    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None)


def test_a_streamed_session_shows_up_in_its_project(client):
    c, mapping = client
    pid = c.post("/api/projects", json={"task_prompt": "Pull a block out of a Jenga tower"}).json()["id"]
    mapping.write_text(json.dumps({"jenga": pid}))
    r = c.get("/api/live-policy/stream/jenga/58")
    assert r.status_code == 200 and r.content.count(b"--frame") == 40  # the browser gets the stream unchanged

    rows = []
    for _ in range(100):  # saved in the background after the stream ends
        rows = c.get(f"/api/projects/{pid}/replays").json()
        if rows:
            break
        time.sleep(0.1)
    assert len(rows) == 1
    rp = rows[0]
    assert rp["kind"] == "live_inference" and rp["status"] == "complete" and rp["gpu"] == "NVIDIA H200"
    assert rp["checkpoint_name"] == "jenga_010000" and rp["checkpoint_id"] is None  # not registered here
    ep = rp["episodes"][0]
    assert ep["episode"] == 58 and ep["frames"] == 40 and ep["joint_mae_deg"] == 0.5
    assert ep["inference_ms"]["mean"] == pytest.approx(np.mean([40 + i % 3 for i in range(40)]), abs=0.01)
    assert ep["video_key"].endswith("ep58.mp4") and ep["instruction"] == "pull a block"
    rollouts = [x for x in c.get(f"/api/projects/{pid}/rollouts").json() if x["kind"] == "replay"]
    assert rollouts[0]["replay_kind"] == "live_inference" and rollouts[0]["success"] is None


def test_a_task_without_a_project_is_streamed_but_not_saved(client):
    c, mapping = client
    mapping.write_text("{}")
    assert c.get("/api/live-policy/stream/jenga/58").status_code == 200
