"""Live GPU sessions are saved to the task's project when the stream ends (#179)."""

import io
import json
import re
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


def _sim_stream(n: int, succeed_at: int, seed: int = 7) -> bytes:
    """serve_policy.py's /sim/<task>/<seed> parts: step/max_steps, inference ms, the simulator's verdict."""
    out = b""
    for i in range(n):
        jpg = _jpeg(10 + i)
        out += (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\nX-Frame: {i}/600\r\n"
                f"X-Inference-Ms: 20\r\nX-Success: {int(i >= succeed_at)}\r\nX-Seed: {seed}\r\n"
                f"X-Policy: act@030000\r\nX-Instruction: push%20the%20block\r\n\r\n").encode() + jpg + b"\r\n"
    return out


def test_a_sim_stream_gives_the_verdict_steps_and_policy():
    rec = StreamRecorder()
    rec.feed(_sim_stream(45, succeed_at=30))
    assert rec.success is True and rec.last_index == 44 and rec.total == 600 and rec.policy == "act@030000"
    assert rec.complete  # the simulator's verdict is in: the server stops soon after success, before max_steps


def test_a_run_cut_off_before_its_end_is_partial_even_if_the_connection_closed_cleanly():
    """A dropped tunnel, Stop GPU or a crashed server close the connection like a finished stream."""
    rec = StreamRecorder()
    rec.feed(_sim_stream(25, succeed_at=30))  # still running at step 24 of 600
    rec.finished = True
    assert not rec.complete
    rec = StreamRecorder()
    rec.feed(_stream(40, total=200))
    rec.finished = True
    assert not rec.complete


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
                                                         "held_out": [58]}},
            "sim": {"jenga": {"variant": "act", "step": "030000", "success_rate": 0.9, "instruction": "push"}}}

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
            self.wfile.write(_sim_stream(45, succeed_at=30) if self.path.startswith("/sim/") else _stream(40))

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


def _replays(c, pid):
    rows = []
    for _ in range(100):  # saved in the background after the stream ends
        rows = c.get(f"/api/projects/{pid}/replays").json()
        if rows:
            break
        time.sleep(0.1)
    return rows


def test_a_live_sim_run_is_saved_with_the_simulators_verdict(client):
    from cortex.api import app as app_mod
    from cortex.models import Checkpoint

    c, mapping = client
    pid = c.post("/api/projects", json={"task_prompt": "Push a block out of a Jenga tower (sim)"}).json()["id"]
    mapping.write_text(json.dumps({"sim_jenga": pid}))
    ck = Checkpoint(project_id=pid, training_job_id=None, policy_type="act", step=30000,
                    path="futurama:~/cortex-sim/jenga/runs/act/checkpoints/030000/pretrained_model")
    c.portal.call(app_mod.CTX.repo.save, "checkpoints", ck)
    r = c.get("/api/live-policy/sim/jenga/7")
    assert r.status_code == 200 and r.content.count(b"--frame") == 45

    rows = _replays(c, pid)
    assert len(rows) == 1
    rp, ep = rows[0], rows[0]["episodes"][0]
    assert rp["kind"] == "live_sim" and rp["status"] == "complete" and rp["checkpoint_id"] == ck.id
    assert rp["checkpoint_name"] == "act@030000" and "judged by the simulator" in rp["note"]
    assert ep["episode"] == 7 and ep["success"] is True and ep["steps"] == 45 and ep["inference_ms"]["mean"] == 20
    assert ep["joint_mae_deg"] is None and ep["video_key"].endswith("seed7.mp4")
    row = [x for x in c.get(f"/api/projects/{pid}/rollouts").json() if x["kind"] == "replay"][0]
    assert row["replay_kind"] == "live_sim" and row["success"] is True and row["seed"] == 7
    assert row["label"] == "Live sim · seed 7" and row["metric"] == "45 steps · 20 ms/frame"
    st = c.get(f"/api/projects/{pid}/state").json()
    assert st["evaluations"] == [] and st["replays"][0]["success"] is True  # one run is not an evaluation


def test_a_live_sim_run_without_a_project_is_kept_as_a_physical_skills_replay(client):
    """Served sim skills (e.g. unscrew) have no project: their live runs are kept in skill_replays (#319)."""
    import av

    c, mapping = client
    mapping.write_text("{}")
    assert c.get("/api/live-policy/sim/jenga/7").status_code == 200

    reps = []
    for _ in range(100):  # saved in the background after the stream ends
        reps = next(s for s in c.get("/api/skills").json() if s["task"] == "jenga")["replays"]
        if reps:
            break
        time.sleep(0.1)
    assert len(reps) == 1 and re.fullmatch(r"/api/skills/jenga/replays/live_\d{8}-\d{6}_seed7\.mp4", reps[0])
    r = c.get(reps[0])
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    with av.open(io.BytesIO(r.content)) as v:
        assert sum(1 for _ in v.decode(video=0)) == 23  # 45 frames streamed, every other kept
    from cortex.config import settings

    assert [f.name for f in (settings().data_dir / "skill_replays" / "jenga").iterdir()] == [reps[0].split("/")[-1]]


def test_skill_replays_keep_the_newest_live_runs_and_never_the_curated_ones(tmp_path):
    import os

    import av

    from cortex.services.live_recording import KEEP_LIVE, keep_skill_replay

    d = tmp_path / "skill_replays" / "unscrew"
    d.mkdir(parents=True)
    (d / "unscrew_live_00.mp4").write_bytes(b"curated")
    os.utime(d / "unscrew_live_00.mp4", (1, 1))  # older than every live run
    for i in range(KEEP_LIVE + 2):
        (d / f"live_2026010{i // 10}-0000{i % 10:02d}_seed{i}.mp4").write_bytes(b"old")
        os.utime(d / f"live_2026010{i // 10}-0000{i % 10:02d}_seed{i}.mp4", (100 + i, 100 + i))
    jpgs = []
    for v in range(6):  # the GPU server's 960x540 frames
        buf = io.BytesIO()
        Image.fromarray(np.full((540, 960, 3), 10 * v, np.uint8)).save(buf, format="JPEG")
        jpgs.append(buf.getvalue())

    p = keep_skill_replay(tmp_path, "unscrew", 42, jpgs)
    assert p.parent == d and re.fullmatch(r"live_\d{8}-\d{6}_seed42\.mp4", p.name)
    with av.open(str(p)) as v:
        assert (v.streams.video[0].width, v.streams.video[0].height) == (960, 540)
    live = sorted(f.name for f in d.glob("live_*.mp4"))
    assert len(live) == KEEP_LIVE and p.name in live
    assert "live_20260100-000000_seed0.mp4" not in live and "live_20260100-000001_seed1.mp4" not in live
    assert (d / "unscrew_live_00.mp4").read_bytes() == b"curated"
    assert not [f for f in d.iterdir() if f.name.startswith(".")]  # no half-written file left
    with pytest.raises(ValueError):
        keep_skill_replay(tmp_path, "../x", 1, jpgs)
