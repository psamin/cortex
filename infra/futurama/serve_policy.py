"""Live GPU inference for trained SmolVLA policies (run inside a GPU job).

    python serve_policy.py <job_dir> <data_root> <port> [<sim_dir>]

Loads the newest checkpoint per task from <job_dir>/output/<task>/ and that task's held-out episodes
(prepared by prepare_dataset.py under <data_root>/<task>). Serves:
  GET  /                      page listing policies and held-out episodes
  GET  /policies.json         the same as JSON (GPU name, checkpoint, held-out episodes per task)
  GET  /live/<task>/<ep>      MJPEG: the policy runs on the GPU frame by frame on a held-out episode it
                              never trained on; each frame shows the camera, its joint targets vs the
                              human's, and the measured inference time. Each part also carries those
                              numbers as headers (X-Inference-Ms, X-Joint-Err-Deg, X-Gripper-Err, X-Frame)
                              so the studio can save the session
  POST /act                   {task, state:[6], images:{camera: base64 JPEG}} -> {action:[6], ms}
                              for a real SO-101 client
  GET  /sim/<task>/<seed>     MJPEG: a sim-trained policy runs closed loop in MuJoCo on the GPU, on a
                              layout it never saw (serve_sim.py); listed under "sim" in /policies.json
"""

import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from PIL import Image, ImageDraw

CAMS = {"jenga": ["main", "secondary_0", "secondary_1"], "conveyor": ["top", "wrist"],
        "surgical": ["front", "side"], "needle": ["external", "wrist"]}
STAGE = {"jenga": "jenga", "conveyor": "conveyor", "suture": "needle"}
JOINTS = ["pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper"]
LOCK = threading.Lock()  # one GPU, one stream at a time

job_dir, data_root, port = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
POLICIES = {}
for task, stage in STAGE.items():
    cks = sorted((c for c in (job_dir / "output" / task).glob("*_[0-9]*") if (c / "config.json").exists()),
                 key=lambda p: p.stat().st_mtime)  # skip half-copied checkpoints
    if not cks or not (data_root / stage).exists():
        continue
    ck = cks[-1]
    if task == "suture" and not ck.name.startswith("needle"):
        stage = "surgical"  # only the first stage finished: serve that one honestly
    policy = get_policy_class(json.loads((ck / "config.json").read_text())["type"]).from_pretrained(str(ck))
    policy = policy.to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ck),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    split = json.loads((data_root / "split.json").read_text())[stage]
    POLICIES[task] = {"policy": policy, "pre": pre, "post": post, "stage": stage, "checkpoint": ck.name,
                      "held_out": split["held_out"],
                      "rename": {f"observation.images.{c}": f"observation.images.camera{i + 1}"
                                 for i, c in enumerate(CAMS[stage])}}
    print(f"[serve] {task}: {ck.name} on {stage}, held-out episodes {split['held_out']}", flush=True)


SIM: dict = {}
if len(sys.argv) > 4 and Path(sys.argv[4]).is_dir():
    import serve_sim

    SIM = serve_sim.load(Path(sys.argv[4]))


def act(p: dict, obs: dict, task_text: str) -> tuple[np.ndarray, float]:
    batch = {p["rename"].get(k, k): (v[None] if torch.is_tensor(v) else v) for k, v in obs.items()}
    batch["task"] = [task_text]
    t0 = time.perf_counter()
    with torch.inference_mode():
        a = p["post"](p["policy"].select_action(p["pre"](batch)))
    torch.cuda.synchronize()
    return a.squeeze(0).float().cpu().numpy(), (time.perf_counter() - t0) * 1000


def frame(img: np.ndarray, pred: np.ndarray, human: np.ndarray, ms: float, task: str, i: int, n: int) -> bytes:
    cam = Image.fromarray(img).resize((640, 480))
    canvas = Image.new("RGB", (1000, 480), (18, 18, 18))
    canvas.paste(cam, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((650, 12), f"{task}  live on GPU", fill=(240, 240, 240))
    d.text((650, 30), f"frame {i}/{n}   inference {ms:5.1f} ms", fill=(160, 200, 255))
    for j in range(6):
        y = 70 + j * 62
        d.text((650, y), JOINTS[j], fill=(200, 200, 200))
        for val, color, dy in ((pred[j], (90, 200, 120), 16), (human[j], (230, 160, 70), 32)):
            x = 650 + int(np.clip((val + 110) / 220, 0, 1) * 330)
            d.rectangle([650, y + dy, 980, y + dy + 8], outline=(60, 60, 60))
            d.rectangle([x - 2, y + dy - 1, x + 2, y + dy + 9], fill=color)
    d.text((650, 448), "green: policy   orange: human demo (held out)", fill=(150, 150, 150))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def part_header(jpg: bytes, i: int, n: int, ms: float, err: np.ndarray, instruction: str) -> bytes:
    """One MJPEG part's headers. Browsers read only Content-Type; Cortex records the rest (#179):
    inference time, mean |policy - human| over the 5 arm joints (degrees) and on the gripper."""
    return (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\nX-Frame: {i}/{n}\r\n"
            f"X-Inference-Ms: {ms:.2f}\r\nX-Joint-Err-Deg: {float(err[:5].mean()):.4f}\r\n"
            f"X-Gripper-Err: {float(err[5]):.4f}\r\nX-Instruction: {quote(instruction)}\r\n\r\n").encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        parts = [x for x in self.path.split("?")[0].split("/") if x]
        if parts == ["policies.json"]:
            body = json.dumps({"gpu": torch.cuda.get_device_name(0), "policies": {
                t: {"checkpoint": p["checkpoint"], "stage": p["stage"], "held_out": p["held_out"]}
                for t, p in POLICIES.items()}, "sim": {
                t: {k: p[k] for k in ("variant", "step", "success_rate", "eval_episodes", "eval_seeds", "instruction")}
                for t, p in SIM.items()}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if not parts:
            rows = "".join(f"<li><b>{t}</b> ({p['checkpoint']}): " + " ".join(
                f'<a href="/live/{t}/{e}">episode {e}</a>' for e in p["held_out"]) + "</li>"
                for t, p in POLICIES.items())
            body = f"<html><body style='font-family:sans-serif;background:#111;color:#eee'><h2>Cortex live policies" \
                   f" (GPU: {torch.cuda.get_device_name(0)})</h2><ul>{rows}</ul></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
            return
        if len(parts) == 3 and parts[0] == "sim" and parts[1] in SIM and parts[2].isdigit():
            return self.stream_sim(SIM[parts[1]], int(parts[2]))
        if len(parts) == 3 and parts[0] == "live" and parts[1] in POLICIES:
            return self.stream(POLICIES[parts[1]], parts[1], int(parts[2]))
        self.send_error(404)

    def stream(self, p: dict, task: str, ep: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        ds = LeRobotDataset(f"local/{p['stage']}", root=data_root / p["stage"], episodes=[ep])
        with LOCK:
            p["policy"].reset()
            for i in range(len(ds)):
                t0 = time.perf_counter()
                item = ds[i]
                obs = {k: v for k, v in item.items() if k.startswith("observation.")}
                pred, ms = act(p, obs, item["task"])
                human = item["action"].numpy()
                img = (item[f"observation.images.{CAMS[p['stage']][0]}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                jpg = frame(img, pred, human, ms, task, i, len(ds))
                try:
                    self.wfile.write(part_header(jpg, i, len(ds), ms, np.abs(pred - human), item["task"]) + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))  # real time (30 fps recordings)

    def stream_sim(self, p: dict, seed: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        with LOCK:
            for jpg, _step, _ms, _ok in serve_sim.stream(p, seed):
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

    def do_POST(self):
        if self.path != "/act":
            return self.send_error(404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = POLICIES[req["task"]]
        obs = {"observation.state": torch.tensor(req["state"], dtype=torch.float32)}
        for cam, b64 in req["images"].items():
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.float32) / 255
            obs[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1)
        with LOCK:
            action, ms = act(p, obs, req.get("instruction", ""))
        body = json.dumps({"action": action.tolist(), "ms": round(ms, 1)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


print(f"[serve] READY port={port} policies={list(POLICIES)} sim={list(SIM)}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
