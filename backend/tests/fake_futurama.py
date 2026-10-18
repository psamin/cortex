"""A stand-in for Futurama, for the live-inference end-to-end tests (test_live_e2e.py).

`install(dir)` writes executables `ssh`, `squeue`, `sbatch` and `scancel` into <dir>/bin. Put that on PATH
and live_policy_service runs unchanged against it:

  ssh <host> "<cmd>"   runs <cmd> in a real shell with HOME=<dir>/home (holding cortex-jenga/), so the
                       service's own pipelines (cd, grep, tail, cat, rm) run for real; prints the GT banner
                       on stderr like the real login node
  ssh -G <host>        prints the ssh config (user pi34)
  ssh -N -L l:127.0.0.1:r ... <node>   a tunnel: forwards 127.0.0.1:l to 127.0.0.1:r (the fake GPU node)
  squeue / sbatch / scancel            a Slurm whose jobs follow a timeline: each entry is
                       [seconds after submit, "%T %R" line or None (the job left the queue), log line or None]

Everything is driven by <dir>/state.json (see DEFAULT). `FakeServe` mimics infra/futurama/serve_policy.py:
/policies.json, /live/<task>/<ep> and /sim/<task>/<seed> as MJPEG with the same part headers, one stream
at a time (the real server's LOCK, taken after the response headers are sent).
"""

from __future__ import annotations

import asyncio
import fcntl
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

BANNER = "  " + "*" * 40 + "\n  *  Georgia Institute of Technology - Terms of Use  *\n  " + "*" * 40 + "\n"
NO_VPN = "ssh: connect to host planetexpress-login.cc.gatech.edu port 22: Operation timed out\n"


def timeline(port: int, tasks: str = "['jenga']", sims: str = "['jenga']") -> list:
    """serve_policy.sbatch's progress, compressed to a few seconds (each step outlasts a poll)."""
    return [[0.0, "PENDING (Resources)", None],
            [0.6, "RUNNING zapp", "[live] Installing LeRobot on zapp (NVIDIA L40S)"],
            [1.2, "RUNNING zapp", "[live] Preparing held-out episodes: jenga"],
            [1.8, "RUNNING zapp", "[live] Loading the policies onto the GPU"],
            [2.4, "RUNNING zapp", f"[serve] READY port={port} policies={tasks} sim={sims}"]]


DEFAULT = {"next_id": 10700, "jobs": {}, "unreachable": False, "hang_s": 0, "sbatch_error": None,
           "tunnel_fails": False, "serve_port": 0, "timeline": []}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ state
class Cluster:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.file = self.root / "state.json"
        self.home = self.root / "home"

    @contextmanager
    def edit(self):
        with open(self.root / "state.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            st = json.loads(self.file.read_text())
            yield st
            self.file.write_text(json.dumps(st))

    def read(self) -> dict:
        return json.loads(self.file.read_text())

    def set(self, **kw) -> None:
        with self.edit() as st:
            st.update(kw)

    def tunnels(self) -> list[dict]:
        """Every `ssh -N` started: its pid and forward."""
        f = self.root / "tunnels.jsonl"
        return [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []

    def calls(self) -> list[list[str]]:
        f = self.root / "calls.jsonl"
        return [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []

    def add_job(self, name: str = "cortex-serve", tl: list | None = None, age: float = 0.0) -> str:
        """A job already in the queue (e.g. submitted before a backend restart), `age` seconds old."""
        with self.edit() as st:
            jid = str(st["next_id"])
            st["next_id"] += 1
            st["jobs"][jid] = {"name": name, "dir": str(self.home / "cortex-jenga"), "submitted": time.time() - age,
                               "timeline": tl if tl is not None else st["timeline"], "gone": False, "argv": []}
        return jid


def _advance(st: dict) -> None:
    """Bring every job to now: its squeue line, its Slurm log and live/server.json, as the sbatch writes them."""
    now = time.time()
    for jid, j in st["jobs"].items():
        if j["gone"]:
            continue
        seen = [e for e in j["timeline"] if now - j["submitted"] >= e[0]]
        if seen and seen[-1][1] is None:
            j["gone"] = True
        j["line"] = seen[-1][1] if seen else "PENDING (Priority)"
        logs = [e[2] for e in seen if e[2]]
        d = Path(j["dir"])
        if logs:
            (d / f"slurm-{j['name']}-{jid}.out").write_text("".join(f"{x}\n" for x in logs))
        if any("Loading the policies" in x or "READY" in x for x in logs):
            (d / "live").mkdir(exist_ok=True)
            (d / "live" / "server.json").write_text(
                json.dumps({"node": "zapp", "port": st["serve_port"], "job": int(jid)}))


def _slurm(cluster: Cluster, prog: str, args: list[str]) -> int:
    with cluster.edit() as st:
        _advance(st)
        jobs = st["jobs"]
        if prog == "squeue":
            if "-j" in args:
                jid = args[args.index("-j") + 1]
                if jid not in jobs or jobs[jid]["gone"]:
                    sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
                    return 1
                print(jobs[jid]["line"])
                return 0
            name = args[args.index("-n") + 1] if "-n" in args else None
            for jid, j in jobs.items():
                if not j["gone"] and (name is None or j["name"] == name):
                    print(jid)
            return 0
        if prog == "sbatch":
            if st["sbatch_error"]:
                sys.stderr.write(st["sbatch_error"] + "\n")
                return 1
            jid = str(st["next_id"])
            st["next_id"] += 1
            jobs[jid] = {"name": args[args.index("-J") + 1] if "-J" in args else "sbatch", "dir": os.getcwd(),
                         "submitted": time.time(), "timeline": st["timeline"], "gone": False, "argv": args}
            print(jid)
            return 0
        if prog == "scancel":
            name = args[args.index("-n") + 1] if "-n" in args else None
            for jid, j in jobs.items():
                if jid in args or (name and j["name"] == name):
                    j["gone"] = True
            return 0
    return 2


# ------------------------------------------------------------------ ssh
def _forward(local: int, remote: int) -> int:
    async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while data := await r.read(65536):
                w.write(data)
                await w.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            w.close()

    async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            r2, w2 = await asyncio.open_connection("127.0.0.1", remote)
        except OSError:
            w.close()
            return
        await asyncio.gather(pipe(r, w2), pipe(r2, w))

    async def main() -> int:
        try:
            srv = await asyncio.start_server(handle, "127.0.0.1", local)
        except OSError:
            sys.stderr.write(f"bind [127.0.0.1]:{local}: Address already in use\n"
                             f"channel_setup_fwd_listener_tcpip: cannot listen to port: {local}\n"
                             "Could not request local forwarding.\n")
            return 255
        async with srv:
            await srv.serve_forever()
        return 0

    return asyncio.run(main())


def _ssh(cluster: Cluster, args: list[str]) -> int:
    with open(cluster.root / "calls.jsonl", "a") as f:
        f.write(json.dumps(["ssh", *args]) + "\n")
    st = cluster.read()
    if "-G" in args:
        print("user pi34\nhostname planetexpress-login.cc.gatech.edu\nport 22")
        return 0
    if st["hang_s"]:
        time.sleep(st["hang_s"])
    if st["unreachable"]:
        sys.stderr.write(NO_VPN)
        return 255
    opts, rest, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "-J", "-L", "-p", "-i", "-l", "-F"):
            opts.setdefault(a, []).append(args[i + 1])
            i += 2
        elif a.startswith("-") and not rest:
            opts[a] = True
            i += 1
        else:
            rest.append(a)
            i += 1
    if "-N" in opts:
        if st["tunnel_fails"]:
            sys.stderr.write("channel 0: open failed: connect failed: No route to host\n")
            return 255
        local, _, remote = opts["-L"][0].split(":")
        with open(cluster.root / "tunnels.jsonl", "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "local": int(local), "remote": int(remote)}) + "\n")
        return _forward(int(local), int(remote))
    sys.stderr.write(BANNER)
    env = {**os.environ, "HOME": str(cluster.home), "USER": "pi34"}
    return subprocess.run(["/bin/sh", "-c", " ".join(rest[1:])], cwd=cluster.home, env=env).returncode


def install(root: Path, serve_port: int, tl: list | None = None) -> Cluster:
    """Write the fake cluster into `root`; returns its handle. Put <root>/bin first on PATH."""
    root = Path(root)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "home" / "cortex-jenga" / "live").mkdir(parents=True, exist_ok=True)
    for prog in ("ssh", "squeue", "sbatch", "scancel"):
        exe = root / "bin" / prog
        exe.write_text(f"#!/bin/sh\nexec {sys.executable} {Path(__file__).resolve()} {root} {prog} \"$@\"\n")
        exe.chmod(0o755)
    (root / "state.json").write_text(json.dumps({**DEFAULT, "serve_port": serve_port,
                                                  "timeline": tl if tl is not None else timeline(serve_port)}))
    return Cluster(root)


# ------------------------------------------------------------------ serve_policy.py
def jpeg(v: int, size: tuple[int, int] = (64, 48)) -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), v % 256, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeServe:
    """serve_policy.py: policies.json, held-out replays and sim runs as MJPEG with the real part headers."""

    def __init__(self, port: int = 0, gpu: str = "NVIDIA L40S"):
        self.info = {"gpu": gpu, "policies": {"jenga": {"checkpoint": "jenga_010000", "stage": "jenga",
                                                        "held_out": [58, 59]}},
                     "sim": {"jenga": {"variant": "act", "step": 30000, "success_rate": 0.9, "eval_episodes": 50,
                                       "eval_seeds": "100000..100049", "instruction": "push the marked block out"}}}
        self.frames = {"live": 40, "sim": 45}  # parts per stream
        self.max_steps = 600  # X-Frame's total for sim streams (env.max_steps)
        self.succeed_at = 30  # sim: X-Success turns 1 at this step
        self.fps = 0.0  # 0: as fast as possible
        self.drop_after: int | None = None  # the connection dies after this many parts (a crash, a dead tunnel)
        self.lock = threading.Lock()  # one GPU, one stream at a time (taken after the headers, as in serve_policy)
        self.started: list[str] = []  # streams that got the lock, in order
        self.ended: list[tuple[str, int, str]] = []  # (path, parts sent, why)
        self._jpgs = [jpeg(10 + i) for i in range(8)]
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                parts = [x for x in self.path.split("?")[0].split("/") if x]
                if parts == ["policies.json"]:
                    body = json.dumps(fake.info).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if len(parts) == 3 and parts[0] == "sim" and parts[1] in fake.info["sim"] and parts[2].isdigit():
                    return fake._stream(self, "sim", parts[1], int(parts[2]))
                if len(parts) == 3 and parts[0] == "live" and parts[1] in fake.info["policies"]:
                    return fake._stream(self, "live", parts[1], int(parts[2]))
                self.send_error(404)

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def _stream(self, h: BaseHTTPRequestHandler, kind: str, task: str, n: int) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        h.end_headers()
        h.wfile.flush()
        sent, why = 0, "finished"
        with self.lock:
            self.started.append(h.path)
            total = self.frames[kind]
            for i in range(total):
                if self.drop_after is not None and i >= self.drop_after:
                    why = "dropped"
                    h.connection.shutdown(socket.SHUT_RDWR)
                    break
                jpg = self._jpgs[i % len(self._jpgs)]
                if kind == "sim":
                    head = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n"
                            f"X-Frame: {i}/{self.max_steps}\r\nX-Inference-Ms: 20.00\r\n"
                            f"X-Success: {int(i >= self.succeed_at)}\r\nX-Seed: {n}\r\n"
                            f"X-Policy: {self.info['sim'][task]['variant']}@{self.info['sim'][task]['step']}\r\n"
                            f"X-Instruction: {quote(self.info['sim'][task]['instruction'])}\r\n\r\n")
                else:
                    head = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\nX-Frame: {i}/{total}\r\n"
                            f"X-Inference-Ms: {40 + i % 3:.2f}\r\nX-Joint-Err-Deg: 0.5000\r\nX-Gripper-Err: 0.2500\r\n"
                            f"X-Instruction: {quote('pull a block')}\r\n\r\n")
                try:
                    h.wfile.write(head.encode() + jpg + b"\r\n")
                    h.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    why = "viewer left"
                    break
                sent += 1
                if self.fps:
                    time.sleep(1 / self.fps)
        self.ended.append((h.path, sent, why))

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


if __name__ == "__main__":  # the executables in <root>/bin
    _root, _prog, *_args = sys.argv[1:]
    _c = Cluster(Path(_root))
    sys.exit(_ssh(_c, _args) if _prog == "ssh" else _slurm(_c, _prog, _args))
