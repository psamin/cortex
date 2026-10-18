"""Copy the light parts of a Futurama result folder (metrics, videos) to this machine (#179).

A source is a local directory or an rsync remote such as ``futurama:~/cortex-jenga/replay/jenga``.
Only files matching the include patterns are copied, so checkpoints never come along by accident.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def is_remote(src: str) -> bool:
    """host:path (rsync/scp syntax), as opposed to a local path (including C:\\ on Windows)."""
    head = src.replace("\\", "/").split("/", 1)[0]
    return ":" in head and not re.fullmatch(r"[A-Za-z]:", head) and not Path(src).exists()


def fetch(src: str, dst: Path, include: list[str], exclude: tuple[str, ...] = (), recursive: bool = True) -> Path:
    """rsync the files matching `include` (rsync patterns, e.g. "*.mp4") from src into dst; returns dst.
    `exclude` patterns win over `include` (e.g. "dataset/videos/"); recursive=False stays in the top folder."""
    dst.mkdir(parents=True, exist_ok=True)
    pats = [*(f"--exclude={p}" for p in exclude), *(["--include=*/"] if recursive else []),
            *(f"--include={p}" for p in include), "--exclude=*"]
    r = subprocess.run(["rsync", "-a", "-m", *pats, src.rstrip("/") + "/", str(dst) + "/"],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        tail = "\n".join(ln for ln in r.stderr.splitlines() if ln.strip() and not ln.lstrip().startswith("*"))
        raise RuntimeError(f"rsync from {src} failed ({r.returncode}): {tail[-400:]}"
                           + (" Is the GT VPN on?" if is_remote(src) else ""))
    return dst


def remote_size_bytes(src: str) -> int | None:
    """Total size of a remote directory (du -sb over ssh), or None when it can't be measured."""
    host, path = src.split(":", 1)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, f"du -sb {path}"],
                       capture_output=True, text=True, timeout=120)
    for ln in r.stdout.splitlines():
        first = ln.split("\t", 1)[0].strip()
        if first.isdigit():
            return int(first)
    return None


def local_label(src: str) -> str:
    """A filesystem-safe name for a source, used for its working copy under the data dir."""
    tail = src.split(":", 1)[-1].strip("/").replace("~", "home")
    return "_".join(p for p in tail.split("/")[-3:] if p) or "import"
