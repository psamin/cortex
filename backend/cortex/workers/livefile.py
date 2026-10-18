"""Best-effort live preview frames (live.jpg) that a reader may have open.

On Windows, replacing a file another process is reading (the studio's MJPEG/JPEG endpoint) raises
PermissionError (WinError 5/32). A live preview is a convenience, so a frame that can't be swapped in
is retried briefly and then skipped. It must never fail the recording or collection it rides on.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def write_live_jpg(frame: np.ndarray, out_dir: Path, name: str = "live.jpg", quality: int = 80) -> bool:
    from PIL import Image

    # per process: two runs in one project (e.g. a recording and a collection) share the folder, and a shared
    # temp name let one swap away the other's file mid-write ("No such file or directory" failed a recording)
    tmp = out_dir / f"{name}.{os.getpid()}.tmp"
    try:
        Image.fromarray(frame).save(tmp, format="JPEG", quality=quality)
    except OSError:
        return False
    for delay in (0.0, 0.02, 0.05, 0.1):
        if delay:
            time.sleep(delay)
        try:
            os.replace(tmp, out_dir / name)
            return True
        except PermissionError:
            continue
        except OSError:  # e.g. the folder was cleared for a new run
            break
    try:
        tmp.unlink()
    except OSError:
        pass
    return False
