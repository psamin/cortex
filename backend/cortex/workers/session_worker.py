"""Interactive manipulation session: one MuJoCo episode driven by an external agent.

    python -m cortex.workers.session_worker

Long-lived process speaking JSON lines on stdin/stdout, one response per request:
  {"op":"reset","seed":..,"scenario":{..},"mode":"camera","live_dir":".."}
  {"op":"call","tool":"observe_scene","args":{..}}    -> {"result":{..},"image_b64":"..."?}
  {"op":"finish","out_dir":"..","meta":{..}}          -> {"success":bool,"num_frames":..}
  {"op":"close"}
Used when the user's own agent (Claude Code over MCP) plans the actions: the same
validated tool API, IK execution, recording and simulator-judged success as the
autonomous worker, with the planning loop living outside this process.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..sim.manipulation import ManipulationSession, annotate_overhead
from ..sim.so101 import PickPlaceEnv, Variation, configure_gl
from ..sim.viewer import DemoCamera
from .agent_worker import live_writer, save_attempt


def reply(**kw: Any) -> None:
    sys.stdout.write(json.dumps(kw, default=str) + "\n")
    sys.stdout.flush()


def jpeg_b64(img) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    configure_gl()
    env = PickPlaceEnv(episode_steps=10_000)
    session: ManipulationSession | None = None
    camera: DemoCamera | None = None
    trace: list[dict[str, Any]] = []
    t0 = time.time()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "reset":
                sc = req["scenario"]
                var = Variation(cube_region=sc["cube_region"], lighting=sc["lighting"],
                                cube_offset=tuple(sc.get("cube_offset", (0, 0))),
                                box_offset=tuple(sc.get("box_offset", (0, 0))))
                env.reset(req["seed"], var)
                if camera is None:
                    camera = DemoCamera(env.world.model)
                session = ManipulationSession(env, mode=req.get("mode", "camera"),
                                              on_frame=live_writer(Path(req["live_dir"])), video_renderer=camera)
                session.reset_environment(req["seed"], var)
                trace, t0 = [], time.time()
                live_writer(Path(req["live_dir"]))(camera.render(env.world.data))
                reply(ok=True, seed=req["seed"])
            elif op == "call":
                assert session is not None, "reset first"
                name, args = req["tool"], req.get("args") or {}
                result, img = session.call(name, args)
                out: dict[str, Any] = {"result": result}
                if img is not None:
                    shown = annotate_overhead(img) if session.last_overhead is img else img
                    out["image_b64"] = jpeg_b64(shown)
                    out["image_kind"] = "overhead (pixel grid every 40 px)" if session.last_overhead is img else name
                trace.append({"i": len(trace), "tool": name, "args": args, "ok": bool(result.get("ok")),
                              "result": {k: v for k, v in result.items() if k != "wall_ms"},
                              "sim_step": len(session.rec)})
                reply(**out, holding=session.holding, sim_steps=len(session.rec))
            elif op == "finish":
                assert session is not None, "reset first"
                success = bool(env.in_box())
                meta = {**(req.get("meta") or {}), "trace": trace, "success": success,
                        "wall_s": round(time.time() - t0, 2), "max_cube_z": env.max_cube_z,
                        "counters": dict(session.counters)}
                info = save_attempt(session, env, Path(req["out_dir"]), meta)
                reply(ok=True, success=success, trace_len=len(trace), counters=dict(session.counters),
                      wall_s=meta["wall_s"], **info)
                session = None
            elif op == "close":
                reply(ok=True)
                break
            else:
                reply(ok=False, error=f"unknown op {op}")
        except Exception as e:  # report and keep serving
            reply(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])
    env.close()


if __name__ == "__main__":
    main()
