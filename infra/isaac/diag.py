"""Isolate Isaac Sim startup: SimulationApp alone, with optional extra Kit settings."""
import sys

from isaacsim import SimulationApp

extra = sys.argv[1:]
cfg = {"headless": True, "extra_args": extra}
print("[diag] starting", extra, flush=True)
app = SimulationApp(cfg)
print("[diag] app up", flush=True)
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

cam = rep.create.camera(position=(1, 1, 1), look_at=(0, 0, 0))
rep.create.cube(position=(0, 0, 0), scale=0.3)
rp = rep.create.render_product(cam, (320, 240))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
for _ in range(5):
    rep.orchestrator.step()
img = np.asarray(ann.get_data())
print("[diag] rendered", img.shape, float(img[..., :3].mean()), flush=True)
app.close()
