"""Presentation camera for the videos and live view people watch.

Purely cosmetic and never fed to policies: training observations stay the
EnvHub env's 96x128 front/wrist images with the scene's own lighting. For the
video only, this camera renders at 1280x960 with 8x MSAA, smoother geometry and sharper shadows
from a three-quarter view, temporarily softens the fill light for contrast
(restored after every frame), then applies a light colour grade and partial
exposure compensation so dim-lighting episodes stay watchable while still
looking dimmer (the condition is labelled in the UI).
"""

from __future__ import annotations

import mujoco
import numpy as np

VIEW_H, VIEW_W = 960, 1280


class DemoCamera:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # MSAA/shadow sizes are read when a GL context is created: set them only for this
        # renderer and restore them, so observation renderers keep their own settings.
        # The offscreen buffer must also be at least the render size.
        q, g = model.vis.quality, model.vis.global_
        saved = (q.offsamples, q.shadowsize, q.numslices, q.numstacks, g.offwidth, g.offheight)
        q.offsamples, q.shadowsize, q.numslices, q.numstacks = 8, 8192, 64, 64
        g.offwidth, g.offheight = max(g.offwidth, VIEW_W), max(g.offheight, VIEW_H)
        try:
            self.renderer = mujoco.Renderer(model, VIEW_H, VIEW_W)
        finally:
            q.offsamples, q.shadowsize, q.numslices, q.numstacks, g.offwidth, g.offheight = saved
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (0.21, 0.04, 0.09)
        self.cam.distance = 0.52
        self.cam.azimuth = 140
        self.cam.elevation = -24
        yy, xx = np.mgrid[0:VIEW_H, 0:VIEW_W]
        r = np.sqrt(((xx - VIEW_W / 2) / (VIEW_W / 2)) ** 2 + ((yy - VIEW_H / 2) / (VIEW_H / 2)) ** 2)
        self.vignette = (1 - 0.22 * np.clip(r - 0.35, 0, 1) ** 1.5)[..., None].astype(np.float32)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        m = self.model
        saved = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy(), m.light_diffuse.copy())
        try:
            m.vis.headlight.diffuse[:] = saved[0] * 0.75
            m.vis.headlight.ambient[:] = saved[1] * 0.55
            if m.nlight:
                m.light_diffuse[0] = saved[2][0] * 0.85
            self.renderer.update_scene(data, camera=self.cam)
            img = self.renderer.render()
        finally:
            m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:], m.light_diffuse[:] = saved
        return grade(img, self.vignette)

    def close(self) -> None:
        self.renderer.close()


def grade(img: np.ndarray, vignette: np.ndarray | None = None, sat: float = 1.18, con: float = 1.10) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    luma = float(x.mean())
    if luma > 1e-3:  # partial auto-exposure: dim stays dimmer, but readable
        x *= float(np.clip(0.42 / luma, 0.85, 2.4)) ** 0.6
    g = x.mean(axis=2, keepdims=True)
    x = g + (x - g) * sat
    x = (x - 0.5) * con + 0.5
    if vignette is not None:
        x *= vignette
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
