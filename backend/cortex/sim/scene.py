"""Generate a MuJoCo scene from an agent's IntakeSpec, on top of the pinned EnvHub tabletop.

The upstream env takes a ``scene`` path, so Cortex never forks it. We copy its
``so101_tabletop.xml`` and:

  * add each ObjectSpec as a free body ``obj_<name>`` (box / cylinder / sphere primitive);
  * add each PlaceZone as a flat, non-colliding marker ``zone_<name>`` (the target the
    verifier checks against) and each obstacle as a static geom ``obs_<name>``;
  * add each conveyor as a static bed ``belt_<name>`` (``sim/belt.py``; SceneEnv drives it), with the
    objects that start ``on`` it resting on its top;
  * resize and recolour the table, set the instruction text, and place the overhead
    camera at a calibrated pose when the intake has one;
  * park the upstream cubes and box out of sight. The env requires those bodies to exist,
    so they stay in the model, under the floor.

Asset paths resolve through ``<compiler assetdir>`` to the snapshot, so the generated file can
live anywhere (we write it under the project's data dir).
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET  # nosec B405 - our own pinned file
from dataclasses import dataclass, field
from pathlib import Path

from ..intake.spec import Conveyor, IntakeSpec, ObjectSpec
from .belt import Belt
from .grasp import SIDE_MIN_RADIUS, grasp_mode
from .so101 import envhub_module

PARK_Z = -0.70  # under the table top, above the floor at -0.75: out of every camera
FLOOR_Z = -0.75  # upstream floor plane
OBJ_FRICTION = "1.2 0.03 0.003"


@dataclass
class SceneInfo:
    path: Path
    objects: dict[str, str]  # object name -> body name
    zones: dict[str, tuple[tuple[float, float], tuple[float, float]]]  # name -> (center, half_size)
    starts: dict[str, tuple[float, float]]  # object name -> nominal start xy
    heights: dict[str, float]  # object name -> height (m)
    pick: str | None
    place: list[str]
    lighting: str = "normal"
    shapes: dict[str, str] = field(default_factory=dict)  # object name -> box | cylinder | sphere
    digest: str = ""
    notes: list[str] = field(default_factory=list)
    conveyors: dict[str, dict] = field(default_factory=dict)  # name -> start, end, width, height, speed_mps
    on: dict[str, str] = field(default_factory=dict)  # object name -> the conveyor it starts on

    def belts(self) -> dict[str, Belt]:
        return {n: belt(Conveyor(name=n, **c)) for n, c in self.conveyors.items()}


    def to_dict(self, data_dir: Path) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["path"] = self.path.relative_to(data_dir).as_posix() if self.path.is_relative_to(data_dir) else str(self.path)
        return d

    @classmethod
    def from_dict(cls, d: dict, data_dir: Path) -> SceneInfo:
        d = dict(d)
        p = Path(d.pop("path"))
        d.pop("feasibility", None)
        d.pop("intake_digest", None)
        d["zones"] = {k: (tuple(c), tuple(h)) for k, (c, h) in d["zones"].items()}
        d["starts"] = {k: tuple(v) for k, v in d["starts"].items()}
        return cls(path=p if p.is_absolute() else data_dir / p, **d)


def belt(c: Conveyor) -> Belt:
    return Belt(c.name, start=tuple(c.start), end=tuple(c.end), width=c.width, height=c.height,  # type: ignore[arg-type]
                speed=c.speed_mps)


def belt_start(spec: IntakeSpec, o: ObjectSpec) -> tuple[float, float]:
    """An object on a conveyor without a start sits on the belt's centre line, just inside its upstream end."""
    c = next(c for c in spec.environment.conveyors if c.name == o.on)
    b = belt(c)
    reach = (max(o.size[:2]) if o.shape == "box" else o.size[0]) / 2 + 0.02
    x, y = b.point(min(reach, b.length / 2))
    return (round(float(x), 4), round(float(y), 4))


def body_name(obj: str) -> str:
    return f"obj_{obj}"


def mj_size(o: ObjectSpec) -> str:
    """ObjectSpec.size (full extents / diameters) -> MuJoCo half-size convention."""
    if o.shape == "box":
        return " ".join(f"{s / 2:.4f}" for s in o.size)
    if o.shape == "cylinder":
        return f"{o.size[0] / 2:.4f} {o.size[1] / 2:.4f}"
    return f"{o.size[0] / 2:.4f}"


def default_starts(spec: IntakeSpec) -> dict[str, tuple[float, float]]:
    """Objects without a start go to free spots across the front of the workspace, away from zones."""
    taken = [z.center for z in spec.zones] + [o.start for o in spec.objects if o.start]
    near = [(0.20, -0.04), (0.25, 0.035), (0.17, -0.12), (0.28, -0.08), (0.15, 0.08), (0.30, 0.10)]
    far = [(0.32, -0.02), (0.30, -0.13), (0.31, 0.10), (0.27, -0.20), (0.34, 0.06)]  # side-grasp reach
    out: dict[str, tuple[float, float]] = {}
    for o in spec.objects:
        if o.start:
            out[o.name] = tuple(o.start)  # type: ignore[assignment]
            continue
        if o.on:
            out[o.name] = belt_start(spec, o)
            continue
        cands = far if grasp_mode(o.height) == "side" else near
        assert all((c[0] ** 2 + c[1] ** 2) ** 0.5 >= SIDE_MIN_RADIUS - 0.005 for c in far)
        free = [c for c in cands if all((c[0] - t[0]) ** 2 + (c[1] - t[1]) ** 2 > 0.07 ** 2 for t in taken)]
        pick = (free or cands)[0]
        out[o.name] = pick
        taken.append(pick)
    return out


def build_scene(spec: IntakeSpec, out_dir: Path) -> SceneInfo:
    m = envhub_module()
    root = Path(m.repo_root())
    tree = ET.parse(root / m.SCENE_FILE)  # nosec B314
    mj = tree.getroot()
    mj.set("model", "cortex_scene")
    ET.SubElement(mj, "compiler", assetdir=root.as_posix())
    mj.insert(0, mj[-1])  # compiler first, for readability

    env = spec.environment
    for t in mj.iter("text"):
        if t.get("name") == m.INSTRUCTION_TEXT:
            t.set("data", instruction(spec))
    for mat in mj.iter("material"):
        if mat.get("name") == "table":
            mat.set("rgba", " ".join(f"{c:.3f}" for c in env.table_rgba))
    wb = mj.find("worldbody")
    assert wb is not None
    for g in wb.findall("geom"):
        if g.get("name") == "table":
            hx, hy = env.table_size[0] / 2, env.table_size[1] / 2
            g.set("size", f"{hx:.3f} {hy:.3f} 0.015")
    for b in wb.findall("body"):
        name = b.get("name", "")
        if name in m.CUBE_BODIES:
            i = m.CUBE_BODIES.index(name)
            b.set("pos", f"{0.6 + 0.08 * i:.3f} 1.0 {FLOOR_Z + 0.015:.3f}")  # resting on the floor
        elif name == m.BOX_BODY:
            b.set("pos", f"0.9 1.2 {PARK_Z}")

    starts = default_starts(spec)
    for c in env.conveyors:
        belt(c).add_bed(wb)
    heights = {c.name: c.height for c in env.conveyors}
    for o in spec.objects:
        x, y = starts[o.name]
        z = heights.get(o.on or "", 0.0) + o.height / 2 + 0.0005
        body = ET.SubElement(wb, "body", name=body_name(o.name), pos=f"{x:.4f} {y:.4f} {z:.4f}")
        ET.SubElement(body, "freejoint", name=f"{body_name(o.name)}_free")
        ET.SubElement(body, "geom", name=f"{body_name(o.name)}_geom", type=o.shape, size=mj_size(o),
                      mass=f"{o.mass_kg:.4f}", rgba=" ".join(f"{c:.3f}" for c in o.rgba), condim="4",
                      friction=OBJ_FRICTION, solref="0.01 1")
    for z in spec.zones:
        (cx, cy), (hx, hy) = z.center, z.half_size
        ET.SubElement(wb, "geom", name=f"zone_{z.name}", type="box", pos=f"{cx:.4f} {cy:.4f} 0.0006",
                      size=f"{hx:.4f} {hy:.4f} 0.0005", rgba="0.15 0.65 0.30 0.55", contype="0", conaffinity="0")
    for ob in env.obstacles:
        size = mj_size(ObjectSpec(name=ob.name, shape=ob.shape, size=ob.size, mass_kg=1.0))
        h = ObjectSpec(name=ob.name, shape=ob.shape, size=ob.size, mass_kg=1.0).height
        ET.SubElement(wb, "geom", name=f"obs_{ob.name}", type=ob.shape, size=size,
                      pos=f"{ob.center[0]:.4f} {ob.center[1]:.4f} {ob.z + h / 2:.4f}",
                      rgba=" ".join(f"{c:.3f}" for c in ob.rgba))

    notes: list[str] = []
    top = next((c for c in spec.cameras if c.role == "overhead" and c.pos and c.xyaxes), None)
    if top is not None:
        for cam in wb.findall("camera"):
            if cam.get("name") == "top":
                cam.set("pos", " ".join(f"{v:.4f}" for v in top.pos))  # type: ignore[union-attr]
                cam.set("xyaxes", " ".join(f"{v:.4f}" for v in top.xyaxes))  # type: ignore[union-attr]
                notes.append("overhead camera placed at the calibrated pose")

    xml = ET.tostring(mj, encoding="unicode")
    speeds = json.dumps({c.name: c.speed_mps for c in env.conveyors}, sort_keys=True) if env.conveyors else ""
    digest = hashlib.sha256((xml + json.dumps(starts, sort_keys=True) + speeds).encode()).hexdigest()[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scene_{digest}.xml"
    path.write_text(xml)
    return SceneInfo(path=path, objects={o.name: body_name(o.name) for o in spec.objects},
                     zones={z.name: (tuple(z.center), tuple(z.half_size)) for z in spec.zones},  # type: ignore[misc]
                     starts=starts, heights={o.name: o.height for o in spec.objects},
                     shapes={o.name: o.shape for o in spec.objects},
                     pick=spec.task.pick, place=list(spec.task.place), lighting=env.lighting,
                     digest=digest, notes=notes,
                     conveyors={c.name: c.model_dump(mode="json", exclude={"name"}) for c in env.conveyors},
                     on={o.name: o.on for o in spec.objects if o.on})


def instruction(spec: IntakeSpec) -> str:
    if spec.task.pick and spec.task.place:
        return f"Pick up the {spec.task.pick.replace('_', ' ')} and place it in {spec.task.place[0].replace('_', ' ')}."
    return spec.prompt or "Pick and place."
