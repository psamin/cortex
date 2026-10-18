"""Intake contracts: what a calling agent tells Cortex about its arm, cameras, objects and task.

Units are SI (metres, kilograms) except joint values, which use the so101_follower
convention: degrees for the arm and 0-100 for the gripper. Table coordinates are
the sim world frame: x away from the arm base, y to the arm's left, z up from the
table surface.

Every value Cortex fills in itself (catalogue defaults, arm-model defaults) is
listed in the owning object's ``assumed`` set, so the agent can see what it still
has to confirm.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec2 = tuple[float, float]
Shape = Literal["box", "cylinder", "sphere"]
ResetPolicy = Literal["return_to_origin", "random_zone", "cycle_zones"]
CameraRole = Literal["overhead", "front", "wrist"]


class Strict(BaseModel):
    """Unknown keys are errors, so an agent's typo is reported instead of silently dropped."""

    model_config = ConfigDict(extra="forbid")

SUPPORTED_ARMS = ("so101",)
# Limits that keep scene builds fast and bounded (they run on the single GL thread).
MAX_OBJECTS, MAX_ZONES, MAX_OBSTACLES, MAX_CAMERAS = 8, 12, 16, 4
NAME = r"^[a-z0-9_]{1,40}$"  # names are slugged by the intake service before validation
TEXT = 500
# The EnvHub env's HOME: arm folded above the table, gripper half open.
SO101_HOME = [0.0, -60.0, 60.0, 35.0, 0.0, 30.0]
SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class RobotProfile(Strict):
    model: Literal["so101"] = "so101"
    port: str | None = None  # e.g. COM5 or /dev/ttyACM0; real arms only
    calibration_file: str | None = None  # LeRobot so101_follower calibration JSON; real arms only
    home_pose: list[float] = Field(default_factory=lambda: list(SO101_HOME), min_length=6, max_length=6)
    # Optional narrowing of the model's joint limits, degrees (gripper 0-100). None = model limits.
    joint_limits: list[tuple[float, float]] | None = None
    max_real_speed_deg_s: float = Field(60.0, gt=0, le=180)
    assumed: set[str] = Field(default_factory=lambda: {"home_pose"})


class CameraSpec(Strict):
    name: str = Field(pattern=NAME)
    role: CameraRole
    device: str | None = None  # index or URL for a real camera; None in sim
    width: int = 640
    height: int = 480
    # Calibrated pose in the table frame: position (m) and MuJoCo-style xyaxes. Set by calibration.
    pos: tuple[float, float, float] | None = None
    xyaxes: tuple[float, float, float, float, float, float] | None = None


class Rect(Strict):
    center: Vec2
    half_size: Vec2


class Workspace(Strict):
    x: Vec2 = (0.08, 0.36)
    y: Vec2 = (-0.22, 0.24)
    forbidden: list[Rect] = []
    assumed: set[str] = Field(default_factory=lambda: {"x", "y"})


class ObjectSpec(Strict):
    """A graspable primitive. ``size``: box = full x, y, z extents; cylinder = diameter, height;
    sphere = diameter."""

    name: str = Field(pattern=NAME)
    shape: Shape
    size: list[float] = Field(min_length=1, max_length=3)
    mass_kg: float = Field(gt=0, le=5)
    rgba: tuple[float, float, float, float] = (0.85, 0.15, 0.12, 1.0)
    start: Vec2 | None = None  # nominal start on the table; None = Cortex picks one
    start_jitter_m: float = Field(0.015, ge=0, le=0.1)
    assumed: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def _size_matches_shape(self) -> ObjectSpec:
        need = {"box": 3, "cylinder": 2, "sphere": 1}[self.shape]
        if len(self.size) != need:
            raise ValueError(f"{self.shape} size needs {need} values, got {len(self.size)}")
        if any(s <= 0 or s > 0.5 for s in self.size):
            raise ValueError("object dimensions must be in (0, 0.5] m")
        return self

    @property
    def grasp_width(self) -> float:
        """Narrowest horizontal extent, the width the gripper must close around."""
        return min(self.size[:2]) if self.shape == "box" else self.size[0]

    @property
    def height(self) -> float:
        return {"box": self.size[-1], "cylinder": self.size[1], "sphere": self.size[0]}[self.shape]


class PlaceZone(Strict):
    name: str = Field(pattern=NAME)
    center: Vec2
    half_size: Vec2 = (0.04, 0.04)


class TaskV2(Strict):
    pick: str | None = None  # ObjectSpec.name
    place: list[str] = []  # PlaceZone names; one is the target, the rest feed cycle_zones
    reset_policy: ResetPolicy = "return_to_origin"
    sim_target: float = Field(0.7, ge=0, le=1)
    twin_target: float = Field(0.8, ge=0, le=1)  # MVP: success target in the physics-sim twin
    real_target: float = Field(0.8, ge=0, le=1)  # hardware track (post-MVP)
    demos_per_round: int = Field(10, ge=1, le=200)
    eval_trials: int = Field(5, ge=1, le=100)
    max_rounds: int = Field(5, ge=1, le=50)
    model_budget_usd: float = Field(10.0, ge=0)


class Obstacle(Strict):
    """A static, non-graspable primitive the arm must avoid (a shelf, a wall, a mug stand)."""

    name: str = Field(pattern=NAME)
    shape: Shape
    size: list[float] = Field(min_length=1, max_length=3)  # same conventions as ObjectSpec.size
    center: Vec2
    z: float = 0.0  # bottom of the obstacle above the table surface
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)


class Environment(Strict):
    """Everything around the objects that the sim scene should match."""

    table_size: Vec2 = (0.80, 0.80)  # metres, centred 0.2 m in front of the arm base
    table_rgba: tuple[float, float, float, float] = (0.80, 0.72, 0.60, 1.0)
    lighting: Literal["dim", "normal", "bright"] = "normal"
    obstacles: list[Obstacle] = Field(default_factory=list, max_length=MAX_OBSTACLES)
    description: str = Field("", max_length=TEXT)  # free text from the agent, e.g. "white desk by a window, afternoon light"
    assumed: set[str] = Field(default_factory=lambda: {"table_size", "table_rgba", "lighting"})


class IntakeSpec(Strict):
    prompt: str = Field("", max_length=TEXT)
    mode: Literal["sim", "real"] = "sim"
    robot: RobotProfile = Field(default_factory=RobotProfile)
    cameras: list[CameraSpec] = Field(default_factory=lambda: [CameraSpec(name="top", role="overhead")],
                                      max_length=MAX_CAMERAS)
    workspace: Workspace = Field(default_factory=Workspace)
    environment: Environment = Field(default_factory=Environment)
    objects: list[ObjectSpec] = Field(default_factory=list, max_length=MAX_OBJECTS)
    zones: list[PlaceZone] = Field(default_factory=list, max_length=MAX_ZONES)
    task: TaskV2 = Field(default_factory=TaskV2)


class Question(BaseModel):
    """Something Cortex still needs. ``path`` is where the answer goes in an ``update_intake`` patch."""

    path: str
    type: Literal["string", "number", "enum", "object", "list", "confirm"]
    prompt: str
    why: str
    options: list[str] = []
    default: object | None = None
    required: bool = True
    # A ready-to-send update_intake patch that accepts the default (or the first option).
    patch: dict | None = None
    # For enum questions whose options need different values: one ready-to-send patch per option.
    option_patches: dict[str, dict] = {}


class IntakeStatus(BaseModel):
    spec: IntakeSpec
    questions: list[Question]
    complete: bool
    notes: list[str] = []
    changed: list[str] = []  # paths the last patch changed
    still_assumed: list[str] = []  # values Cortex filled in that nobody has confirmed
