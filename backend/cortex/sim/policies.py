"""Policies that can drive PickPlaceEnv: the scripted oracle and LeRobot checkpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..errors import UserError
from .expert import ScriptedExpert
from .so101 import TASK_TEXT, PickPlaceEnv

GPU_ONLY = "Trained policies run on the GPU cluster, not this Mac: start the GPU server from the Inference tab"
CPU_FOR_TESTS = "CORTEX_TEST_ONLY_CPU_INFERENCE"  # tests only: lets a tiny fake checkpoint run on CPU


def has_cuda() -> bool:
    if sys.platform == "darwin":  # macOS has no CUDA; don't import torch to learn that
        return False
    import torch

    return torch.cuda.is_available()


def inference_device(device: str = "cuda") -> str:
    """Where a trained policy may run: a CUDA GPU, never this Mac's CPU (#290). Raises UserError otherwise.
    The scripted oracle has no network and never asks."""
    if os.environ.get(CPU_FOR_TESTS) == "1":
        return "cpu"
    if device.startswith("cuda") and has_cuda():
        return device
    raise UserError(GPU_ONLY)


class Policy(Protocol):
    name: str

    def reset(self, env: PickPlaceEnv, obs: dict[str, Any], seed: int) -> None: ...
    def act(self, obs: dict[str, Any]) -> np.ndarray: ...


class ExpertPolicy:
    name = "scripted_expert (oracle, privileged state)"
    needs_pixels = False

    def reset(self, env, obs, seed):
        self.ex = ScriptedExpert(env, np.random.default_rng(seed))
        self.ex.plan(obs["agent_pos"])

    def act(self, obs):
        return self.ex.act()


class LeRobotPolicy:
    """Any LeRobot pretrained_model/ directory (ACT, SmolVLA, ...), run on a CUDA GPU only.

    Uses the saved pre/post processors so normalization matches training."""

    def __init__(self, path: str | Path, device: str = "cuda", task: str = TASK_TEXT,
                 cameras: dict[str, str] | None = None):
        device = inference_device(device)  # before any weights are read
        import json

        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class

        self.torch = torch
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        self.name = f"{cfg.get('type', 'policy')} @ {path.parent.name if path.name == 'pretrained_model' else path.name}"
        cls = get_policy_class(cfg["type"])
        self.policy = cls.from_pretrained(str(path)).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(path))
        self.device, self.task = device, task
        self.cameras = cameras or {}  # policy camera name -> env camera (e.g. smolvla_base's camera1 -> front)
        self.image_keys = [k for k in cfg.get("input_features", {}) if k.startswith("observation.images.")]

    def reset(self, env, obs, seed):
        self.torch.manual_seed(seed)
        self.policy.reset()

    def act(self, obs):
        t = self.torch
        batch: dict[str, Any] = {
            "observation.state": t.from_numpy(np.asarray(obs["agent_pos"], np.float32))[None],
            "task": self.task,
        }
        for key in self.image_keys:
            cam = key.rsplit(".", 1)[-1]
            cam = self.cameras.get(cam, cam)
            if cam not in obs["pixels"]:
                continue  # e.g. smolvla_base's third camera: the policy treats it as absent
            img = t.from_numpy(obs["pixels"][cam]).permute(2, 0, 1).float() / 255.0
            batch[key] = img[None]
        with t.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)


def load_policy(kind: str, checkpoint_path: str | None) -> Policy:
    if kind == "scripted_expert":
        return ExpertPolicy()
    if not checkpoint_path:
        raise ValueError("checkpoint policy needs a checkpoint path")
    return LeRobotPolicy(checkpoint_path)
