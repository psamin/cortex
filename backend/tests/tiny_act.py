"""A tiny randomly initialised ACT checkpoint in LeRobot's on-disk format (config, weights, processors)."""

from pathlib import Path

import numpy as np


def make_tiny_act(out: Path, cameras=("front",)) -> Path:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    images = {f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)) for c in cameras}
    cfg = ACTConfig(input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)), **images},
                    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
                    chunk_size=4, n_action_steps=4, dim_model=32, n_heads=2, dim_feedforward=64,
                    n_encoder_layers=1, n_decoder_layers=1, use_vae=False, pretrained_backbone_weights=None,
                    device="cpu")
    vec = {"mean": np.zeros(6), "std": np.full(6, 30.0), "min": np.full(6, -100.0), "max": np.full(6, 100.0)}
    img = {"mean": np.full((3, 1, 1), 0.5), "std": np.full((3, 1, 1), 0.25), "min": np.zeros((3, 1, 1)),
           "max": np.ones((3, 1, 1))}
    stats = {"observation.state": vec, "action": vec, **{k: img for k in images}}
    stats = {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()} for k, d in stats.items()}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    out.mkdir(parents=True, exist_ok=True)
    ACTPolicy(cfg).save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    return out
