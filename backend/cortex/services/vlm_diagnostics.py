"""VLM failure hypotheses from episode keyframes.

The model sees a contact sheet of keyframes plus the measured telemetry and
returns categories from a fixed taxonomy with a self-reported confidence. These
are stored as *hypotheses*; a human can confirm or reject them, and they never
override simulator-measured outcomes.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from ..models import EpisodeDiagnosis, Hypothesis

TAXONOMY = {
    "grasp_miss": "gripper closes without securing the cube",
    "slip_during_transport": "cube slips or falls while being carried",
    "release_outside_target": "cube released but not inside the box",
    "collision": "arm or cube hits the box wall / table unexpectedly",
    "poor_visibility": "lighting, shadow or occlusion makes the cube hard to see",
    "hesitation_or_timeout": "robot stalls, oscillates or runs out of time",
    "target_not_reached": "arm never gets close to the cube",
    "no_failure_visible": "nothing in the frames indicates a failure",
}

PROMPT = """You are reviewing a robot manipulation episode (SO-101 arm, task: {task}).
The image is a contact sheet of {n} keyframes in time order (left to right, top to bottom), front camera.
Measured from the simulator (ground truth): {telemetry}

Propose up to 3 hypotheses for what went wrong, using ONLY these categories:
{taxonomy}

Return strict JSON: {{"hypotheses": [{{"category": str, "confidence": number 0-1, "rationale": str (<=30 words), "frame_indices": [int]}}]}}
Confidence is your own uncertainty; say "no_failure_visible" if the frames do not show a failure."""


def contact_sheet(frames: list[np.ndarray], cols: int = 3, width: int = 960) -> bytes:
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    img = Image.fromarray(sheet)
    img = img.resize((width, int(width * img.height / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def keyframes_from_video(path: Path, n: int = 6) -> list[np.ndarray]:
    import av

    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def keyframes_from_array(frames: np.ndarray, n: int = 6) -> list[np.ndarray]:
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def parse_hypotheses(text: str) -> list[Hypothesis]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in VLM reply")
    raw = json.loads(m.group(0)).get("hypotheses", [])
    out = []
    for h in raw[:3]:
        cat = h.get("category", "")
        if cat not in TAXONOMY:
            continue
        out.append(Hypothesis(category=cat, confidence=float(np.clip(h.get("confidence", 0.5), 0, 1)),
                              rationale=str(h.get("rationale", ""))[:300],
                              frame_indices=[int(i) for i in h.get("frame_indices", [])][:6]))
    return out


async def diagnose(route, ref_id: str, frames: list[np.ndarray], task: str,
                   telemetry: dict[str, Any]) -> EpisodeDiagnosis:
    """route: context.ModelRoute (self-hosted or OpenRouter); its vlm_model reviews the keyframes."""
    model = route.vlm_model
    img = base64.b64encode(contact_sheet(frames)).decode()
    text = PROMPT.format(task=task, n=len(frames), telemetry=json.dumps(telemetry),
                         taxonomy="\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items()))
    body = {
        "model": model, "temperature": 0, "max_tokens": 600,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}],
    }
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(route.chat_url, json=body, headers={"Authorization": f"Bearer {route.api_key}"})
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
        return EpisodeDiagnosis(episode_id=ref_id, model=model, hypotheses=parse_hypotheses(reply))
    except Exception as e:  # recorded on the analysis, never silently dropped
        return EpisodeDiagnosis(episode_id=ref_id, model=model, hypotheses=[], error=f"{type(e).__name__}: {e}"[:300])
