"""Scene setup for an intake: a photo or a description read by the user's Claude Code (stubbed here, never
run), or the default tabletop. Estimates are validated, kept inside the arm's reach and left `assumed`."""

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cortex.services import claude_code_agent as cc
from cortex.services import claude_code_diagnosis as ccd
from cortex.services import scene_setup
from cortex.sim.so101 import configure_gl

configure_gl()

ESTIMATE = {
    "objects": [
        {"name": "red mug", "catalogue": "cup", "start": [0.25, 0.10], "rgba": [0.8, 0.1, 0.1, 1]},
        {"name": "Juice Box", "catalogue": None, "shape": "box", "size": [0.04, 0.06, 0.1], "mass_kg": 0.2,
         "start": [0.9, -0.5]},  # beyond the arm's reach: pulled inside
        {"name": "ghost", "catalogue": None, "shape": "blob"},  # unusable: dropped, said so
    ],
    "zones": [{"name": "Tray", "center": [0.22, -0.12], "half_size": [0.06, 0.05]}],
    "pick": "red mug", "place": "tray",
    "environment": {"lighting": "dim", "description": "wooden desk by a window",
                    "obstacles": [{"name": "over_tray", "shape": "box", "size": [0.1, 0.1, 0.2], "center": [0.22, -0.12]},
                                  {"name": "lamp", "shape": "cylinder", "size": [0.08, 0.3], "center": [0.12, 0.2]}]},
    "notes": ["the mug could be a little larger"],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "PLANNER_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


@pytest.fixture
def claude(monkeypatch):
    """Stands in for `claude -p`: records each prompt and its work dir, answers with ESTIMATE."""
    seen = []

    async def fake_run(prompt, workdir, model=None, timeout_s=0):
        seen.append({"prompt": prompt, "files": sorted(p.name for p in workdir.iterdir())})
        return "Here is the scene:\n" + json.dumps(ESTIMATE)

    monkeypatch.setattr(cc, "available", lambda: None)
    monkeypatch.setattr(ccd, "run_claude", fake_run)
    return seen


def tool(c, name, status=200, **args):
    r = c.post(f"/api/tools/{name}", json=args, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == status, r.text
    return r.json()


def photo_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), (200, 180, 150)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_default_scene_is_complete_without_a_model(client):
    st = tool(client, "start_intake", scene_source="default")
    assert st["complete"] and st["scene_setup"]["source"] == "default"
    cube = st["spec"]["objects"][0]
    assert cube["name"] == "cube" and cube["start"] == [0.22, 0.12] and cube["assumed"] == []
    assert st["spec"]["task"] == {**st["spec"]["task"], "pick": "cube", "place": ["target"]}
    assert "call build_simulation" in st["summary"]


def test_a_photo_becomes_a_validated_estimate(client, claude):
    st = tool(client, "start_intake", prompt="put the mug on the tray", image_b64=photo_b64())
    rep = st["scene_setup"]
    assert rep["source"] == "photo" and rep["model"].startswith("Claude Code")
    # Claude Code read the photo from its work dir (downscaled JPEG), with the frame and catalogue in the prompt
    assert claude[0]["files"] == ["photo.jpg"] and "./photo.jpg" in claude[0]["prompt"]
    assert "+y to the arm's left" in claude[0]["prompt"] and "water bottle" in claude[0]["prompt"]
    objs = {o["name"]: o for o in st["spec"]["objects"]}
    assert set(objs) == {"red_mug", "juice_box"}
    assert objs["red_mug"]["shape"] == "cylinder" and objs["red_mug"]["rgba"] == [0.8, 0.1, 0.1, 1.0]  # catalogue size, photo colour
    x, y = objs["juice_box"]["start"]
    assert x <= 0.36 and y >= -0.22  # inside the workspace
    assert all({"shape", "size", "mass_kg", "start"} <= set(o["assumed"]) for o in objs.values())  # human confirms
    assert {q["path"] for q in st["questions"] if q["type"] == "confirm"} == {"objects.red_mug", "objects.juice_box"}
    assert st["spec"]["task"]["pick"] == "red_mug" and st["spec"]["task"]["place"] == ["tray"]
    assert st["spec"]["environment"]["lighting"] == "dim"
    assert [o["name"] for o in st["spec"]["environment"]["obstacles"]] == ["lamp"]  # the one over the tray is dropped
    adjusted = " ".join(rep["adjusted"])
    assert "ghost" in adjusted and "over_tray" in adjusted and "juice_box" in adjusted
    assert rep["model_notes"] == ["the mug could be a little larger"]
    # the photo is kept with the project
    assert rep["photo"]["key"] == f"projects/{st['project_id']}/intake/photo.jpg"
    from cortex.api import app as appmod

    assert client.portal.call(appmod.ctx().blobs.get_bytes, rep["photo"]["key"])[:2] == b"\xff\xd8"


def test_a_description_is_read_the_same_way(client, claude):
    st = tool(client, "start_intake", prompt="mug to tray", scene_source="description",
              description="A red mug 25 cm in front of the arm, a tray to its right, a lamp at the back left.")
    assert st["scene_setup"]["source"] == "description" and "A red mug 25 cm" in claude[0]["prompt"]
    assert claude[0]["files"] == []  # no image
    assert {o["name"] for o in st["spec"]["objects"]} == {"red_mug", "juice_box"}


def test_bad_inputs_are_refused_before_a_project_exists(client, claude):
    before = len(client.get("/api/projects").json())
    r = tool(client, "start_intake", status=400, scene_source="photo")
    assert "needs the photo" in r["detail"]
    r = tool(client, "start_intake", status=400, image_b64=base64.b64encode(b"not an image").decode())
    assert "could not be read as an image" in r["detail"]
    assert len(client.get("/api/projects").json()) == before and claude == []


def test_no_claude_code_says_what_to_do_instead(client, monkeypatch):
    monkeypatch.setattr(cc, "available", lambda: "Claude Code not found")
    r = tool(client, "start_intake", status=400, image_b64=photo_b64())
    assert "Claude Code" in r["detail"] and "scene_source='default'" in r["detail"]


def test_an_unreadable_reply_is_an_error_not_a_guess(client, monkeypatch):
    async def chatty(prompt, workdir, model=None, timeout_s=0):
        return "I think there is a mug."

    monkeypatch.setattr(cc, "available", lambda: None)
    monkeypatch.setattr(ccd, "run_claude", chatty)
    r = tool(client, "start_intake", status=400, scene_source="description", description="a mug")
    assert "did not return a scene" in r["detail"]


def test_claude_code_is_found_where_the_desktop_app_installs_it(tmp_path, monkeypatch):
    for v in ("2.1.9", "2.1.111"):
        d = tmp_path / "Claude" / "claude-code" / v / "abc"
        d.mkdir(parents=True)
        (d / "claude.exe").write_text("")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(cc.shutil, "which", lambda name: None)
    from cortex.config import settings

    settings.cache_clear()
    assert cc.executable().endswith(str(tmp_path / "Claude" / "claude-code" / "2.1.111" / "abc" / "claude.exe"))
    assert cc.available() is None


def test_photos_are_normalised_to_a_readable_jpeg():
    jpg = scene_setup.decode_photo("data:image/png;base64," + photo_b64())
    img = Image.open(io.BytesIO(jpg))
    assert img.format == "JPEG" and max(img.size) == scene_setup.MAX_SIDE_PX
