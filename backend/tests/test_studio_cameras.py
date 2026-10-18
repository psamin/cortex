from cortex.services.studio_service import cameras_for


def ids(cams):
    return [c["id"] for c in cams]


def test_sim_demo_names_studio_render_and_policy_cameras():
    cams = cameras_for("demo", {"planner": "claude-code"})
    assert ids(cams) == ["presentation", "front", "wrist"]
    assert cams[0]["label"] == "Studio view" and "never sees" in cams[0]["detail"]
    assert all("policy input" in c["label"] for c in cams[1:])


def test_imported_demo_uses_the_dataset_camera_names():
    prov = {"planner": "human demonstration (imported)", "source": "hf:lerobot/svla_so101_pickplace",
            "cameras": ["observation.images.up", "observation.images.side"]}
    cams = cameras_for("demo", prov)
    assert ids(cams) == ["presentation", "wrist"]
    assert cams[0]["label"].startswith("up camera") and cams[1]["label"].startswith("side camera")
    assert "Studio" not in " ".join(c["label"] for c in cams)
    assert ids(cameras_for("demo", {**prov, "cameras": ["observation.images.top"]})) == ["presentation"]


def test_attempts_and_evaluations_only_have_the_studio_render():
    assert ids(cameras_for("attempt")) == ids(cameras_for("eval")) == ["presentation"]
