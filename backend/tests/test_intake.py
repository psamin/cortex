import pytest

from cortex import intake
from cortex.intake import catalogue


def paths(st):
    return {q.path for q in st.questions if q.required}


def test_prompt_seeds_catalogue_objects_as_assumed():
    spec = intake.start("Pick up the water bottle and put it on the shelf")
    assert [o.name for o in spec.objects] == ["water_bottle"]
    assert spec.objects[0].shape == "cylinder" and spec.objects[0].assumed == {"shape", "size", "mass_kg"}
    assert spec.task.pick == "water_bottle"
    st = intake.status(spec)
    q = next(q for q in st.questions if q.path == "objects.water_bottle")
    assert q.type == "enum" and "small water bottle" in q.options and q.patch is not None
    assert "objects.water_bottle.mass_kg" in st.still_assumed


def test_longest_catalogue_match_wins():
    assert catalogue.lookup("small water bottle") == "small water bottle"
    assert catalogue.mentioned("move the small water bottle") == ["small water bottle"]
    assert catalogue.lookup("soda can") == "can"


def test_sim_intake_completes_after_confirm_and_zones():
    spec = intake.start("bottle")
    spec = intake.apply_patch(spec, {"objects": [{"name": "water_bottle", "confirm": True}],
                                     "zones": [{"name": "spot_a", "center": [0.22, 0.12]},
                                               {"name": "spot_b", "center": [0.22, -0.12]}],
                                     "task": {"place": ["spot_a", "spot_b"], "reset_policy": "cycle_zones"}})
    st = intake.status(spec)
    assert st.complete, [q.path for q in st.questions if q.required]
    assert spec.objects[0].assumed == set()


def test_setting_a_field_clears_its_assumption():
    spec = intake.start("cup")
    spec = intake.apply_patch(spec, {"objects": [{"name": "cup", "mass_kg": 0.18}]})
    assert spec.objects[0].mass_kg == 0.18 and spec.objects[0].assumed == {"shape", "size"}


def test_unknown_object_needs_dimensions():
    with pytest.raises(intake.IntakeError, match="not in the catalogue"):
        intake.apply_patch(intake.start(), {"objects": [{"name": "widget"}]})
    spec = intake.apply_patch(intake.start(), {"objects": [{"name": "widget", "shape": "box",
                                                           "size": [0.03, 0.05, 0.02], "mass_kg": 0.05}]})
    assert spec.objects[0].grasp_width == 0.03 and spec.objects[0].assumed == set()


def test_delete_and_reference_checks():
    spec = intake.start("cube and ball")
    assert {o.name for o in spec.objects} == {"cube", "ball"}
    with pytest.raises(intake.IntakeError, match="task.pick"):
        intake.apply_patch(spec, {"task": {"pick": "bottle"}})
    spec = intake.apply_patch(spec, {"objects": [{"name": "ball", "_delete": True}]})
    assert [o.name for o in spec.objects] == ["cube"]
    with pytest.raises(intake.IntakeError, match="unknown zones"):
        intake.apply_patch(spec, {"task": {"place": ["nowhere"]}})


def test_joint_limits_can_only_narrow():
    spec = intake.start()
    wide = [(-120, 120)] + [(-90, 90)] * 4 + [(0, 100)]
    with pytest.raises(intake.IntakeError, match="shoulder_pan"):
        intake.apply_patch(spec, {"robot": {"joint_limits": wide}})
    narrow = [(-90, 90), (-90, 90), (-90, 90), (-90, 90), (-90, 90), (0, 100)]
    assert intake.apply_patch(spec, {"robot": {"joint_limits": narrow}}).robot.joint_limits is not None


def test_real_mode_asks_for_hardware_details():
    spec = intake.start("cube", mode="real")
    p = paths(intake.status(spec))
    assert {"robot.port", "robot.calibration_file", "robot.home_pose", "cameras"} <= p
    spec = intake.apply_patch(spec, {"robot": {"port": "COM5", "calibration_file": "cal.json",
                                               "home_pose": [0, -60, 60, 35, 0, 30]},
                                     "cameras": [{"name": "top", "device": "0", "pos": [0.2, 0, 0.5],
                                                  "xyaxes": [0, -1, 0, 1, 0, 0]}]})
    p = paths(intake.status(spec))
    assert not {"robot.port", "robot.calibration_file", "robot.home_pose", "cameras"} & p


def test_bad_values_are_refused_with_a_reason():
    with pytest.raises(intake.IntakeError, match="cylinder size needs 2"):
        intake.apply_patch(intake.start(), {"objects": [{"name": "x", "shape": "cylinder", "size": [0.05],
                                                        "mass_kg": 0.1}]})


def test_prompt_seeding_skips_ordinary_words():
    spec = intake.start("Can you pick up the bottle and put it in the box?")
    assert [o.name for o in spec.objects] == ["water_bottle"] and spec.task.pick == "water_bottle"


def test_alias_updates_existing_object_instead_of_duplicating():
    spec = intake.start("pick up the bottle")
    before = spec
    spec = intake.apply_patch(spec, {"objects": [{"name": "bottle", "mass_kg": 0.3}]})
    assert [o.name for o in spec.objects] == ["water_bottle"] and spec.objects[0].mass_kg == 0.3
    assert intake.status(spec, before).changed == ["objects.water_bottle.mass_kg"]


def test_typos_and_assumed_are_refused():
    with pytest.raises(intake.IntakeError, match="pick_object"):
        intake.apply_patch(intake.start(), {"task": {"pick_object": "cube"}})
    with pytest.raises(intake.IntakeError, match="assumed"):
        intake.apply_patch(intake.start("cube"), {"objects": [{"name": "cube", "assumed": []}]})


def test_sanity_checks_on_zones_home_and_resets():
    with pytest.raises(intake.IntakeError, match="outside the workspace"):
        intake.apply_patch(intake.start(), {"zones": [{"name": "far", "center": [5, 5]}]})
    with pytest.raises(intake.IntakeError, match="home pose"):
        intake.apply_patch(intake.start(), {"robot": {"home_pose": [500, 0, 0, 0, 0, 0]}})
    spec = intake.apply_patch(intake.start("cube"), {"objects": [{"name": "cube", "confirm": True}],
                                                    "zones": [{"name": "a", "center": [0.22, 0.1]}],
                                                    "task": {"place": ["a"], "reset_policy": "cycle_zones"}})
    assert not intake.status(spec).complete


def test_every_required_question_patch_is_accepted():
    spec = intake.start("bottle")
    for _ in range(6):
        st = intake.status(spec)
        if st.complete:
            break
        q = next(q for q in st.questions if q.required and q.patch)
        spec = intake.apply_patch(spec, q.patch)
    assert intake.status(spec).complete


def test_environment_and_obstacles():
    spec = intake.apply_patch(intake.start(), {"environment": {
        "lighting": "dim", "description": "white desk",
        "obstacles": [{"name": "shelf", "shape": "box", "size": [0.1, 0.3, 0.2], "center": [0.35, 0.0]}]}})
    assert spec.environment.lighting == "dim" and "lighting" not in spec.environment.assumed
    spec = intake.apply_patch(spec, {"environment": {"obstacles": [{"name": "shelf", "_delete": True}]}})
    assert spec.environment.obstacles == []


def test_limits_names_and_obstacle_overlap():
    many = [{"name": f"z{i}", "center": [0.2, 0.0]} for i in range(13)]
    with pytest.raises(intake.IntakeError, match="zones"):
        intake.apply_patch(intake.start(), {"zones": many})
    spec = intake.apply_patch(intake.start(), {"zones": [{"name": "Spot A!", "center": [0.22, 0.12]}]})
    assert spec.zones[0].name == "spot_a"  # names are slugged
    with pytest.raises(intake.IntakeError, match="too large"):
        intake.apply_patch(intake.start(), {"environment": {"description": "x" * 30_000}})
    with pytest.raises(intake.IntakeError, match="arm base"):
        intake.apply_patch(intake.start(), {"environment": {"obstacles": [
            {"name": "slab", "shape": "box", "size": [0.5, 0.5, 0.1], "center": [0.0, 0.0]}]}})
    with pytest.raises(intake.IntakeError, match="zone 'spot_a'"):
        intake.apply_patch(spec, {"environment": {"obstacles": [
            {"name": "mug", "shape": "cylinder", "size": [0.08, 0.1], "center": [0.23, 0.12]}]}})
