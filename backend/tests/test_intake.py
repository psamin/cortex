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
    assert "objects[water_bottle]" in paths(intake.status(spec))


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
