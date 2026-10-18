"""Collection on Windows: live preview frames can't fail a run, and the Claude Code demonstrator gets the real task."""

import numpy as np

from cortex.models import Scenario
from cortex.services.claude_code_collector import _prompt
from cortex.workers.livefile import write_live_jpg


def test_live_frame_survives_a_reader_holding_the_file(tmp_path):
    frame = np.zeros((8, 8, 3), np.uint8)
    assert write_live_jpg(frame, tmp_path)
    with open(tmp_path / "live.jpg", "rb"):  # a reader (the studio) has it open: on Windows replace is refused
        write_live_jpg(frame + 1, tmp_path)  # must not raise, whatever the OS does
    assert write_live_jpg(frame + 2, tmp_path)  # and frames resume once the reader lets go
    assert not list(tmp_path.glob("*.tmp"))


def test_demo_prompt_names_the_intake_object_and_zone():
    sc = Scenario(cube_region="center", lighting="dim")
    p = _prompt("rs_1", sc, {"scene": "intake", "object": "cup", "object_text": "the white cup (8 cm wide)",
                             "grasp_mode": "side", "target_zone": "coaster"})
    assert "the white cup" in p and "'coaster'" in p and "zone='coaster'" in p and "red cube" not in p


def test_demo_prompt_keeps_the_cube_task_without_a_scene():
    p = _prompt("rs_1", Scenario(cube_region="left", lighting="normal"), None)
    assert "red cube" in p and "green box" in p and "cube left" in p
