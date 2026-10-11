"""A pipeline that the keeper starts again in the middle of a reach carries on with it."""
import json
import time

import pytest

import main
from test_launcher import fake_mods, loop_args


def _left_behind(tmp_path, **status):
    folder = tmp_path / "status"
    folder.mkdir(parents=True)
    was = {"t": time.time() - 5, "state": "running", "target": "blue cup", "guidance": "guiding", "pid": 1}
    (folder / "pipeline.json").write_text(json.dumps({**was, **status}))


def _asked(tmp_path, monkeypatch, keeper="1"):
    monkeypatch.setenv("PROPRIO_KEEPER", keeper) if keeper else monkeypatch.delenv("PROPRIO_KEEPER", raising=False)
    mods = fake_mods()
    asked = []
    listen = mods.listen
    mods.listen = lambda: asked.append(1) or listen()
    beat = main.Heartbeat(tmp_path / "status")
    main.run(loop_args(tmp_path, "--frames", "3"), mods, beat)
    return len(asked), beat.target


def test_a_run_that_died_mid_reach_is_carried_on_with(tmp_path, monkeypatch):
    _left_behind(tmp_path)
    assert _asked(tmp_path, monkeypatch) == (0, "blue cup")


@pytest.mark.parametrize("status", [{"state": "stopped"}, {"state": "listening"}, {"guidance": "idle"}, {"guidance": "grab"},
                                    {"t": time.time() - 600}, {"target": None}, {"t": "yesterday"}])
def test_anything_else_asks_the_wearer_as_usual(tmp_path, monkeypatch, status):
    _left_behind(tmp_path, **status)
    assert _asked(tmp_path, monkeypatch) == (1, "mug")


def test_only_a_run_started_by_the_keeper_carries_on(tmp_path, monkeypatch):
    _left_behind(tmp_path)
    assert _asked(tmp_path, monkeypatch, keeper=None) == (1, "mug")


def test_a_first_run_with_no_status_left_behind_asks(tmp_path, monkeypatch):
    assert _asked(tmp_path, monkeypatch) == (1, "mug")


def test_the_wearer_is_told_when_the_camera_goes_and_when_it_is_back(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CAMERA_LOST_S", 0.05)
    mods = fake_mods()
    said = []
    mods.guidance.say_text = lambda text, wait=False: said.append(text)
    real, calls = mods.get_frame, [0]

    def get_frame():
        calls[0] += 1
        return None if 4 <= calls[0] < 60 else real()        # the picture goes for a while mid-reach
    mods.get_frame = get_frame
    main.run(loop_args(tmp_path, "--frames", "8"), mods, main.Heartbeat("off"))
    assert said == ["I have lost the camera. Hold still.", "The camera is back."]
