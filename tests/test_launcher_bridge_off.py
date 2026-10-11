"""PROPRIO_BRIDGE=0: the way back to the pipeline reading the phone itself."""
import launcher_checks as checks
from launcher_checks import AMBER


def test_switching_the_bridge_off_makes_everything_act_as_if_there_were_none(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "phone_bridge.py").write_text("")
    on = checks.Setup(root=tmp_path, env={})
    off = checks.Setup(root=tmp_path, env={"PROPRIO_BRIDGE": "0"})
    assert on.bridge.exists() and not off.bridge.exists()
    stream = {l["id"]: l for l in checks.phone(off)}["phone_stream"]
    assert stream["level"] == AMBER and not stream["required"] and "switched off" in stream["message"]


def test_an_edit_to_the_settings_file_counts_at_the_next_start_without_ending_the_keeper(tmp_path):
    import launcher
    sup = launcher.Supervisor(root=tmp_path, env={"PROPRIO_OPEN": "0"}, run=lambda cmd, timeout=4.0: (0, ""), home=tmp_path / "home")
    sup._env_from_file = True                       # as when the icon starts it
    held = sup.s.env
    (tmp_path / ".env.launcher").write_text("GUIDANCE_HAND_GIVE_UP_S=60\nPROPRIO_BRIDGE=0\n")
    sup.reread_settings()
    assert sup.env["GUIDANCE_HAND_GIVE_UP_S"] == "60" and sup.s.env is held and held["PROPRIO_BRIDGE"] == "0"
    (tmp_path / ".env.launcher").write_text("")
    sup.reread_settings()
    assert "GUIDANCE_HAND_GIVE_UP_S" not in sup.env and sup.env["GUIDANCE_MODE"] == "3"


FAMILIES = [{"family": "guidance", "heading": "Guidance sound", "say": "option", "default": 3,
             "options": [{"n": n, "key": k, "title": k, "about": ""} for n, k in enumerate(("tones", "steps", "combined"), 1)]},
            {"family": "hand", "heading": "Hand tracking", "say": "hand option", "default": 1,
             "options": [{"n": n, "key": k, "title": k, "about": ""} for n, k in enumerate(("mediapipe", "steady", "yolo", "big", "hybrid"), 1)]}]


def _keeper(tmp_path):
    import launcher
    sup = launcher.Supervisor(root=tmp_path, env={"PROPRIO_OPEN": "0", "GUIDANCE_MODE": "3"}, run=lambda cmd, timeout=4.0: (0, ""),
                              home=tmp_path / "home")
    sup.families = [dict(f) for f in FAMILIES]
    return sup


def test_an_option_picked_before_start_is_what_the_pipeline_starts_with(tmp_path):
    sup = _keeper(tmp_path)
    assert sup.choose("hand", 5) == (True, "Hand tracking: hybrid.")
    env = sup.pipeline_env()
    assert env["HAND_OPTION"] == "5" and env["GUIDANCE_MODE"] == "3"
    shown = {f["family"]: f for f in sup.options_state()}
    assert shown["hand"]["current"] == 5 and shown["hand"]["changed"] and not shown["guidance"]["changed"]
    again = _keeper(tmp_path)                           # the keeper is restarted: the choice is still there
    assert again.chosen == {"hand": 5} and again.pipeline_env()["HAND_OPTION"] == "5"


def test_a_click_while_running_goes_to_the_pipeline_and_a_voice_switch_is_remembered(tmp_path):
    sup = _keeper(tmp_path)
    sup.mode = "running"
    sup.pipeline.alive = lambda: True
    asked = []

    def pipeline(path, body=None, timeout_s=1.5):
        asked.append((path, body))
        if path == "/option":
            return {"ok": True, "title": "x"}
        return {"families": [{"family": "guidance", "current": 2}, {"family": "hand", "current": 5}]}
    sup.ask_pipeline = pipeline
    assert sup.choose("hand", 5)[0] and asked == [("/option", {"family": "hand", "n": 5})]
    sup.options_at = 0.0
    sup.follow_options(1000.0)                          # the wearer said "option two" meanwhile
    assert sup.chosen == {"hand": 5, "guidance": 2} and sup.pipeline_env()["GUIDANCE_MODE"] == "2"


def test_back_to_the_usual_options_and_nonsense_is_refused(tmp_path):
    sup = _keeper(tmp_path)
    sup.choose("hand", 3)
    sup.choose("guidance", 1)
    assert not sup.choose("hand", 9)[0] and not sup.choose("smell", 1)[0]
    assert sup.default_options()[0] and sup.chosen == {}
    assert "HAND_OPTION" not in sup.pipeline_env() and sup.pipeline_env()["GUIDANCE_MODE"] == "3"
    sup.choose("hand", 2)
    sup.choose("hand", 1)                               # picking the usual one again is not a change
    assert sup.chosen == {}
