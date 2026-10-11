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
