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
