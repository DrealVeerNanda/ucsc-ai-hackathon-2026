"""The dashboard's options panel: what it lists and how a click reaches the pipeline."""
import json
import urllib.request

import pytest

import dashboard
import guidance
import perception


@pytest.fixture
def dash(monkeypatch):
    monkeypatch.setattr(guidance, "_default", None)
    monkeypatch.setattr(perception, "_default", None)
    d = dashboard.Dashboard(host="127.0.0.1", port=0)
    d.start()
    d.port = d._server.server_address[1]            # port 0 asked the system for a free one
    yield d
    d.stop()


def _get(d, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{d.port}{path}", timeout=3) as r:
        return json.load(r)


def _post(d, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{d.port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.load(r)


def test_the_guidance_options_are_listed_without_starting_guidance(dash):
    families = {f["family"]: f for f in _get(dash, "/options")["families"]}
    assert [o["key"] for o in families["guidance"]["options"]][:3] == ["tones", "steps", "combined"]
    assert families["guidance"]["current"] is None and guidance._default is None       # no sound device was opened


def test_a_click_switches_the_option_and_names_it(dash, monkeypatch):
    said = []
    monkeypatch.setattr(guidance, "_default", object())                                 # "running"
    monkeypatch.setattr(guidance, "set_option", lambda n: {3: "combined words"}[n])
    monkeypatch.setattr(guidance, "option", lambda: (3, "combined", "combined words"))
    monkeypatch.setattr(guidance, "say_text", lambda text, wait=False: said.append(text))
    assert _post(dash, "/option", {"family": "guidance", "n": 3}) == {"ok": True, "title": "combined words"}
    assert said == ["Option 3: combined words."]
    assert {f["family"]: f["current"] for f in _get(dash, "/options")["families"]}["guidance"] == 3


def test_a_family_added_later_shows_up_and_is_switched_through_its_own_functions(dash, monkeypatch):
    chosen = []
    monkeypatch.setattr(perception, "_default", object())
    monkeypatch.setattr(perception, "HAND_OPTIONS", [("mediapipe", "as it was", "today's tracker"), ("depth", "arm in the lidar", "")], raising=False)
    monkeypatch.setattr(perception, "hand_option", lambda: (1, "mediapipe", "as it was"), raising=False)
    monkeypatch.setattr(perception, "set_hand_option", lambda n: chosen.append(n) or "arm in the lidar", raising=False)
    hand = {f["family"]: f for f in _get(dash, "/options")["families"]}["hand"]
    assert hand["current"] == 1 and [o["title"] for o in hand["options"]] == ["as it was", "arm in the lidar"]
    assert _post(dash, "/option", {"family": "hand", "n": 2})["ok"] and chosen == [2]


def test_bad_requests_are_answered_not_crashed_on(dash):
    assert _post(dash, "/option", {"family": "guidance", "n": 2}) == {"ok": False, "error": "not running yet"}
    assert not _post(dash, "/option", {"family": "nonsense", "n": 1})["ok"]
    assert not _post(dash, "/option", {})["ok"] and not _post(dash, "/option", [1, 2])["ok"]
