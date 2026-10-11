"""Cosmos Reason as a slow helper that looks at one picture. It is never in the steering loop:
a call takes about 0.6 s (2B) to 2 s (8B), against 35 ms for a whole frame of guidance.

    cosmos.list_objects(rgb)     -> ["mug", "phone", ...]        for "what's on the table?"
    cosmos.find(rgb, "can")      -> {"box": (x1, y1, x2, y2) or None, ...}   when the detector fails
    cosmos.available()           -> is the server running?

The server is llama.cpp on this machine, local only: bash tools/run_cosmos.sh  (port 8001).
COSMOS_URL changes the address; COSMOS_URL=off disables the helper.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request

import cv2
import numpy as np

import tracing

URL = os.environ.get("COSMOS_URL", "http://127.0.0.1:8001/v1")

# Box first and in 0-1000 coordinates, behind an explicit "visible": the request shape that scored
# best in the RTHACK tests. Without "visible" both models drew a box for objects that were not there.
FIND_PROMPT = ('Is there a {target} in this image? Reply with JSON only: {{"visible": true or false, '
               '"box": [x1, y1, x2, y2] bounding box of the {target} with coordinates normalized to 0-1000, '
               'or null if it is not visible, "grasp": "<where to hold it, a few words>"}}')
FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "box": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                          {"type": "null"}]},
        "grasp": {"type": "string", "maxLength": 60},
    },
    "required": ["visible", "box", "grasp"],
    "additionalProperties": False,
}
LIST_PROMPT = ('List the separate objects in this image that a person could pick up with one hand. Leave out '
               'furniture, people, and anything too big to lift with one hand. Use a short plain name for each, '
               'one to three words, most prominent first, at most {n}. Reply with JSON only: {{"objects": ["..."]}}')
LIST_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 8}},
    "required": ["objects"],
    "additionalProperties": False,
}


def available(url: str | None = None, timeout_s: float = 0.4) -> bool:
    url = url or URL
    if url == "off":
        return False
    try:
        with urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/health", timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def _ask(rgb: np.ndarray, prompt: str, schema: dict, name: str, url: str | None, max_side: int,
         max_tokens: int, timeout_s: float) -> tuple[dict, float, dict]:
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = rgb if s == 1.0 else cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    body = {
        "model": "cosmos", "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": prompt}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
    }
    req = urllib.request.Request((url or URL).rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        reply = json.load(r)
    return json.loads(reply["choices"][0]["message"]["content"]), time.perf_counter() - t0, reply.get("usage", {})


def find(rgb: np.ndarray, target: str, url: str | None = None, max_side: int = 640, timeout_s: float = 30.0) -> dict:
    """{"box": (x1, y1, x2, y2) in pixels of rgb or None, "grasp": str, "seconds": float}. Raises OSError
    if the server is not answering."""
    d, seconds, usage = _ask(rgb, FIND_PROMPT.format(target=target), FIND_SCHEMA, "find", url, max_side, 120, timeout_s)
    h, w = rgb.shape[:2]
    box = d.get("box") if d.get("visible", True) else None
    if box is not None:
        x1, x2 = sorted((box[0] * w / 1000.0, box[2] * w / 1000.0))
        y1, y2 = sorted((box[1] * h / 1000.0, box[3] * h / 1000.0))
        box = (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            box = None
    tracing.event("cosmos.find", seconds * 1000, target=target, found=box is not None, tokens=usage.get("completion_tokens"))
    return {"box": box, "grasp": d.get("grasp", ""), "seconds": seconds, "usage": usage}


def list_objects(rgb: np.ndarray, n: int = 5, url: str | None = None, max_side: int = 768, timeout_s: float = 30.0) -> list[str]:
    """Short names of what could be picked up, most prominent first, without repeats."""
    d, seconds, usage = _ask(rgb, LIST_PROMPT.format(n=n), LIST_SCHEMA, "objects", url, max_side, 120, timeout_s)
    out = []
    for name in d.get("objects", []):
        name = " ".join(str(name).lower().replace(".", " ").split())
        if name and name not in out:
            out.append(name)
    out = out[:n]
    tracing.event("cosmos.list", seconds * 1000, objects=out, tokens=usage.get("completion_tokens"))
    return out
