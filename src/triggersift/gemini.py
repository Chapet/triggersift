"""Google Gemini as an alternative detector, for comparison: same prompt, frames and JSON schema as Muse.

Key: GEMINI_API_KEY (env var or ../.env). Model: GEMINI_MODEL (default below).
"""

import base64
import copy
import json
import os
import time
from pathlib import Path

import requests

from .detect import Event, _prompt, _schema, audio_chunk, frame, frame_times, to_events

API = "https://generativelanguage.googleapis.com/v1beta"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def api_key() -> str | None:
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _gemini_schema() -> dict:
    """Gemini's JSON schema subset has no additionalProperties."""
    s = copy.deepcopy(_schema())

    def strip(node):
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for v in node.values():
                strip(v)
        elif isinstance(node, list):
            for v in node:
                strip(v)
    strip(s)
    return s


def detect_chunk(path: Path, start: float, length: float) -> tuple[list[Event], dict]:
    key = api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (env var or ~/Projects/triggersift/.env)")
    times = frame_times(length)
    parts = [{"text": _prompt(length, times)}]
    parts += [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(frame(path, start + t)).decode()}}
              for t in times]
    parts.append({"inline_data": {"mime_type": "audio/mp3",
                                  "data": base64.b64encode(audio_chunk(path, start, length)).decode()}})
    t0 = time.monotonic()
    r = requests.post(f"{API}/models/{MODEL}:generateContent", headers={"x-goog-api-key": key}, json={
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": _gemini_schema()},
    }, timeout=600)
    if not r.ok:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:800]}")
    body = r.json()
    text = "".join(p.get("text", "") for c in body.get("candidates", [])[:1]
                   for p in c.get("content", {}).get("parts", []))
    if not text:
        raise RuntimeError(f"Gemini returned no text: {json.dumps(body)[:800]}")
    stats = {"seconds": round(time.monotonic() - t0, 2), "usage": body.get("usageMetadata"), "model": MODEL}
    return to_events(json.loads(text), start, length), stats
