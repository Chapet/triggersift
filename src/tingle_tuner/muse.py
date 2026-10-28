"""Minimal Meta Model API client (Responses API, structured JSON output)."""

import json
import os
import time
from pathlib import Path

import requests

API = "https://api.meta.ai/v1"


def api_key() -> str:
    if key := os.environ.get("MODEL_API_KEY"):
        return key
    # .env in the cwd, or next to the repo (kept outside it on purpose)
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("MODEL_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("MODEL_API_KEY not set (env var or .env)")


def respond_json(model: str, content: list[dict], schema: dict, name: str,
                 max_output_tokens: int = 16000) -> tuple[dict, dict]:
    """One Responses call with a JSON schema. Returns (parsed output, stats)."""
    t0 = time.monotonic()
    r = requests.post(
        f"{API}/responses",
        headers={"Authorization": f"Bearer {api_key()}"},
        json={
            "model": model,
            "max_output_tokens": max_output_tokens,
            "input": [{"type": "message", "role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        },
        timeout=600,
    )
    if not r.ok:
        raise RuntimeError(f"Muse HTTP {r.status_code}: {r.text[:800]}")
    body = r.json()
    text = "".join(
        part.get("text", "")
        for item in body.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    )
    if not text:
        raise RuntimeError(f"Muse returned no text (status={body.get('status')}): {json.dumps(body)[:800]}")
    stats = {"seconds": round(time.monotonic() - t0, 2), "usage": body.get("usage"), "model": model}
    return json.loads(text), stats
