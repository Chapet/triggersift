"""Labeled clip bank: endpoints for the labeling page.

Clips/labels.jsonl is the source of truth (one JSON clip per line); Clips/<id>.mp4 are cut from it.
"""

import json
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from .profile import taxonomy

CLIPS = Path(os.environ.get("TRIGGERSIFT_CLIPS", "~/Videos/ASMR/Clips")).expanduser()
LABELS = CLIPS / "labels.jsonl"
_lock = threading.Lock()

router = APIRouter()


def youtube_id(name: str) -> str | None:
    m = re.search(r"\[([A-Za-z0-9_-]{11})\]$", Path(name).stem)
    return m.group(1) if m else None


def load() -> list[dict]:
    if not LABELS.exists():
        return []
    return [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]


def _save(clips: list[dict]) -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    tmp = LABELS.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips))
    tmp.replace(LABELS)


def _cut(src: Path, clip: dict) -> None:
    """Accurate cut (re-encode): H.264 + AAC 256k, so any tool can read it."""
    out = CLIPS / f"{clip['id']}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{clip['start']:.3f}", "-i", str(src),
         "-t", f"{clip['end'] - clip['start']:.3f}", "-vf", "scale=-2:min(720\\,ih)",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-b:a", "256k", str(out)],
        check=False)


def _clean(body: dict, video: Path) -> dict:
    known = set(taxonomy())
    triggers = [t for t in body.get("triggers", []) if t in known]
    start, end = float(body["start"]), float(body["end"])
    if end - start < 0.5 or end - start > 60:
        raise HTTPException(400, "a clip must be 0.5–60 s long")
    if not triggers and not body.get("note"):
        raise HTTPException(400, "pick at least one trigger, or describe the sound in the note")
    yt = youtube_id(video.name)
    return {
        "id": f"{yt or re.sub(r'[^A-Za-z0-9]+', '-', video.stem)[:20]}_{round(start * 1000):08d}",
        "video": video.name, "youtube_id": yt,
        "start": round(start, 3), "end": round(end, 3),
        "triggers": triggers,  # all equal: no "main" sound
        "note": body.get("note", "").strip(), "unsure": bool(body.get("unsure")),
    }


def make_routes(video_of) -> APIRouter:
    """video_of(vid) -> Path, shared with the player so both pages list the same library."""

    @router.get("/api/clips")
    def list_clips():
        return load()

    @router.post("/api/clips/{vid}")
    def add_clip(vid: str, body: dict = Body(...)):
        src = video_of(vid)
        clip = _clean(body, src)
        clip["created"] = datetime.now().isoformat(timespec="seconds")
        with _lock:
            clips = [c for c in load() if c["id"] != body.get("replace") and c["id"] != clip["id"]]
            if body.get("replace") and body["replace"] != clip["id"]:
                (CLIPS / f"{body['replace']}.mp4").unlink(missing_ok=True)
            _save(clips + [clip])
        threading.Thread(target=_cut, args=(src, clip), daemon=True).start()
        return clip

    @router.delete("/api/clips/{clip_id}")
    def delete_clip(clip_id: str):
        with _lock:
            clips = load()
            if not any(c["id"] == clip_id for c in clips):
                raise HTTPException(404, "unknown clip")
            _save([c for c in clips if c["id"] != clip_id])
        (CLIPS / f"{clip_id}.mp4").unlink(missing_ok=True)
        return {"ok": True}

    @router.get("/api/clips/{clip_id}/media")
    def clip_media(clip_id: str):
        f = CLIPS / f"{clip_id}.mp4"
        if not f.exists():
            raise HTTPException(404, "clip file not cut yet")
        return FileResponse(f)

    @router.get("/label")
    def label_page():
        return FileResponse(Path(__file__).with_name("web") / "label.html")

    return router

