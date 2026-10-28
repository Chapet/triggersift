"""Detect ASMR triggers in a video with Muse Spark, chunk by chunk (audio only for now)."""

import base64
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .muse import respond_json
from .profile import taxonomy

DEFAULT_MODEL = "muse-wonder-1.0"  # only model that reliably hears non-speech audio (tested 2026-10-28)
CACHE = Path("runs")


@dataclass
class Event:
    start: float  # seconds in the full video
    end: float
    trigger: str  # "category.sub" from the taxonomy, or "other"
    confidence: float
    note: str  # what the model heard, free text (useful for "other" and for refining the taxonomy)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def audio_chunk(path: Path, start: float, length: float) -> bytes:
    """Mono 64 kb/s MP3 of [start, start+length): small upload, enough detail for trigger sounds."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length), "-i", str(path),
         "-vn", "-ac", "1", "-ar", "32000", "-b:a", "64k", "-f", "mp3", "-"],
        capture_output=True, check=True).stdout


def _schema() -> dict:
    ids = sorted(taxonomy()) + ["other"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["events"],
        "properties": {"events": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start", "end", "trigger", "confidence", "note"],
            "properties": {
                "start": {"type": "number"},
                "end": {"type": "number"},
                "trigger": {"type": "string", "enum": ids},
                "confidence": {"type": "number"},
                "note": {"type": "string"},
            },
        }}},
    }


def _prompt(length: float) -> str:
    lines = "\n".join(f"- {t.id}: {t.label}" for t in taxonomy().values())
    return f"""You are labelling ASMR trigger sounds in a {length:.0f} second audio clip.

List every trigger sound you hear, with start and end time in seconds from the start of this clip.
- One event per continuous stretch of one trigger. If a trigger repeats with gaps under 2 seconds, keep it as one event.
- Sounds can overlap (e.g. whispering over tapping): list each one separately.
- Isolated short sounds (a single kiss, one mouth click) are their own short events; do not skip them.
- Use the most specific trigger id below. Use "other" only if none fits, and describe the sound in "note".
- "note": a few words on what you hear (material, rhythm, intensity).
- "confidence": 0 to 1.
- Times must be accurate to about half a second.

Trigger ids:
{lines}"""


def detect_chunk(path: Path, start: float, length: float, model: str) -> tuple[list[Event], dict]:
    audio = base64.b64encode(audio_chunk(path, start, length)).decode()
    content = [
        {"type": "input_text", "text": _prompt(length)},
        {"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}},
    ]
    out, stats = respond_json(model, content, _schema(), "asmr_triggers")
    events = []
    for e in out["events"]:
        s, t = max(0.0, e["start"]), min(length, e["end"])
        if t <= s:
            continue
        events.append(Event(round(start + s, 2), round(start + t, 2), e["trigger"],
                            e["confidence"], e["note"]))
    return events, stats


def _file_id(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()[:16]


def detect(path: Path, model: str = DEFAULT_MODEL, chunk_s: float = 60.0, log=print) -> list[Event]:
    """All events of a video, cached per (file, model, chunk length, chunk index) in runs/."""
    total = duration(path)
    run = CACHE / f"{_file_id(path)}-{model}-{chunk_s:g}"
    run.mkdir(parents=True, exist_ok=True)
    events: list[Event] = []
    i, start = 0, 0.0
    while start < total - 0.5:
        length = min(chunk_s, total - start)
        cached = run / f"chunk{i:03d}.json"
        if cached.exists():
            data = json.loads(cached.read_text())
        else:
            chunk_events, stats = detect_chunk(path, start, length, model)
            data = {"start": start, "length": length, "stats": stats,
                    "events": [asdict(e) for e in chunk_events]}
            cached.write_text(json.dumps(data, indent=1))
            log(f"chunk {i} [{start:.0f}-{start + length:.0f}s]: {len(chunk_events)} events, "
                f"{stats['seconds']}s, usage={stats['usage']}")
        events += [Event(**e) for e in data["events"]]
        i, start = i + 1, start + chunk_s
    return sorted(events, key=lambda e: e.start)
