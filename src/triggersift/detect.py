"""Detect ASMR triggers in a video, chunk by chunk, with a choice of detector (see DETECTORS)."""

import base64
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
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


FRAMES_PER_CHUNK = 6


def frame(path: Path, t: float) -> bytes:
    """One 512 px wide JPEG at time t."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
         "-vf", "scale=512:-2", "-q:v", "5", "-f", "image2", "-c:v", "mjpeg", "-"],
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


def _prompt(length: float, frame_times: list[float]) -> str:
    lines = "\n".join(f"- {t.id}: {t.label} ({t.covers})" for t in taxonomy().values())
    times = ", ".join(f"{t:.0f}s" for t in frame_times)
    return f"""You are labelling ASMR trigger sounds in a {length:.0f} second audio clip.
Video frames from the same clip are attached, in order, taken at {times}.
- The AUDIO decides the kind of sound (tapping vs. crinkling vs. scratching) and the timing. Six frames miss most of
  what happens: a box can be wrapped in plastic, a hand can hide a wrapper. If it sounds crinkly, it is crinkling.
- The FRAMES help with the object or material (a keyboard vs. a box), and with the source: no mouth sounds or
  whispering unless a mouth plausibly makes them (a click while only hands are visible is likely hands/nails).
- Wet pops, squelches and sloshing while a jar, bottle, lotion or gel is being handled are liquid sounds, not mouth
  sounds. Mouth sounds need the mouth: lips/tongue moving, breath, or no liquid object in view.

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


def frame_times(length: float) -> list[float]:
    return [length * (k + 0.5) / FRAMES_PER_CHUNK for k in range(FRAMES_PER_CHUNK)]


def to_events(out: dict, start: float, length: float) -> list[Event]:
    """LLM JSON output (times relative to the chunk) → events in video time."""
    events = []
    for e in out["events"]:
        s, t = max(0.0, e["start"]), min(length, e["end"])
        if t > s:
            events.append(Event(round(start + s, 2), round(start + t, 2), e["trigger"], e["confidence"], e["note"]))
    return events


def _muse_chunk(path: Path, start: float, length: float, model: str) -> tuple[list[Event], dict]:
    audio = base64.b64encode(audio_chunk(path, start, length)).decode()
    times = frame_times(length)
    content = [{"type": "input_text", "text": _prompt(length, times)}]
    content += [{"type": "input_image",
                 "image_url": "data:image/jpeg;base64," + base64.b64encode(frame(path, start + t)).decode()}
                for t in times]
    content.append({"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}})
    out, stats = respond_json(model, content, _schema(), "asmr_triggers")
    return to_events(out, start, length), stats


# id → (label for the UI, where it runs)
DETECTORS = {
    "muse-wonder-1.0": ("Muse Wonder · Meta API · ~$0.07/min", "api"),
    "pe-a-frame-large": ("PE-A-Frame · Meta, local GPU · free", "local"),
    "gemini": ("Gemini · Google API", "api"),
}


def detect_chunk(path: Path, start: float, length: float, model: str) -> tuple[list[Event], dict]:
    if model.startswith("pe-a-frame"):
        from . import pe_frame  # torch: only loaded when this detector is used
        return pe_frame.detect_chunk(path, start, length)
    if model.startswith("gemini"):
        from . import gemini
        return gemini.detect_chunk(path, start, length)
    return _muse_chunk(path, start, length, model)


_ID_CACHE: dict[tuple, str] = {}


def _file_id(path: Path) -> str:
    """Content hash (first 16 hex of sha1), memoized per (path, size, mtime): videos are ~1 GB."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _ID_CACHE:
        h = hashlib.sha1()
        with path.open("rb") as f:
            while block := f.read(1 << 20):
                h.update(block)
        _ID_CACHE[key] = h.hexdigest()[:16]
    return _ID_CACHE[key]


@dataclass
class Chunk:
    index: int
    start: float
    length: float


def chunks(total: float, chunk_s: float = 60.0) -> list[Chunk]:
    return [Chunk(i, i * chunk_s, min(chunk_s, total - i * chunk_s))
            for i in range(math.ceil((total - 0.5) / chunk_s))]


def run_dir(path: Path, model: str = DEFAULT_MODEL, chunk_s: float = 60.0) -> Path:
    """Cache folder for one (video, model, chunk length, prompt version): editing the prompt re-runs detection."""
    version = hashlib.sha1((_prompt(chunk_s, []) + json.dumps(_schema()) + str(FRAMES_PER_CHUNK)).encode()).hexdigest()[:8]
    run = CACHE / f"{_file_id(path)}-{model}-{chunk_s:g}-{version}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def cached_chunk(run: Path, c: Chunk) -> list[Event] | None:
    f = run / f"chunk{c.index:03d}.json"
    return [Event(**e) for e in json.loads(f.read_text())["events"]] if f.exists() else None


def analyze_chunk(run: Path, path: Path, c: Chunk, model: str = DEFAULT_MODEL, log=print) -> list[Event]:
    """Events of one chunk, from the cache or from Muse."""
    if (events := cached_chunk(run, c)) is not None:
        return events
    events, stats = detect_chunk(path, c.start, c.length, model)
    data = {"start": c.start, "length": c.length, "stats": stats, "events": [asdict(e) for e in events]}
    (run / f"chunk{c.index:03d}.json").write_text(json.dumps(data, indent=1))
    log(f"chunk {c.index} [{c.start:.0f}-{c.start + c.length:.0f}s]: {len(events)} events, "
        f"{stats['seconds']}s, usage={stats['usage']}")
    return events


def detect(path: Path, model: str = DEFAULT_MODEL, chunk_s: float = 60.0, log=print,
           only: set[int] | None = None, workers: int = 4) -> list[Event]:
    """All events of a video (or of the chunk indices in `only`), cached per chunk in runs/."""
    run = run_dir(path, model, chunk_s)
    todo = [c for c in chunks(duration(path), chunk_s) if only is None or c.index in only]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda c: analyze_chunk(run, path, c, model, log), todo))
    return sorted((e for r in results for e in r), key=lambda e: e.start)
