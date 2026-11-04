"""What the demo screen shows of a video before it plays: a thumbnail and the outline of its soundtrack.

Made once per video and kept in data/preview (never in the repo: thumbnails come from other people's videos).
"""

import functools
import json
import subprocess
import threading
from pathlib import Path

import numpy as np

from .detect import duration, frame

OUT = Path("data/preview")
BINS = 1600  # points in the outline; the page draws fewer, keeping the highest of each group
FLOOR_DB = 36.0
_slots = threading.Semaphore(4)  # the library asks for every thumbnail at once


def _file(path: Path, vid: str, ext: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{vid}-{int(path.stat().st_mtime)}.{ext}"


@functools.cache
def length(path: Path) -> float:
    try:
        return duration(path)
    except (subprocess.CalledProcessError, ValueError):  # half-copied file: list it, don't break the library
        return 0.0


def thumb(path: Path, vid: str) -> Path:
    f = _file(path, vid, "jpg")
    if not f.exists():
        with _slots:
            f.write_bytes(frame(path, length(path) * 0.15))
    return f


def wave(path: Path, vid: str) -> list[float]:
    """Peak level per bin, 0–1 on a dB scale relative to the loudest moment: ASMR is mostly quiet, and on a linear
    scale only the bumps would show. 8 kHz is enough for an outline."""
    f = _file(path, vid, "json")
    if f.exists():
        return json.loads(f.read_text())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.abs(np.frombuffer(raw, np.float32))
    peak = np.maximum.reduceat(x, np.linspace(0, len(x), BINS, endpoint=False).astype(int))
    db = 20 * np.log10(peak + 1e-6)
    out = np.clip((db - db.max() + FLOOR_DB) / FLOOR_DB, 0, 1).round(3).tolist()
    f.write_text(json.dumps(out))
    return out
