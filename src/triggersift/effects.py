"""Signal-processing effects on a video's soundtrack (ffmpeg filters), rendered once and cached.

The picture is stream-copied (no re-encode), only the audio is filtered, so a 40 min video takes ~10–60 s.
Timings measured 2026-10-29 on 60 s of audio: adeclick 1.2 s, loudnorm 0.7 s, afftdn 0.15 s, alimiter 0.1 s.
"""

import subprocess
import threading
from pathlib import Path

from .detect import _file_id, duration

OUT = Path("runs/fx")

# id → (label, ffmpeg audio filter)
EFFECTS = {
    "declick": ("Mouth-click & crackle removal", "adeclick"),
    "limiter": ("Spike limiter (lookahead, −12 dB)", "alimiter=limit=0.25:attack=5:release=50:level=disabled"),
    "loudnorm": ("Even loudness (EBU R128)", "loudnorm=I=-23:TP=-2:LRA=11"),
    "denoise": ("Background noise reduction (fan, hiss)", "afftdn=nr=12:nf=-40:tn=1"),
    "dehum": ("Hum removal (50/60 Hz)", ",".join(f"bandreject=f={f}:width_type=q:w=20"
                                              for f in (50, 100, 150, 60, 120, 180))),
}

_jobs: dict[tuple[str, str], dict] = {}
_lock = threading.Lock()


def output(src: Path, fx: str) -> Path:
    # webm keeps its codecs with Opus audio; everything else goes to mp4 with AAC
    ext = ".webm" if src.suffix.lower() == ".webm" else ".mp4"
    return OUT / f"{_file_id(src)}-{fx}{ext}"


def _render(src: Path, fx: str, job: dict) -> None:
    out = output(src, fx)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    audio = ["-c:a", "libopus", "-b:a", "160k"] if out.suffix == ".webm" else ["-c:a", "aac", "-b:a", "192k"]
    total = duration(src)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0?", "-map", "0:a:0", "-c:v", "copy",
         "-af", EFFECTS[fx][1], *audio, "-progress", "pipe:1", "-nostats", str(tmp)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.startswith("out_time_us=") and line.strip() != "out_time_us=N/A":
            job["progress"] = min(0.99, int(line.split("=")[1]) / 1e6 / total)
    err = p.stderr.read()
    if p.wait() == 0:
        tmp.replace(out)
        job.update(status="done", progress=1.0)
    else:
        tmp.unlink(missing_ok=True)
        job.update(status="error", error=err[-300:])


def start(src: Path, fx: str) -> dict:
    if fx not in EFFECTS:
        raise KeyError(fx)
    OUT.mkdir(parents=True, exist_ok=True)
    key = (str(src), fx)
    with _lock:
        job = _jobs.get(key)
        if output(src, fx).exists():
            job = _jobs[key] = {"status": "done", "progress": 1.0}
        elif job is None or job["status"] == "error":
            job = _jobs[key] = {"status": "rendering", "progress": 0.0}
            threading.Thread(target=_render, args=(src, fx, job), daemon=True).start()
        return dict(job)
