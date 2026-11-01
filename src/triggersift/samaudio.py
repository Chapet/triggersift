"""Remove disliked triggers with SAM Audio (Meta, open weights, local GPU) instead of muting them.

SAM Audio runs in its own environment (vendor/sam-env) as a worker process (tools/sam_worker.py) under a RAM cap:
loading the 3B model naively once peaked at ~24 GB and got the whole session OOM-killed. It needs ~13 GB of GPU,
so PE-A-Frame's model is unloaded first. Tested 2026-11-01: the clipped mouth pop at 0:04 of 0CwxN5mx7l4 went from
0 dB to −23 dB peak with "mouth click", the rest of the audio within 0.5 dB; ~1.2 s per 10 s per ear.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from .detect import _file_id
from .effects import OUT, _audio_codec, output as effect_output
from .plan import Action, Step
from .profile import taxonomy

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / "vendor" / "sam-env" / "bin" / "python"
WORKER = Path(__file__).resolve().parents[2] / "tools" / "sam_worker.py"
MODEL = "facebook/sam-audio-large-tv"
SR = 48_000
PAD_S = 0.3  # context around each section, blended back with a crossfade
FADE_S = 0.02
RAM_CAP = "14G"

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def available() -> str | None:
    """None if usable, else why not."""
    return None if PYTHON.exists() else "needs the SAM Audio environment in vendor/sam-env (see requirements.md)"


def _segments(steps: list[Step], total: float) -> list[dict]:
    segs = []
    for s in steps:
        if s.action != Action.FILTER:
            continue
        a, b = max(0.0, s.start - PAD_S), min(total, s.end + PAD_S)
        phrases = list(dict.fromkeys(taxonomy()[t].sound for t in s.triggers if t in taxonomy()))
        if phrases and b - a > 0.2:
            segs.append({"start": round(a, 3), "end": round(b, 3), "phrases": phrases,
                         "anchors": [[round(s.start - a, 3), round(s.end - a, 3)]]})
    return segs


def key(src: Path, steps: list[Step], fx: str | None) -> str:
    segs = [(round(s.start, 2), round(s.end, 2), sorted(s.triggers)) for s in steps if s.action == Action.FILTER]
    return hashlib.sha1(json.dumps([_file_id(src), segs, fx, MODEL]).encode()).hexdigest()[:12]


def output(src: Path, k: str) -> Path:
    return OUT / f"{_file_id(src)}-sam-{k}{'.webm' if src.suffix.lower() == '.webm' else '.mp4'}"


def _free_gpu() -> None:
    """Unload PE-A-Frame if the server has it in memory (both models together don't fit in 16 GB)."""
    pe = sys.modules.get("triggersift.pe_frame")
    if pe is not None:
        pe.unload()


def _run_worker(jobfile: Path, job: dict) -> tuple[int, str]:
    cmd = [str(PYTHON), str(WORKER), str(jobfile)]
    if shutil.which("systemd-run"):  # hard RAM cap: on overflow only the worker dies, not the session
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={RAM_CAP}", "-p", "MemorySwapMax=0", *cmd]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.startswith("progress "):
            done, total = map(int, line.split()[1:3])
            job["progress"] = 0.1 + 0.85 * done / max(total, 1)
    err = p.stderr.read()
    return p.wait(), err


def _render(src: Path, audio_src: Path, steps: list[Step], total: float, k: str, job: dict) -> None:
    work = OUT / f"sam-{k}"
    work.mkdir(parents=True, exist_ok=True)
    out = output(src, k)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        wav = work / "full.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio_src), "-vn", "-ac", "2", "-ar", str(SR),
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        segs = _segments(steps, total)
        (work / "job.json").write_text(json.dumps({"wav": str(wav), "segments": segs, "out_dir": str(work),
                                                   "model": MODEL}))
        job["progress"] = 0.1
        pe = sys.modules.get("triggersift.pe_frame")
        lock = pe._gpu if pe is not None else threading.Lock()
        with lock:  # no PE-A-Frame analysis on the GPU while SAM Audio runs
            _free_gpu()
            code, err = _run_worker(work / "job.json", job)
        if code != 0:
            raise RuntimeError(f"SAM Audio worker failed: {err[-400:]}")

        with wave.open(str(wav), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).T.astype(np.float32) / 32768
        fade = int(FADE_S * SR)
        for i, seg in enumerate(segs):
            a = round(seg["start"] * SR)
            y = np.load(work / f"seg_{i}.npy")
            b = min(audio.shape[1], a + y.shape[1])
            y = y[:, :b - a]
            w_ = np.ones(b - a, np.float32)  # crossfade in/out against the original
            r = min(fade, (b - a) // 2)
            if r:
                w_[:r] = np.linspace(0, 1, r)
                w_[-r:] = np.linspace(1, 0, r)
            audio[:, a:b] = audio[:, a:b] * (1 - w_) + y * w_
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).T.tobytes()
        enc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
             "-map", "0:v:0?", "-map", "1:a:0", "-c:v", "copy", *_audio_codec(tmp), "-shortest", str(tmp)],
            input=pcm, capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode()[-300:])
        tmp.replace(out)
        job.update(status="done", progress=1.0, sections=len(segs))
    except Exception as e:  # shown in the player instead of an endless "Preparing…"
        tmp.unlink(missing_ok=True)
        job.update(status="error", error=str(e)[-400:])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def start(src: Path, steps: list[Step], total: float, fx: str | None) -> dict:
    k = key(src, steps, fx)
    audio_src = effect_output(src, fx) if fx else src  # chain after an effect if one is selected
    if fx and not audio_src.exists():
        return {"status": "error", "error": "render the effect first", "key": k}
    with _lock:
        job = _jobs.get(k)
        if output(src, k).exists():
            job = _jobs[k] = {"status": "done", "progress": 1.0}
        elif not any(s.action == Action.FILTER for s in steps):
            job = {"status": "done", "progress": 1.0, "nothing": True}
        elif job is None or job["status"] == "error":
            job = _jobs[k] = {"status": "rendering", "progress": 0.0}
            threading.Thread(target=_render, args=(src, audio_src, steps, total, k, job), daemon=True).start()
        return {**job, "key": k}
