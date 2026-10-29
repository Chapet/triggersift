"""Signal-processing effects on a video's soundtrack, rendered once and cached.

The picture is stream-copied (no re-encode); only the audio is processed. Measured 2026-10-29 on the fan video
(0CwxN5mx7l4, 2:00–3:00): ffmpeg's afftdn left the fan background unchanged (−0.3 dB) at any strength, so noise
reduction is our own spectral subtraction: background −10.7 dB (medium) / −13.8 dB (strong), foreground −1.7/−2.1 dB,
0.4 s per minute of stereo audio. adeclick repaired 5.4% of the samples of a mouth-sound minute (Y3zH6IVgVd0 37:00).
"""

import subprocess
import threading
from pathlib import Path

import numpy as np
from scipy.signal import istft, stft

from .detect import _file_id, duration

OUT = Path("runs/fx")
SR = 48_000

# id → (label, ffmpeg audio filter, or ("denoise", strength, floor dB) for our own processing)
EFFECTS = {
    "denoise": ("Background noise reduction (fan, hiss) · medium", ("denoise", 1.5, -20)),
    "denoise_strong": ("Background noise reduction · strong", ("denoise", 2.0, -25)),
    "limiter": ("Spike limiter (lookahead, −12 dB)", "alimiter=limit=0.25:attack=5:release=50:level=disabled"),
    "declick": ("Mouth-click & crackle removal (subtle)", "adeclick"),
}

_jobs: dict[tuple[str, str], dict] = {}
_lock = threading.Lock()


def output(src: Path, fx: str) -> Path:
    # webm keeps its codecs with Opus audio; everything else goes to mp4 with AAC
    ext = ".webm" if src.suffix.lower() == ".webm" else ".mp4"
    return OUT / f"{_file_id(src)}-{fx}{ext}"


def _audio_codec(out: Path) -> list[str]:
    return ["-c:a", "libopus", "-b:a", "160k"] if out.suffix == ".webm" else ["-c:a", "aac", "-b:a", "192k"]


def _ffmpeg_effect(src: Path, af: str, tmp: Path, job: dict, total: float) -> tuple[int, str]:
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0?", "-map", "0:a:0", "-c:v", "copy",
         "-af", af, *_audio_codec(tmp), "-progress", "pipe:1", "-nostats", str(tmp)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.startswith("out_time_us=") and line.strip() != "out_time_us=N/A":
            job["progress"] = min(0.99, int(line.split("=")[1]) / 1e6 / total)
    err = p.stderr.read()
    return p.wait(), err


# ---------- noise reduction: spectral subtraction against the video's steady background ----------
NPERSEG, NOVERLAP = 2048, 1536
BLOCK_S, PAD_S = 30.0, 0.5


def _read(src: Path, start: float, length: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
                          "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T


def _noise_profile(src: Path, total: float, samples: int = 16) -> np.ndarray:
    """Per-channel, per-frequency background level: a low percentile of the magnitude over time, from short
    excerpts spread over the whole video (the fan/hiss is there all the time; triggers come and go)."""
    mags = []
    for k in range(samples):
        x = _read(src, total * (k + 0.5) / samples, 4.0)
        if x.shape[1] > NPERSEG:
            mags.append(np.abs(stft(x, SR, nperseg=NPERSEG, noverlap=NOVERLAP)[2]))
    return np.percentile(np.concatenate(mags, axis=-1), 15, axis=-1, keepdims=True)  # [2, freqs, 1]


def _denoise_block(x: np.ndarray, noise: np.ndarray, strength: float, floor_db: float) -> np.ndarray:
    _, _, Z = stft(x, SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    gain = np.maximum(1 - strength * noise / (np.abs(Z) + 1e-12), 10 ** (floor_db / 20))
    k = np.ones(5) / 5  # smooth the gain over time: avoids warbly "musical noise"
    gain = np.apply_along_axis(lambda g: np.convolve(g, k, "same"), -1, gain)
    _, y = istft(Z * gain, SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    return y[:, :x.shape[1]].astype(np.float32)


def _denoise_effect(src: Path, strength: float, floor_db: float, tmp: Path, job: dict, total: float):
    """Decode the soundtrack once, start to finish (seeking per block is not sample-exact with Opus and left a
    click at every block join), process it in blocks with half a second of context on each side."""
    noise = _noise_profile(src, total)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR),
                            "-f", "f32le", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
         "-map", "0:v:0?", "-map", "1:a:0", "-c:v", "copy", *_audio_codec(tmp), "-shortest", str(tmp)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    block, pad = int(BLOCK_S * SR), int(PAD_S * SR)

    def read() -> np.ndarray:
        raw = dec.stdout.read(block * 8)  # 2 channels × float32
        return np.frombuffer(raw[:len(raw) // 8 * 8], np.float32).reshape(-1, 2).T

    tail, cur, done = np.zeros((2, 0), np.float32), read(), 0
    while cur.shape[1]:
        nxt = read()
        y = _denoise_block(np.concatenate([tail, cur, nxt[:, :pad]], axis=1), noise, strength, floor_db)
        enc.stdin.write(np.ascontiguousarray(y[:, tail.shape[1]:tail.shape[1] + cur.shape[1]].T).tobytes())
        done += cur.shape[1]
        job["progress"] = min(0.99, done / SR / total)
        tail, cur = cur[:, -pad:], nxt
    dec.wait()
    enc.stdin.close()
    err = enc.stderr.read().decode()
    return enc.wait(), err


def _render(src: Path, fx: str, job: dict) -> None:
    out = output(src, fx)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    total = duration(src)
    spec = EFFECTS[fx][1]
    try:
        if isinstance(spec, tuple):
            code, err = _denoise_effect(src, spec[1], spec[2], tmp, job, total)
        else:
            code, err = _ffmpeg_effect(src, spec, tmp, job, total)
    except Exception as e:  # report in the UI instead of hanging at "Preparing…"
        code, err = 1, str(e)
    if code == 0:
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
