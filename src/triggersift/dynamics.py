"""Our own level-dependent effects, applied block by block (x: float32 [2, samples] at 48 kHz, with context).

They judge every moment against its *surroundings*, not against 0 dBFS: a mouth pop at 0:06 of 0CwxN5mx7l4 is 28 dB
above the median level, but the absolute spike limiter only took 12 dB off it and it still sounded like a pop.
Measured on that video (5 ms frames vs the 90th percentile of the last/next 4 s): ordinary taps stay ≤ ~10 dB above
it (p99), lone pops and knocks reach 20+ dB. So "local loud level" = a high percentile, which repeated taps lift
with them while a lone spike doesn't.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, percentile_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

SR = 48_000
HOP = 240  # 5 ms frames
EPS = 1e-12


def _frames_db(x: np.ndarray) -> np.ndarray:
    """Level per 5 ms frame in dB; stereo linked (the louder channel), so the stereo image doesn't shift."""
    n = x.shape[-1] // HOP * HOP
    p = (x[..., :n].reshape(*x.shape[:-1], -1, HOP) ** 2).mean(-1)
    if p.ndim == 2:
        p = p.max(0)
    return 10 * np.log10(p + EPS)


def _to_samples(g_db: np.ndarray, n: int, smooth_frames: int = 3) -> np.ndarray:
    """Frame gains (dB) → per-sample linear gain. Smoothed so the gain never clicks."""
    g_db = uniform_filter1d(g_db, smooth_frames, mode="nearest")
    t = (np.arange(len(g_db)) + 0.5) * HOP
    return (10 ** (np.interp(np.arange(n), t, g_db) / 20)).astype(np.float32)


def tame_spikes(x: np.ndarray, over_db: float, window_s: float = 8.0, pct: float = 90) -> np.ndarray:
    """Bring anything more than over_db above the local loud level back down to it. Zero-phase, so it reacts
    before the spike starts (no lost attack) and holds 15 ms so the tail of the spike goes down too."""
    e = _frames_db(x)
    ref = percentile_filter(e, pct, int(window_s * SR / HOP) | 1, mode="nearest")
    cut = np.maximum(0.0, e - ref - over_db)
    cut = maximum_filter1d(cut, 3)  # hold over neighbouring frames
    return x * _to_samples(-cut, x.shape[-1])


def _band(x: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    """Zero-phase band; x - band is then exactly the rest, so splitting and summing back is lossless."""
    if lo and hi:
        sos = butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
    elif hi:
        sos = butter(4, hi, "lowpass", fs=SR, output="sos")
    else:
        sos = butter(4, lo, "highpass", fs=SR, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def tame_band(x: np.ndarray, lo: float | None, hi: float | None, over_db: float) -> np.ndarray:
    """Spike tamer on one frequency band only; the rest of the spectrum is left exactly as it was."""
    b = _band(x, lo, hi)
    return x - b + tame_spikes(b, over_db)


def thuds(x: np.ndarray) -> np.ndarray:
    """Desk knocks, mic bumps, chair thuds: bursts below 150 Hz. Taps keep their click, lose the boom."""
    return tame_band(x, None, 150.0, 6.0)


def clicks(x: np.ndarray) -> np.ndarray:
    """Wet mouth clicks, sharp crinkle crackles: bursts in 2–8 kHz."""
    return tame_band(x, 2000.0, 8000.0, 8.0)


def ear_balance(x: np.ndarray, allow_db: float = 6.0, min_width: float = 0.4) -> np.ndarray:
    """When one ear gets more than allow_db louder than the other, narrow the stereo image (mid/side): a sound
    hard in one ear ends up ~7 dB to that side instead of only in it, and that ear is ~3 dB quieter."""
    if x.shape[0] != 2:
        return x
    d = np.abs(_frames_db(x[0]) - _frames_db(x[1]))
    d = uniform_filter1d(d, 20, mode="nearest")  # 100 ms: follow the sound, not every sample
    width = np.clip(1 - (d - allow_db) / 20, min_width, 1.0)
    k = _to_samples(20 * np.log10(width), x.shape[-1], 10)
    m, s = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2 * k
    return np.stack([m + s, m - s])


def clean_gaps(x: np.ndarray, depth_db: float = 10.0) -> np.ndarray:
    """Between triggers, push the remaining background down a further depth_db. Background = within 6 dB of the
    local quiet level (10th percentile over 8 s, which fits the 4 s block context, so no level step at block joins);
    fades in over the next 6 dB, so triggers are never touched."""
    e = uniform_filter1d(_frames_db(x), 8, mode="nearest")  # 40 ms
    floor = percentile_filter(e, 10, int(8 * SR / HOP) | 1, mode="nearest")
    g = -depth_db * np.clip(1 - (e - floor - 6) / 6, 0, 1)
    g = maximum_filter1d(g, 20)  # open fast: keep the gate open 100 ms around anything louder
    return x * _to_samples(g, x.shape[-1], 10)
