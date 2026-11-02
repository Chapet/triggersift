"""Local trigger detection with Meta's PE-A-Frame: every taxonomy sound phrase scored every 40 ms.

Needs `uv sync --extra local` and the perception_models repo in ../vendor (see requirements.md).
"""

import importlib.util
import subprocess
import sys
import types
from functools import cache
from pathlib import Path

import numpy as np

from .detect import Event
from .profile import taxonomy

VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "perception_models"
MODEL = "pe-a-frame-large"
SR = 48_000
WINDOW_S = 15.0  # audio per forward pass; 30 s peaked at ~7 GB of GPU memory


def _import_pe():
    """Import Meta's perception_models code with two optional dependencies stubbed out:
    torchcodec (file decoding; we pass waveforms) and xformers (LLM attention; unused by the audio encoder).
    The stubs live only during the import, so transformers' own availability checks never see them."""
    sys.path.insert(0, str(VENDOR))
    stubs = []
    for names, attrs, probe in ((("torchcodec", "torchcodec.decoders"), ("AudioDecoder", "VideoDecoder"), "torchcodec"),
                                (("xformers", "xformers.ops"), ("AttentionBias", "fmha"), "xformers")):
        if importlib.util.find_spec(probe) is not None:
            continue
        for name in names:
            m = types.ModuleType(name)
            for a in attrs:
                setattr(m, a, None)
            sys.modules[name] = m
            stubs.append(name)
    try:
        from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return PEAudioFrame, PEAudioFrameTransform


@cache
def _model(name: str = MODEL):
    import torch
    PEAudioFrame, PEAudioFrameTransform = _import_pe()
    model = PEAudioFrame.from_config(name, pretrained=True).to("cuda").eval()
    transform = PEAudioFrameTransform.from_config(name)
    return model, transform, torch


def load_audio(path: Path, start: float = 0.0, length: float | None = None) -> np.ndarray:
    args = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path)]
    if length:
        args += ["-t", str(length)]
    raw = subprocess.run(args + ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


@cache
def _text_embeds(phrases: tuple[str, ...], name: str = MODEL):
    model, transform, torch = _model(name)
    inputs = transform(text=list(phrases)).to("cuda")
    with torch.inference_mode():
        out = model._get_text_output(inputs["input_ids"], inputs.get("attention_mask"))
        return model.text_head(out.pooler_output)


def frame_embeds(audio: np.ndarray, name: str = MODEL) -> tuple[np.ndarray, float]:
    """Frame embeddings [frames, D] (float32, one per 40 ms) and the frame hop in seconds."""
    model, transform, torch = _model(name)
    dac = model.config.audio_model.dac_vae_encoder
    hop = dac.hop_length / dac.sampling_rate
    out = []
    step = int(WINDOW_S * SR)
    for i in range(0, len(audio), step):
        piece = audio[i:i + step]
        if len(piece) < SR // 10:
            break
        inputs = transform(audio=[torch.from_numpy(piece.copy())[None]], sampling_rate=SR).to("cuda")  # [channels, samples]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            a = model.audio_model(inputs["input_values"], inputs.get("padding_mask"))
            emb = model.audio_head(a.last_hidden_state)[0]  # [T, D]
        n = round(len(piece) / SR / hop)  # drop padding frames
        out.append(emb[:n].float().cpu().numpy())
    return np.concatenate(out), hop


def probs_from_embeds(emb: np.ndarray, phrases: tuple[str, ...], name: str = MODEL) -> np.ndarray:
    """Zero-shot probabilities [frames, phrases]: frame embeddings against the text embeddings of the phrases."""
    model, _, torch = _model(name)
    text = _text_embeds(phrases, name).float()
    with torch.inference_mode():
        logits = torch.from_numpy(emb).cuda() @ text.T * model.logit_scale.float() + model.logit_bias.float()
        return logits.sigmoid().cpu().numpy()


def frame_probs(audio: np.ndarray, phrases: tuple[str, ...], name: str = MODEL) -> tuple[np.ndarray, float]:
    """Probabilities [frames, phrases] and the frame hop in seconds."""
    emb, hop = frame_embeds(audio, name)
    return probs_from_embeds(emb, phrases, name), hop


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def events_from_probs(probs: np.ndarray, hop: float, ids: list[str], offset: float = 0.0,
                      threshold: float = 0.3, min_s: float = 0.2, gap_s: float = 0.5) -> list[Event]:
    events = []
    for k, tid in enumerate(ids):
        mask = probs[:, k] > threshold
        # close gaps shorter than gap_s
        for s, e in _spans(~mask):
            if s > 0 and e < len(mask) and (e - s) * hop < gap_s:
                mask[s:e] = True
        for s, e in _spans(mask):
            if (e - s) * hop >= min_s:
                events.append(Event(round(offset + s * hop, 2), round(offset + e * hop, 2), tid,
                                    round(float(probs[s:e, k].mean()), 3), "pe-a-frame"))
    return sorted(events, key=lambda e: e.start)


def phrases() -> tuple[list[str], tuple[str, ...]]:
    t = taxonomy()
    return list(t), tuple(x.sound for x in t.values())


CATEGORY_THRESHOLD = 0.6  # zero-shot, untuned: to calibrate on the clip bank
_gpu = __import__("threading").Lock()


def category_events(probs: np.ndarray, hop: float, ids: list[str], offset: float,
                    threshold: float = CATEGORY_THRESHOLD, min_s: float = 0.3, gap_s: float = 0.5) -> list[Event]:
    """One event per active stretch of a main category, named after its best-scoring subcategory.

    Zero-shot scores of sibling subcategories move together (all crinkling.* high at once), so the category is
    the unit that turns on/off, and the subcategory is only the best guess inside it."""
    cats: dict[str, list[int]] = {}
    for k, tid in enumerate(ids):
        cats.setdefault(tid.split(".")[0], []).append(k)
    events = []
    for cat, ks in cats.items():
        score = probs[:, ks].max(axis=1)
        mask = score > threshold
        for s, e in _spans(~mask):
            if s > 0 and e < len(mask) and (e - s) * hop < gap_s:
                mask[s:e] = True
        for s, e in _spans(mask):
            if (e - s) * hop < min_s:
                continue
            means = probs[s:e, ks].mean(axis=0)
            best = ks[int(means.argmax())]
            runner = sorted(((float(m), ids[k]) for m, k in zip(means, ks)), reverse=True)[1:3]
            note = "also: " + ", ".join(f"{t.split('.')[-1]} {m:.2f}" for m, t in runner) if runner else ""
            events.append(Event(round(offset + s * hop, 2), round(offset + e * hop, 2), ids[best],
                                round(float(means.max()), 3), note))
    return sorted(events, key=lambda e: e.start)


def detect_chunk(path: Path, start: float, length: float) -> tuple[list[Event], dict]:
    import time
    t0 = time.monotonic()
    ids, ph = phrases()
    audio = load_audio(path, start, length)
    with _gpu:
        try:
            probs, hop = frame_probs(audio, ph)
        finally:
            import torch
            torch.cuda.empty_cache()  # hand memory back between chunks
    return (category_events(probs, hop, ids, start),
            {"seconds": round(time.monotonic() - t0, 2), "usage": None, "model": MODEL})


def unload() -> None:
    """Free the GPU (SAM Audio needs ~13 GB of the 16). The model reloads on the next analysis (~7 s)."""
    import gc
    _model.cache_clear()
    _text_embeds.cache_clear()
    gc.collect()
    import torch
    torch.cuda.empty_cache()
