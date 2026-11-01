"""SAM Audio worker: runs in its own environment (vendor/sam-env), started by the TriggerSift server.

    vendor/sam-env/bin/python app/tools/sam_worker.py job.json

job.json: {"wav": full soundtrack, 48 kHz 16-bit stereo; "segments": [{"start", "end", "phrases", "anchors"}];
           "out_dir": where seg_<k>.npy (float32 [2, samples], the residual = audio without the phrases) go;
           "model": HF id}
Prints "progress <done> <total>" lines. Each ear is separated on its own: SAM Audio works in mono.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

SR = 48_000
WINDOW_S = 20.0  # longest piece per separate() call; longer segments are cut into windows with overlap
OVERLAP_S = 0.5


def read(wav_path: str, start: float, end: float) -> np.ndarray:
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 2 and w.getsampwidth() == 2
        a = max(0, round(start * SR))
        w.setpos(min(a, w.getnframes()))
        raw = w.readframes(max(0, round(end * SR) - a))
    return (np.frombuffer(raw, np.int16).reshape(-1, 2).T / 32768.0).astype(np.float32)


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    from sam_audio import SAMAudio, SAMAudioProcessor
    # no rankers, no span predictor (a 1.4B fp32 PE-A-Frame copy): we always pass the time spans ourselves
    model = SAMAudio.from_pretrained(job["model"], visual_ranker=None, text_ranker=None, span_predictor=None,
                                     dtype=torch.bfloat16).eval().cuda()
    proc = SAMAudioProcessor.from_pretrained(job["model"])
    out = Path(job["out_dir"])
    segs = job["segments"]
    print(f"progress 0 {len(segs)}", flush=True)
    for k, seg in enumerate(segs):
        x = read(job["wav"], seg["start"], seg["end"])
        y = x.copy()
        n, win, hop = x.shape[1], int(WINDOW_S * SR), int((WINDOW_S - OVERLAP_S) * SR)
        for phrase in seg["phrases"]:  # remove one trigger after the other
            acc, wsum = np.zeros_like(y), np.zeros(n, np.float32)
            for a in range(0, max(1, n - int(OVERLAP_S * SR)), hop):
                b = min(n, a + win)
                t0, t1 = a / SR, b / SR
                anchors = [["+", max(0.0, s - t0), min(t1, e) - t0] for s, e in seg["anchors"] if s < t1 and e > t0]
                piece = []
                for ch in range(2):
                    batch = proc(audios=[torch.from_numpy(y[ch:ch + 1, a:b])], descriptions=[phrase],
                                 anchors=[anchors] if anchors else None).to("cuda")
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        r = model.separate(batch, predict_spans=False, reranking_candidates=1)
                    piece.append(r.residual[0].float().cpu().numpy().reshape(-1)[:b - a])
                # triangular weights so overlapping windows crossfade
                w = np.minimum(1.0, np.minimum(np.arange(1, b - a + 1), np.arange(b - a, 0, -1)) / (OVERLAP_S * SR))
                w = w.astype(np.float32) if n > win else np.ones(b - a, np.float32)
                acc[:, a:b] += np.stack(piece) * w
                wsum[a:b] += w
                if b >= n:
                    break
            y = acc / np.maximum(wsum, 1e-6)
        np.save(out / f"seg_{k}.npy", y.astype(np.float32))
        print(f"progress {k + 1} {len(segs)}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
