"""Extra training data from FSD50K (CC-BY 4.0, Freesound clips with AudioSet labels), mapped onto our categories.

    uv run --extra local python -m triggersift.fsd50k     # download the selected clips + cache PE embeddings

Clips go to ../datasets/fsd50k/clips/, labels from the official dev/eval CSVs (Fhrozen/FSD50k on Hugging Face).
Many FSD50K clips are general sound effects, not close-mic ASMR: they add variety, not ASMR style.
"""

import csv
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "datasets" / "fsd50k"
REPO = "Fhrozen/FSD50k"
EMB = Path("runs/bank/fsd")
PER_CLASS = 120
MAX_S = 10.0  # only the first seconds of long clips

# FSD50K class → our sound class. Left out: "Scratching_(performance_technique)" (DJ scratching), parents like "Hands".
MAP = {
    "Tap": "tapping_sharp", "Knock": "tapping_soft",
    "Crumpling_and_crinkling": "crinkling",
    "Typing": "clicks", "Computer_keyboard": "clicks", "Finger_snapping": "clicks",
    "Writing": "rubbing",
    "Chewing_and_mastication": "mouth",
    "Whispering": "whisper", "Male_speech_and_man_speaking": "spoken", "Female_speech_and_woman_speaking": "spoken",
    "Breathing": "breathing", "Sigh": "breathing", "Thump_and_thud": "handling",
    "Rain": "nature", "Thunder": "nature", "Fire": "nature",
    "Mechanical_fan": "steady_noise", "Hiss": "steady_noise", "Buzz": "steady_noise",
    "Pour": "liquid", "Drip": "liquid", "Splash_and_splatter": "liquid", "Trickle_and_dribble": "liquid",
    "Gurgling": "liquid",
    "Scissors": "crunch",
}


def select(seed: int = 0) -> list[dict]:
    """Up to PER_CLASS clips per mapped FSD50K class; each clip keeps every category its labels map to."""
    rows = []
    for split in ("dev", "eval"):
        for r in csv.DictReader(open(ROOT / f"{split}.csv")):
            labels = r["labels"].split(",")
            cats = sorted({MAP[l] for l in labels if l in MAP})
            if cats:
                rows.append({"fname": r["fname"], "split": split, "labels": labels, "cats": cats})
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked, per = {}, {c: 0 for c in MAP}
    for r in rows:
        want = [l for l in r["labels"] if l in MAP and per[l] < PER_CLASS]
        if want:
            picked[r["fname"]] = r
            for l in r["labels"]:
                if l in per:
                    per[l] += 1
    return list(picked.values())


def path(r: dict) -> Path:
    return ROOT / "clips" / r["split"] / f"{r['fname']}.wav"


def download(rows: list[dict], workers: int = 16) -> None:
    from huggingface_hub import hf_hub_download

    def one(r):
        if not path(r).exists():
            hf_hub_download(REPO, f"clips/{r['split']}/{r['fname']}.wav", repo_type="dataset", local_dir=ROOT)

    todo = [r for r in rows if not path(r).exists()]
    print(f"{len(rows)} clips selected, {len(todo)} to download")
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(one, todo), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}")


def embeds(rows: list[dict]) -> dict[str, np.ndarray]:
    from .pe_frame import frame_embeds, load_audio
    EMB.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, r in enumerate(rows, 1):
        f = EMB / f"{r['fname']}.npy"
        if not f.exists():
            emb, _ = frame_embeds(load_audio(path(r), 0.0, MAX_S))
            np.save(f, emb.astype(np.float16))
            if i % 500 == 0:
                print(f"  embedded {i}/{len(rows)}")
        out[r["fname"]] = np.load(f).astype(np.float32)
    return out


def main() -> None:
    rows = select()
    download(rows)
    embeds(rows)
    from collections import Counter
    print("clips per category:", dict(Counter(c for r in rows for c in r["cats"]).most_common()))


if __name__ == "__main__":
    main()
