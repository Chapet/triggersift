"""Convert v1 object-level labels ("tapping.glass") to the v2 sound classes ("tapping_sharp").

    uv run python -m triggersift.migrate          # dry run: prints what changes
    uv run python -m triggersift.migrate --write  # rewrites labels.jsonl, the profile and footage/labels.yaml

Idempotent: v2 ids pass through unchanged. Backups were made by hand before the first run (2026-11-02):
labels.before-sound-classes.jsonl, data/profile.v1.json, ../taxonomy.v1-objects.yaml.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .labeling import LABELS
from .profile import taxonomy

SHARP = {"nail_tapping", "wood", "glass", "plastic", "metal"}  # user: hard surfaces & nails = sharp, the rest = soft

OLD = {
    "hands": {"rubbing": "rubbing", "fluttering": "brushing", "snapping": "clicks", "tapping": "tapping_soft",
              "lotion_hands": "liquid"},
    "tapping": {**{k: "tapping_sharp" for k in SHARP},
                "finger_tapping": "tapping_soft", "leather": "tapping_soft", "cardboard": "tapping_soft",
                "mic": "brushing"},
    "scratching": {"fabric": "rubbing", "hard_surface": "rubbing", "beard": "rubbing", "foam": "rubbing",
                   "mic_cover": "brushing"},
    "crinkling": "crinkling",
    "mouth": {"breathing": "breathing", "*": "mouth"},
    "voice": {"whisper": "whisper", "unintelligible": "whisper", "*": "spoken"},
    "brushing": "brushing",
    "liquid": "liquid",
    "mic_texture": "brushing",
    "clicking": "clicks",
    "cutting": "crunch",
    "ambient": {"rain": "nature", "thunder": "nature", "fire": "nature", "outside": "nature", "music": "music",
                "room": "handling", "*": "steady_noise"},
    "noise": {"breathing": "breathing", "*": "handling"},
}
# whole v1 categories (profile keys, coarse ground truth) → every class they can become
OLD_CATEGORY = {cat: sorted({v for v in (m.values() if isinstance(m, dict) else [m])}) for cat, m in OLD.items()}


RENAMED = {"mic_contact": "brushing"}  # v2 classes merged later (2026-11-03)


def convert(tid: str) -> str | None:
    if tid in taxonomy() or tid == "other":
        return tid
    if tid in RENAMED:
        return RENAMED[tid]
    cat, _, sub = tid.partition(".")
    m = OLD.get(cat)
    if m is None:
        return None
    return m if isinstance(m, str) else m.get(sub, m.get("*"))


def labels(write: bool) -> None:
    clips = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    moved, lost = Counter(), Counter()
    for c in clips:
        new = []
        for t in c["triggers"]:
            n = convert(t)
            if n is None:
                lost[t] += 1
            else:
                moved[f"{t} → {n}"] += t != n
                if n not in new:
                    new.append(n)
        c["triggers"] = new
    print(f"{len(clips)} clips; per class now:", dict(Counter(t for c in clips for t in c["triggers"]).most_common()))
    if lost:
        print("dropped (unknown):", dict(lost))
    if write:
        LABELS.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips))


def profile(path: Path, write: bool) -> None:
    """A v1 subcategory setting moves to its class; a whole v1 category setting goes to every class it covers,
    unless that class already got its own setting."""
    if not path.exists():
        return
    old = json.loads(path.read_text())
    new: dict[str, str] = {}
    for k, v in old.items():
        if "." in k or k in taxonomy():
            if (n := convert(k)) and v != "neutral":
                new[n] = v
    for k, v in old.items():
        if k in OLD_CATEGORY and v != "neutral":
            for n in OLD_CATEGORY[k]:
                new.setdefault(n, v)
    print(f"profile {old} → {new}")
    if write:
        path.write_text(json.dumps(new, indent=2) + "\n")


def footage(path: Path, write: bool) -> None:
    """Coarse ground truth: `expect: [hands, tapping]` → the classes those categories became."""
    if not path.exists():
        return
    text = path.read_text()

    def repl(m: re.Match) -> str:
        out = []
        for c in (x.strip() for x in m.group(1).split(",")):
            for n in OLD_CATEGORY.get(c, [c]):
                if n not in out:
                    out.append(n)
        return f"expect: [{', '.join(out)}]"

    new = re.sub(r"expect: \[([^\]]*)\]", repl, text)
    print(f"{path}: {sum(a != b for a, b in zip(text.splitlines(), new.splitlines()))} lines changed")
    if write:
        path.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    labels(a.write)
    profile(Path("data/profile.json"), a.write)
    footage(root / "footage" / "labels.yaml", a.write)


if __name__ == "__main__":
    main()
