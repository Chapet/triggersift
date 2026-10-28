"""Score detection against the coarse comment-timestamp labels (../footage/labels.yaml).

Per labelled segment: the dominant detected category (most seconds) should be one of `expect`,
and `cover` = share of the segment covered by events of an expected category.

    uv run python -m triggersift.evaluate [--minutes 0:00,20:00,26:30] [--video "ASMR Tapping"]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from .detect import CACHE, DEFAULT_MODEL, Event, detect, duration

LABELS = Path(__file__).resolve().parents[3] / "footage" / "labels.yaml"
PRICE = {"muse-wonder-1.0": (4.00, 20.00)}  # $ per 1M input / output tokens


def secs(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def _union(spans: list[tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for s, e in sorted(spans):
        if cur is None or s > cur[1]:
            total += (cur[1] - cur[0]) if cur else 0.0
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + ((cur[1] - cur[0]) if cur else 0.0)


def score(segments: list[dict], events: list[Event], total: float, chunks: set[int] | None, chunk_s: float):
    rows = []
    for k, seg in enumerate(segments):
        s = secs(seg["at"])
        e = secs(segments[k + 1]["at"]) if k + 1 < len(segments) else total
        if chunks is not None:  # only score the part of the segment that was analyzed
            analyzed = [(max(s, i * chunk_s), min(e, (i + 1) * chunk_s)) for i in chunks]
            analyzed = [(a, b) for a, b in analyzed if b > a]
            if not analyzed:
                continue
            s, e = min(a for a, _ in analyzed), max(b for _, b in analyzed)
        per_cat: dict[str, list] = defaultdict(list)
        for ev in events:
            a, b = max(s, ev.start), min(e, ev.end)
            if b > a:
                per_cat[ev.trigger.split(".")[0]].append((a, b))
        secs_by_cat = {c: _union(sp) for c, sp in per_cat.items()}
        top = max(secs_by_cat, key=secs_by_cat.get) if secs_by_cat else "-"
        cover = _union([sp for c in seg["expect"] for sp in per_cat.get(c, [])]) / (e - s)
        rows.append((s, e, seg["what"], seg["expect"], top, top in seg["expect"], cover,
                     sorted(secs_by_cat.items(), key=lambda kv: -kv[1])[:4]))
    return rows


def cost(run_dirs: list[Path]) -> tuple[float, int]:
    dollars, n = 0.0, 0
    for d in run_dirs:
        for f in d.glob("chunk*.json"):
            st = json.loads(f.read_text())["stats"]
            pin, pout = PRICE.get(st["model"], (0, 0))
            u = st["usage"] or {}
            dollars += u.get("input_tokens", 0) * pin / 1e6 + u.get("output_tokens", 0) * pout / 1e6
            n += 1
    return dollars, n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", help="comma-separated start times; only the chunks containing them are analyzed")
    p.add_argument("--video", help="substring of the video name")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk", type=float, default=60.0)
    a = p.parse_args()

    labels = yaml.safe_load(LABELS.read_text())["videos"]
    hits = n = 0
    for name, spec in labels.items():
        if a.video and a.video not in name:
            continue
        path = Path(spec["file"]).expanduser()
        total = duration(path)
        only = {int(secs(t) // a.chunk) for t in a.minutes.split(",")} if a.minutes else None
        events = detect(path, a.model, a.chunk, only=only)
        print(f"\n## {name}  ({len(events)} events)")
        for s, e, what, expect, top, ok, cover, cats in score(spec["segments"], events, total, only, a.chunk):
            hits, n = hits + ok, n + 1
            detail = ", ".join(f"{c} {t:.0f}s" for c, t in cats)
            print(f"{'✓' if ok else '✗'} {int(s // 60):2}:{s % 60:02.0f}–{int(e // 60):2}:{e % 60:02.0f} "
                  f"{what[:38]:38} expect {'/'.join(expect):24} top {top:11} cover {cover:4.0%}  [{detail}]")
    dollars, chunks = cost(list(CACHE.glob(f"*-{a.model}-{a.chunk:g}-*")))
    print(f"\nsegments right: {hits}/{n}   |   API cost so far: ${dollars:.2f} over {chunks} chunks")


if __name__ == "__main__":
    main()
