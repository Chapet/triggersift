"""Score detectors on the labeled clip bank (~/Videos/ASMR/Clips/labels.jsonl), per main category.

    uv run python -m triggersift.bank_eval                     # PE-A-Frame zero-shot (free) + Muse Wonder (cached)
    uv run python -m triggersift.bank_eval --no-wonder

Ground truth per clip = the set of main categories among all its labels (main + secondary).
Only categories with at least --min clips are scored. PE frame embeddings are cached in runs/bank/pe/ for training.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from .detect import detect_chunk, duration
from .labeling import CLIPS, load
from .profile import taxonomy

OUT = Path("runs/bank")
PE_THRESHOLD = 0.6  # same as the player's zero-shot setting
WONDER = "muse-wonder-1.0"
PRICE = (4.00, 20.00)  # Wonder $ per 1M input / output tokens


def cat(trigger: str) -> str:
    return trigger.split(".")[0]


def truth(clip: dict) -> set[str]:
    return {cat(t) for t in clip["triggers"]}


# ---------- PE-A-Frame ----------
def pe_embeds(clips: list[dict]) -> dict[str, np.ndarray]:
    from .pe_frame import frame_embeds, load_audio
    d = OUT / "pe"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in clips:
        f = d / f"{c['id']}.npy"
        if not f.exists():
            emb, _ = frame_embeds(load_audio(CLIPS / f"{c['id']}.mp4"))
            np.save(f, emb.astype(np.float16))
        out[c["id"]] = np.load(f).astype(np.float32)
    return out


def pe_scores(clips: list[dict], cats: list[str]) -> dict[str, dict[str, float]]:
    """Per clip and category: share of frames where the category's best subcategory beats the threshold,
    and the mean of that best score (used for ranking)."""
    from .pe_frame import probs_from_embeds
    ids = list(taxonomy())
    phrases = tuple(taxonomy()[i].sound for i in ids)
    emb = pe_embeds(clips)
    scores = {}
    for c in clips:
        p = probs_from_embeds(emb[c["id"]], phrases)
        scores[c["id"]] = {k: float(p[:, [i for i, t in enumerate(ids) if cat(t) == k]].max(axis=1).mean())
                           for k in cats}
    return scores


# ---------- Muse Wonder ----------
def wonder_results(clips: list[dict], workers: int = 6) -> dict[str, dict]:
    d = OUT / "wonder"
    d.mkdir(parents=True, exist_ok=True)

    def one(c):
        f = d / f"{c['id']}.json"
        if f.exists():
            return c["id"], json.loads(f.read_text())
        path = CLIPS / f"{c['id']}.mp4"
        try:
            events, stats = detect_chunk(path, 0.0, duration(path), WONDER)
            data = {"events": [asdict(e) for e in events], "stats": stats}
        except Exception as e:  # keep going; failed clips are reported, not cached
            return c["id"], {"error": str(e)[:300]}
        f.write_text(json.dumps(data, indent=1))
        return c["id"], data

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, clips))


def wonder_cost(results: dict[str, dict]) -> float:
    total = 0.0
    for r in results.values():
        u = (r.get("stats") or {}).get("usage") or {}
        total += u.get("input_tokens", 0) * PRICE[0] / 1e6 + u.get("output_tokens", 0) * PRICE[1] / 1e6
    return total


# ---------- metrics ----------
def auc(pos: list[float], neg: list[float]) -> float | None:
    """Chance that a random positive clip scores above a random negative one (0.5 = guessing, 1 = perfect)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=10, help="minimum clips per category to score it")
    ap.add_argument("--no-wonder", action="store_true")
    a = ap.parse_args()

    clips = [c for c in load() if (CLIPS / f"{c['id']}.mp4").exists() and c["triggers"]]
    counts = Counter(k for c in clips for k in truth(c))
    cats = sorted((k for k, n in counts.items() if n >= a.min), key=lambda k: -counts[k])
    skipped = {k: n for k, n in counts.items() if n < a.min}
    lines = [f"# Clip bank evaluation {date.today()}", "",
             f"{len(clips)} clips from {len({c['video'] for c in clips})} videos. "
             f"Scored categories (≥ {a.min} clips): {', '.join(f'{k} {counts[k]}' for k in cats)}. "
             f"Skipped: {', '.join(f'{k} {n}' for k, n in skipped.items()) or 'none'}.", ""]

    pe = pe_scores(clips, cats)
    wonder = {} if a.no_wonder else wonder_results(clips)
    failed = [k for k, r in wonder.items() if "error" in r]

    def wonder_cats(cid: str) -> set[str]:
        return {cat(e["trigger"]) for e in wonder.get(cid, {}).get("events", [])}

    lines += ["## Per category", "",
              "| category | clips | PE AUC | PE P / R / F1 @0.6 | Wonder P / R / F1 |",
              "|---|---|---|---|---|"]
    for k in cats:
        pos = [pe[c["id"]][k] for c in clips if k in truth(c)]
        neg = [pe[c["id"]][k] for c in clips if k not in truth(c)]
        a_ = auc(pos, neg)
        tp = sum(1 for c in clips if k in truth(c) and pe[c["id"]][k] > PE_THRESHOLD)
        fp = sum(1 for c in clips if k not in truth(c) and pe[c["id"]][k] > PE_THRESHOLD)
        pe_prf = prf(tp, fp, len(pos) - tp)
        row = f"| {k} | {counts[k]} | {a_:.2f} | {pe_prf[0]:.2f} / {pe_prf[1]:.2f} / {pe_prf[2]:.2f} |"
        if wonder:
            ok = [c for c in clips if c["id"] in wonder and "error" not in wonder[c["id"]]]
            wtp = sum(1 for c in ok if k in truth(c) and k in wonder_cats(c["id"]))
            wfp = sum(1 for c in ok if k not in truth(c) and k in wonder_cats(c["id"]))
            wfn = sum(1 for c in ok if k in truth(c) and k not in wonder_cats(c["id"]))
            w = prf(wtp, wfp, wfn)
            row += f" {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} |"
        else:
            row += " – |"
        lines.append(row)

    # main sound: does the detector's top category match the clip's main label?
    scored = [c for c in clips if c["main"] and cat(c["main"]) in cats]
    pe_main = sum(1 for c in scored if max(cats, key=lambda k: pe[c["id"]][k]) == cat(c["main"]))
    lines += ["", "## Main sound (top category = the clip's ★ label)", "",
              f"- PE-A-Frame zero-shot: {pe_main}/{len(scored)} ({pe_main / len(scored):.0%})"]
    if wonder:
        def wonder_top(cid):
            secs = Counter()
            for e in wonder[cid].get("events", []):
                if cat(e["trigger"]) in cats:
                    secs[cat(e["trigger"])] += e["end"] - e["start"]
            return secs.most_common(1)[0][0] if secs else None
        ok = [c for c in scored if "error" not in wonder.get(c["id"], {"error": 1})]
        w_main = sum(1 for c in ok if wonder_top(c["id"]) == cat(c["main"]))
        lines.append(f"- Muse Wonder: {w_main}/{len(ok)} ({w_main / len(ok):.0%})")

    # the false alarm that matters most to viewers
    no_mouth = [c for c in clips if "mouth" not in truth(c)]
    lines += ["", "## Mouth-sound false alarms (clips without mouth sounds flagged as mouth)", "",
              f"- PE-A-Frame @0.6: {sum(1 for c in no_mouth if pe[c['id']].get('mouth', 0) > PE_THRESHOLD)}/{len(no_mouth)}"]
    if wonder:
        ok = [c for c in no_mouth if "error" not in wonder.get(c["id"], {"error": 1})]
        lines.append(f"- Muse Wonder: {sum(1 for c in ok if 'mouth' in wonder_cats(c['id']))}/{len(ok)}")
        lines += ["", f"Wonder: {len(wonder) - len(failed)} clips scored, {len(failed)} failed, "
                      f"total cost of all cached clip calls ${wonder_cost(wonder):.2f}."]

    report = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report-{date.today()}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
