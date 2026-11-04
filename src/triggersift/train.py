"""Train per-category classifiers on PE-A-Frame embeddings of the labeled clip bank, tested on unseen videos.

    uv run --extra local python -m triggersift.train

Leave-one-video-out: for each video, train on the clips of the other videos and predict its clips, so every score
comes from a video the model never saw. Compared against PE-A-Frame zero-shot on the same clips.
"""

import argparse
from collections import Counter
from datetime import date

import numpy as np
import torch

from .bank_eval import OUT, PE_THRESHOLD, auc, pe_embeds, pe_scores, prf, truth
from .labeling import CLIPS, load


def features(emb: np.ndarray) -> np.ndarray:
    """One fingerprint per clip: average and peak of each embedding dimension over its 40 ms frames."""
    return np.concatenate([emb.mean(axis=0), emb.max(axis=0)])


def text_features(emb: np.ndarray) -> np.ndarray:
    """PE-A-Frame's own zero-shot scores for every taxonomy sound phrase (average and peak over the frames):
    ~75 numbers that carry what the model already knows ("rain" and "fan" are both background noise)."""
    from .pe_frame import probs_from_embeds
    from .profile import taxonomy
    p = probs_from_embeds(emb, tuple(t.sound for t in taxonomy().values()))
    return np.concatenate([p.mean(axis=0), p.max(axis=0)])


def fit_all(Xtr, Ytr, Wtr, C, active) -> tuple[torch.Tensor, ...]:
    """Balanced L2 logistic regressions for all categories at once on the GPU: same objective as scikit-learn's
    LogisticRegression(C, class_weight="balanced") per category, which took hours for ~1,900 fits on 700 clips + FSD50K.
    Returns (mean, std, weights, bias) of the standardised model."""
    x = torch.as_tensor(Xtr, dtype=torch.float32, device="cuda")
    y = torch.as_tensor(Ytr[:, active], dtype=torch.float32, device="cuda")
    s = torch.as_tensor(Wtr, dtype=torch.float32, device="cuda")[:, None]
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / std
    n, pos = len(y), y.sum(0)
    cw = torch.where(y > 0, n / (2 * pos), n / (2 * (n - pos))) * s * C
    w = torch.zeros(x.shape[1], y.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(y.shape[1], device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, history_size=20, tolerance_grad=1e-4, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (cw * torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none")).sum()
        loss = loss + 0.5 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mean, std, w.detach(), b.detach()


def predict(model, X) -> np.ndarray:
    mean, std, w, b = model
    x = torch.as_tensor(X, dtype=torch.float32, device="cuda")
    return torch.sigmoid((x - mean) / std @ w + b).cpu().numpy()


def oof(X, Y, Z, vid, C, min_pos=3, extra=None, ours=True, w_ours=1.0):
    """Out-of-video probabilities [clips, cats]; zero-shot where a training fold lacks positives/negatives.

    extra = (X, Y) of outside clips (FSD50K) added to every training fold, never tested on.
    ours = False trains on the outside clips only; w_ours weighs our clips against them."""
    P, fallback = np.zeros_like(Z), Counter()
    for v in np.unique(vid):
        test, train = vid == v, vid != v
        Xs, Ys, Ws = [], [], []
        if ours:
            Xs.append(X[train]); Ys.append(Y[train]); Ws.append(np.full(train.sum(), w_ours))
        if extra is not None:
            Xs.append(extra[0]); Ys.append(extra[1]); Ws.append(np.ones(len(extra[0])))
        Xtr, Ytr, Wtr = np.vstack(Xs), np.vstack(Ys), np.concatenate(Ws)
        ok = (Ytr.sum(0) >= min_pos) & ((~Ytr).sum(0) >= min_pos)
        for j in np.flatnonzero(~ok):
            P[test, j] = Z[test, j]
            fallback[j] += int(test.sum())
        P[np.ix_(test, ok)] = predict(fit_all(Xtr, Ytr, Wtr, C, ok), X[test])
    return P, fallback


def summary(P, Y, thr):
    aucs = [auc(list(P[Y[:, j], j]), list(P[~Y[:, j], j])) for j in range(Y.shape[1])]
    f1s = [prf(int((P[Y[:, j], j] > thr).sum()), int((P[~Y[:, j], j] > thr).sum()),
               int((P[Y[:, j], j] <= thr).sum()))[2] for j in range(Y.shape[1])]
    return aucs, f1s


def save_heads(X, Y, Xf, Yf, cats, C) -> None:
    """The shipped detector = variant H trained on every clip: head 0 on yours, head 1 on FSD50K + yours (x10)."""
    from .pe_frame import HEADS
    ok = np.ones(len(cats), bool)
    a = fit_all(X, Y, np.ones(len(X)), C, ok)
    g = fit_all(np.vstack([X, Xf]), np.vstack([Y, Yf]), np.r_[np.full(len(X), 10.0), np.ones(len(Xf))], C, ok)
    arrays = {f"{name}{i}": t.cpu().numpy() for i, m in enumerate((a, g)) for name, t in zip("msWb", m)}
    np.savez_compressed(HEADS, cats=np.array(cats), **arrays)
    print(f"saved {HEADS}: {len(cats)} classes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.05, help="inverse regularization: smaller = simpler model")
    ap.add_argument("--fsd", action="store_true", help="add FSD50K clips (python -m triggersift.fsd50k first)")
    ap.add_argument("--save", action="store_true", help="with --fsd: also train on everything and save the heads "
                                                         "the app uses (pe-a-frame-trained)")
    a = ap.parse_args()

    clips = [c for c in load() if (CLIPS / f"{c['id']}.mp4").exists() and c["triggers"]]
    counts = Counter(k for c in clips for k in truth(c))
    cats = sorted((k for k, n in counts.items() if n >= a.min), key=lambda k: -counts[k])
    videos = sorted({c["video"] for c in clips})

    emb = pe_embeds(clips)
    X = np.stack([features(emb[c["id"]]) for c in clips])
    Y = np.array([[k in truth(c) for k in cats] for c in clips])
    vid = np.array([videos.index(c["video"]) for c in clips])
    zs = pe_scores(clips, cats)
    Z = np.array([[zs[c["id"]][k] for k in cats] for c in clips])

    T = np.stack([text_features(emb[c["id"]]) for c in clips])
    variants = {
        "A raw fingerprint": (X, a.C),
        "B zero-shot phrase scores": (T, 1.0),
        "C both": (np.hstack([X, T]), a.C),
    }
    results = {}
    for name, (Xv, C) in variants.items():
        results[name] = oof(Xv, Y, Z, vid, C)
        print("done:", name, flush=True)
    if a.fsd:
        from . import fsd50k
        rows = fsd50k.select()
        femb = fsd50k.embeds(rows)
        Xf = np.stack([features(femb[r["fname"]]) for r in rows])
        Yf = np.array([[k in r["cats"] for k in cats] for r in rows])
        results["E FSD50K only"] = oof(X, Y, Z, vid, a.C, extra=(Xf, Yf), ours=False)
        results["F FSD50K + yours (x1)"] = oof(X, Y, Z, vid, a.C, extra=(Xf, Yf))
        results["G FSD50K + yours (x10)"] = oof(X, Y, Z, vid, a.C, extra=(Xf, Yf), w_ours=10.0)
        # H: yours alone is best on tapping/crunch, with FSD50K on liquid/crinkling: average the two
        results["H average of A and G"] = ((results["A raw fingerprint"][0] + results["G FSD50K + yours (x10)"][0]) / 2,
                                           Counter())
        if a.save:
            save_heads(X, Y, Xf, Yf, cats, a.C)
    # D: average of B's trained probability and the plain zero-shot category score
    results["D B averaged with zero-shot"] = ((results["B zero-shot phrase scores"][0] + Z) / 2, Counter())

    cmp = ["## Variants (AUC per category, average AUC / F1)", "",
           "| variant | " + " | ".join(cats) + " | avg AUC | avg F1 |", "|---" * (len(cats) + 3) + "|"]
    za, zf = summary(Z, Y, PE_THRESHOLD)
    cmp.append("| zero-shot | " + " | ".join(f"{x:.2f}" for x in za) + f" | {np.mean(za):.2f} | {np.mean(zf):.2f} |")
    best, best_score = None, -1
    for name, (Pv, _) in results.items():
        va, vf = summary(Pv, Y, .5)
        cmp.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in va) + f" | {np.mean(va):.2f} | {np.mean(vf):.2f} |")
        if np.mean(va) + np.mean(vf) > best_score:
            best, best_score = name, np.mean(va) + np.mean(vf)
    print("\n".join(cmp) + "\n")
    P, fb = results[best]
    fallback = Counter({cats[j]: n for j, n in fb.items()})

    lines = [f"# Trained classifier on PE-A-Frame embeddings, {date.today()}", "",
             f"{len(clips)} clips, {len(videos)} videos, leave-one-video-out. Logistic regression per category "
             f"(balanced classes). Best variant: **{best}**.", ""] + cmp + ["", "## Best variant in detail", "",
             "| category | clips | in videos | zero-shot AUC | trained AUC | zero-shot F1 @0.6 | trained P / R / F1 @0.5 |",
             "|---|---|---|---|---|---|---|"]
    z_aucs, t_aucs, z_f1s, t_f1s = [], [], [], []
    for j, k in enumerate(cats):
        y = Y[:, j]
        za = auc(list(Z[y, j]), list(Z[~y, j]))
        ta = auc(list(P[y, j]), list(P[~y, j]))
        zf = prf(int((Z[y, j] > PE_THRESHOLD).sum()), int((Z[~y, j] > PE_THRESHOLD).sum()),
                 int((Z[y, j] <= PE_THRESHOLD).sum()))[2]
        tp, fp, fn = int((P[y, j] > .5).sum()), int((P[~y, j] > .5).sum()), int((P[y, j] <= .5).sum())
        tprf = prf(tp, fp, fn)
        z_aucs.append(za); t_aucs.append(ta); z_f1s.append(zf); t_f1s.append(tprf[2])
        nv = len({vid[i] for i in np.flatnonzero(y)})
        note = f" (zero-shot used for {fallback[k]} clips)" if fallback[k] else ""
        lines.append(f"| {k} | {int(y.sum())} | {nv}/{len(videos)} | {za:.2f} | **{ta:.2f}** | {zf:.2f} | "
                     f"{tprf[0]:.2f} / {tprf[1]:.2f} / **{tprf[2]:.2f}**{note} |")
    lines += ["", f"**Average:** AUC {np.mean(z_aucs):.2f} → {np.mean(t_aucs):.2f}, "
                  f"F1 {np.mean(z_f1s):.2f} → {np.mean(t_f1s):.2f}", ""]

    if "mouth" in cats:
        j = cats.index("mouth")
        neg = ~Y[:, j]
        lines.append(f"Mouth false alarms on clips without mouth sounds: zero-shot {int((Z[neg, j] > PE_THRESHOLD).sum())}"
                     f"/{int(neg.sum())}, trained {int((P[neg, j] > .5).sum())}/{int(neg.sum())}")

    lines += ["", "Per held-out video (share of the clip's categories found, and of extra categories claimed):", ""]
    for v, name in enumerate(videos):
        m = vid == v
        found = (P[m] > .5) & Y[m]
        extra = (P[m] > .5) & ~Y[m]
        lines.append(f"- {name[:60]}: {int(m.sum())} clips, found {found.sum()}/{Y[m].sum()} labels, "
                     f"{extra.sum()} false labels")

    report = "\n".join(lines) + "\n"
    (OUT / f"train-{date.today()}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
