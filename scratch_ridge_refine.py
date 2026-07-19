"""
Ridge-alpha refinement — full (UNCAPPED) wideB resolution re-validation.
========================================================================
Track 4 found RidgeClassifier(alpha=1, balanced) passing both lenses on the
CAPPED rep (+0.0050/+0.0043). Here we (a) move to the real uncapped wideB rep
(min_df=2 char, no max_features — exactly the 0.74477 submission's rep), and
(b) sweep alpha around 1 to see whether the edge is a stable ridge or a spike.

Efficiency: vectorize each fold ONCE, reuse the fold matrices across the anchor
(LinearSVC C=0.25 bal) and every Ridge alpha. 10 vectorizations total, not 90.
Anchor recomputed uncapped so all deltas are apples-to-apples on this rep.

Classical ML only. Two-lens gated (Lens A / Lens B from scratch_lens).
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

from scratch_lens import load_data, get_folds, macro_f1

SEED = 42
t0 = time.time()
ALPHAS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]


def wideB_vecs_uncapped():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True)]


def fold_matrices(texts, folds):
    """Yield (Xtr, ytr_idx, Xev, yval_idx) per fold, vectorized once (uncapped)."""
    for fi, (tr, val) in enumerate(folds):
        vecs = wideB_vecs_uncapped()
        Xtr = sparse.hstack([v.fit(texts[tr]).transform(texts[tr]) for v in vecs]).tocsr()
        Xev = sparse.hstack([v.transform(texts[val]) for v in vecs]).tocsr()
        yield fi, tr, val, Xtr, Xev


def eval_all_ests(texts, Y, folds, ests):
    """ests: dict name->factory. Returns dict name->(mean, per_fold). One vectorize/fold."""
    acc = {name: [] for name in ests}
    for fi, tr, val, Xtr, Xev in fold_matrices(texts, folds):
        for name, fac in ests.items():
            clf = fac()
            clf.fit(Xtr, Y[tr])
            acc[name].append(macro_f1(Y[val], clf.predict(Xev)))
        print(f"    fold {fi} done nfeat={Xtr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    return {n: (float(np.mean(v)), [round(x, 4) for x in v]) for n, v in acc.items()}


def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    print(f"train={len(texts)} pos={Y.mean():.4f} foldsA={len(foldsA)} foldsB={len(foldsB)}", flush=True)

    ests = {"ANCHOR_LinSVC_C0.25": lambda: LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)}
    for al in ALPHAS:
        ests[f"Ridge_a{al}"] = (lambda al=al: RidgeClassifier(alpha=al, class_weight="balanced"))

    print("=== LENS A (uncapped) ===", flush=True)
    resA = eval_all_ests(texts, Y, foldsA, ests)
    print("=== LENS B (uncapped) ===", flush=True)
    resB = eval_all_ests(texts, Y, foldsB, ests)

    aA, aB = resA["ANCHOR_LinSVC_C0.25"][0], resB["ANCHOR_LinSVC_C0.25"][0]
    print(f"\nUNCAPPED anchor (LinearSVC C0.25 bal)  A={aA:.4f}  B={aB:.4f}", flush=True)
    print(f"{'candidate':22s} {'LensA':>7s} {'dA':>8s} {'LensB':>7s} {'dB':>8s}  PASS", flush=True)
    passers = []
    for name in ests:
        a, b = resA[name][0], resB[name][0]
        dA, dB = a - aA, b - aB
        passed = (a > aA) and (b > aB)
        if name.startswith("Ridge") and passed:
            passers.append((name, a, b, dA, dB, min(dA, dB)))
        print(f"{name:22s} {a:7.4f} {dA:+8.4f} {b:7.4f} {dB:+8.4f}  {'PASS' if passed else 'fail'}", flush=True)
    print(f"\nLensA per-fold: {{ {', '.join(f'{n}:{resA[n][1]}' for n in ests)} }}", flush=True)
    print(f"LensB per-fold: {{ {', '.join(f'{n}:{resB[n][1]}' for n in ests)} }}", flush=True)

    if not passers:
        print("\nNO Ridge alpha passes BOTH lenses at uncapped resolution. Null.", flush=True)
        return
    passers.sort(key=lambda r: -r[5])
    print(f"\nPASSERS (both lenses, uncapped), by min-margin:", flush=True)
    for name, a, b, dA, dB, mm in passers:
        print(f"  {name:16s} A={a:.4f}({dA:+.4f}) B={b:.4f}({dB:+.4f}) min={mm:+.4f}", flush=True)

    # write prediction for the best passer (refit uncapped on all 20k)
    best = passers[0][0]
    al = float(best.split("a")[1])
    print(f"\nrefitting best={best} on all 20k (uncapped) ...", flush=True)
    vecs = wideB_vecs_uncapped()
    X = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
    Xt = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
    clf = RidgeClassifier(alpha=al, class_weight="balanced")
    clf.fit(X, Y)
    pred = clf.predict(Xt).astype(int)
    import pandas as pd
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv("scratch_ridge_refine_pred.csv", index=False)
    print(f"wrote scratch_ridge_refine_pred.csv rows={len(pred)} machine={int(pred.sum())} "
          f"human={int((1-pred).sum())} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
