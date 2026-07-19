"""
Task 3 — refinement round 3: DECISION-THRESHOLD tuning for Macro-F1, two-lens.
================================================================================
Best model = wideB: LinearSVC C=0.25 balanced on word(1,3)+char_wb(2,6), real 0.74477.
Every model so far classifies at decision_function > 0. But the metric is MACRO-F1
on imbalanced classes, and the optimal threshold under the SHIFTED test distribution
may not be 0. Threshold tuning is same-family (no capacity added, ~-0.008 deflation)
and directly optimizes the graded metric.

Method (leakage-safe, two-lens):
  - Collect wideB OOF decision scores on Lens A and Lens B.
  - Sweep a global threshold t; compute Macro-F1(y, score > t) per lens.
  - A shift is TRUSTED only if a SINGLE t improves BOTH lenses over t=0 AND the
    two lenses' individual optima roughly agree (else it's fold-specific noise).
  - Critical transfer test: apply Lens-A-optimal t to Lens B (and vice-versa) —
    if the tuned t only helps the lens it was tuned on, it will NOT transfer.

Run:  .venv/bin/python Task3_Refined3_threshold.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score

from Task3_Improved_Model import cluster_folds, macro_f1

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
np.random.seed(SEED)

WR, CR, C = (1, 3), (2, 6), 0.25   # wideB config


def wideB_oof_scores(texts, Y, folds):
    """Leakage-safe OOF decision scores for wideB over the given folds."""
    scores = np.zeros(len(Y)); mask = np.zeros(len(Y), bool)
    for tr, val in folds:
        vecs = [TfidfVectorizer(analyzer="word", ngram_range=WR, min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=CR, min_df=2, sublinear_tf=True)]
        Xtr = sparse.hstack([v.fit(texts[tr]).transform(texts[tr]) for v in vecs]).tocsr()
        Xev = sparse.hstack([v.transform(texts[val]) for v in vecs]).tocsr()
        clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
        scores[val] = clf.decision_function(Xev); mask[val] = True
    return scores, mask


def lensB_folds(texts, k=16, n_splits=5, seed=2026):
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    cl = MiniBatchKMeans(k, random_state=seed, n_init=5, batch_size=2048).fit_predict(cv.fit_transform(texts))
    order = np.argsort(-np.bincount(cl, minlength=k))
    grp = {c: i % n_splits for i, c in enumerate(order)}
    g = np.array([grp[c] for c in cl])
    return [(np.where(g != f)[0], np.where(g == f)[0]) for f in range(n_splits)]


def best_threshold(y, s, grid):
    f1s = [(t, macro_f1(y, (s > t).astype(int))) for t in grid]
    return max(f1s, key=lambda x: x[1])


def main():
    t0 = time.time()
    print("=" * 76, flush=True)
    print("TASK 3 — round 3: threshold tuning for Macro-F1 (wideB, two-lens)", flush=True)
    print("=" * 76, flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv"); test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy(); Y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy(); test_ids = test["id"].to_numpy()

    lensA = cluster_folds(texts, Y)[0]; lensB = lensB_folds(texts)
    sA, mA = wideB_oof_scores(texts, Y, lensA)
    sB, mB = wideB_oof_scores(texts, Y, lensB)
    print(f"  OOF scores ready ({time.time()-t0:.0f}s)", flush=True)

    grid = np.linspace(-0.8, 0.8, 161)
    f0A = macro_f1(Y[mA], (sA[mA] > 0).astype(int))
    f0B = macro_f1(Y[mB], (sB[mB] > 0).astype(int))
    tA, fA = best_threshold(Y[mA], sA[mA], grid)
    tB, fB = best_threshold(Y[mB], sB[mB], grid)

    print(f"\n  default t=0:      Lens A {f0A:.4f}   Lens B {f0B:.4f}", flush=True)
    print(f"  Lens-A-optimal t={tA:+.3f}: Lens A {fA:.4f}  (applied to Lens B: {macro_f1(Y[mB],(sB[mB]>tA).astype(int)):.4f})", flush=True)
    print(f"  Lens-B-optimal t={tB:+.3f}: Lens B {fB:.4f}  (applied to Lens A: {macro_f1(Y[mA],(sA[mA]>tB).astype(int)):.4f})", flush=True)

    # search a SINGLE t that maximizes the MIN of the two lenses (robust choice)
    both = [(t, min(macro_f1(Y[mA], (sA[mA] > t).astype(int)),
                    macro_f1(Y[mB], (sB[mB] > t).astype(int)))) for t in grid]
    tStar, fStar = max(both, key=lambda x: x[1])
    fA_star = macro_f1(Y[mA], (sA[mA] > tStar).astype(int))
    fB_star = macro_f1(Y[mB], (sB[mB] > tStar).astype(int))
    print(f"\n  robust single t*={tStar:+.3f}: Lens A {fA_star:.4f} (vs {f0A:.4f})  "
          f"Lens B {fB_star:.4f} (vs {f0B:.4f})", flush=True)

    improves_both = (fA_star > f0A + 1e-9) and (fB_star > f0B + 1e-9)
    print("\n" + "=" * 76, flush=True)
    if improves_both:
        print(f"  -> t*={tStar:+.3f} improves BOTH lenses. TRUSTED. Writing thresholded prediction.", flush=True)
        vecs = [TfidfVectorizer(analyzer="word", ngram_range=WR, min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=CR, min_df=2, sublinear_tf=True)]
        Xtr = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
        Xte = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
        clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED).fit(Xtr, Y)
        pred = (clf.decision_function(Xte) > tStar).astype(int)
        out = OUT_DIR / "Task3_Refined3_thresh_Prediction.csv"
        pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
        print(f"  wrote {out}  machine={int(pred.sum())} ({pred.mean():.1%})  "
              f"proj ~= {min(fA_star,fB_star)-0.008:.3f}", flush=True)
    else:
        print(f"  -> NO single t beats t=0 on BOTH lenses. Threshold tuning does not transfer.", flush=True)
        print(f"     Plateau confirmed; keep wideB at default threshold (0.74477). Do NOT submit.", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
