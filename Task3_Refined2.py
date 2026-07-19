"""
Task 3 — refinement round 2, TWO-LENS validated. Push in-family around wideB.
================================================================================
Current best eligible = wideB: LinearSVC C=0.25 balanced on word(1,3)+char_wb(2,6)
TF-IDF, REAL Kaggle 0.74477. This round tests a few PRE-REGISTERED in-family
variants (a wider char range, word 4-grams, a C tweak) and accepts one ONLY if it
beats wideB on BOTH independent topic-shift lenses (anti-winner's-curse; few
hypotheses, no grid-mining).

Lens A = cluster_folds (word-unigram KMeans, proven, calibrated ~-0.008 to real LB)
Lens B = char_wb(3,5) KMeans k=16 seed 2026 (independent)

Run:  .venv/bin/python Task3_Refined2.py
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

from Task3_Improved_Model import cluster_folds, macro_f1

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
np.random.seed(SEED)

# (name, word_range, char_range, C) — first entry is the reference (wideB).
CANDIDATES = [
    ("wideB  w(1,3) c(2,6) C0.25 [ref]", (1, 3), (2, 6), 0.25),
    ("R2a    w(1,4) c(2,6) C0.25",       (1, 4), (2, 6), 0.25),
    ("R2b    w(1,3) c(2,7) C0.25",       (1, 3), (2, 7), 0.25),
    ("R2c    w(1,3) c(2,6) C0.50",       (1, 3), (2, 6), 0.50),
    ("R2d    w(1,4) c(2,7) C0.25",       (1, 4), (2, 7), 0.25),
]


def mk(texts, tr, ev, wr, cr):
    vecs = [TfidfVectorizer(analyzer="word", ngram_range=wr, min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=cr, min_df=2, sublinear_tf=True)]
    Xtr = sparse.hstack([v.fit(texts[tr]).transform(texts[tr]) for v in vecs]).tocsr()
    Xev = sparse.hstack([v.transform(texts[ev]) for v in vecs]).tocsr()
    return Xtr, Xev


def lensB_folds(texts, k=16, n_splits=5, seed=2026):
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    cl = MiniBatchKMeans(k, random_state=seed, n_init=5, batch_size=2048).fit_predict(cv.fit_transform(texts))
    order = np.argsort(-np.bincount(cl, minlength=k))
    grp = {c: i % n_splits for i, c in enumerate(order)}
    g = np.array([grp[c] for c in cl])
    return [(np.where(g != f)[0], np.where(g == f)[0]) for f in range(n_splits)]


def eval_cfg(texts, Y, folds, wr, cr, C):
    pred = np.full(len(Y), -1, dtype=int); mask = np.zeros(len(Y), bool); trf = []
    for tr, val in folds:
        Xtr, Xev = mk(texts, tr, val, wr, cr)
        clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
        pred[val] = clf.predict(Xev); mask[val] = True
        trf.append(macro_f1(Y[tr], clf.predict(Xtr)))
    return macro_f1(Y[mask], pred[mask]), float(np.mean(trf))


def main():
    t0 = time.time()
    print("=" * 78, flush=True)
    print("TASK 3 — refinement round 2, TWO-LENS (accept only if beats wideB on BOTH)", flush=True)
    print("=" * 78, flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv"); test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy(); Y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy(); test_ids = test["id"].to_numpy()
    lensA = cluster_folds(texts, Y)[0]; lensB = lensB_folds(texts)
    print(f"train={len(texts)}  lensA={len(lensA)}  lensB={len(lensB)}\n", flush=True)

    print(f"  {'candidate':<34} {'lensA':>8} {'gapA':>7} {'lensB':>8} {'gapB':>7}", flush=True)
    res = {}
    for name, wr, cr, C in CANDIDATES:
        aF1, aTr = eval_cfg(texts, Y, lensA, wr, cr, C)
        bF1, bTr = eval_cfg(texts, Y, lensB, wr, cr, C)
        res[name] = (aF1, bF1, wr, cr, C)
        print(f"  {name:<34} {aF1:>8.4f} {aTr-aF1:>+7.3f} {bF1:>8.4f} {bTr-bF1:>+7.3f}  ({time.time()-t0:.0f}s)", flush=True)

    refA, refB = res[CANDIDATES[0][0]][0], res[CANDIDATES[0][0]][1]
    print(f"\n  reference wideB: lensA {refA:.4f}  lensB {refB:.4f}", flush=True)
    winners = {n: v for n, v in res.items()
               if n != CANDIDATES[0][0] and v[0] > refA and v[1] > refB}
    if winners:
        best = max(winners, key=lambda n: winners[n][0] + winners[n][1])
        a, b, wr, cr, C = res[best]
        print(f"  -> PASSES both lenses: {best}  (lensA {a:.4f}>{refA:.4f}, lensB {b:.4f}>{refB:.4f})", flush=True)
        print(f"     same-family projection (min lens - 0.008) ~= {min(a,b)-0.008:.3f}", flush=True)
        vecs_wr, vecs_cr, vecs_C = wr, cr, C
        chosen = best
    else:
        print("  -> NO candidate beats wideB on BOTH lenses. Guard HOLDS: keep wideB (0.74477); do NOT submit.", flush=True)
        _, _, vecs_wr, vecs_cr, vecs_C = res[CANDIDATES[0][0]]
        chosen = CANDIDATES[0][0]

    # write chosen candidate's test prediction (skip if it's just the ref)
    if chosen != CANDIDATES[0][0]:
        vecs = [TfidfVectorizer(analyzer="word", ngram_range=vecs_wr, min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=vecs_cr, min_df=2, sublinear_tf=True)]
        Xtr = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
        Xte = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
        clf = LinearSVC(C=vecs_C, class_weight="balanced", random_state=SEED).fit(Xtr, Y)
        pred = clf.predict(Xte).astype(int)
        out = OUT_DIR / "Task3_Refined2_Prediction.csv"
        pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
        print(f"\n  chosen={chosen}  wrote {out}  machine={int(pred.sum())} ({pred.mean():.1%})", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
