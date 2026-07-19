"""
Task 3 — Refinement of the winning classical model, TWO-LENS validated.
================================================================================
Goal: find a refinement that plausibly beats the LinearSVC baseline (0.72990 real
Kaggle) on the ACTUAL leaderboard, not just offline. Classical only, no DL.

Strategy (why this and not an ensemble):
  - Ensembles/stacks/Markov-blends here deflate ~0.05-0.08 proxy->real (winner's
    curse + topic shift). A refinement that STAYS IN THE BASELINE'S FAMILY
    (single sparse-text LinearSVC, just a different n-gram range) carries only the
    ~-0.008 same-family deflation, so a genuine offline gain can survive.
  - The representation sweep already flagged wide-char(2,6) as the best single rep
    (cluster 0.7450 vs base 0.7404). Here we STRESS-TEST it on TWO INDEPENDENT
    topic-shift lenses and accept it ONLY if it beats base on BOTH (the anti-
    winner's-curse rule this project adopted after being burned 5x).

Lenses:
  A = cluster_folds (word-unigram KMeans, the proven proxy)
  B = char_wb(3,5) KMeans k=16, seed 2026 (built to be independent of lens A)

Run:  .venv/bin/python Task3_Refined.py
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
from sklearn.model_selection import StratifiedKFold

from Task3_Improved_Model import cluster_folds, macro_f1

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
OUT_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)


def mk_vecs(char_range, word_range):
    return [TfidfVectorizer(analyzer="word", ngram_range=word_range, min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=char_range, min_df=2, sublinear_tf=True)]


CANDIDATES = {
    "base char(3,5)+word(1,2)":  (mk_vecs, (3, 5), (1, 2)),
    "wideA char(2,6)+word(1,2)": (mk_vecs, (2, 6), (1, 2)),
    "wideB char(2,6)+word(1,3)": (mk_vecs, (2, 6), (1, 3)),
}


def lensB_folds(texts, y, k=16, n_splits=5, seed=2026):
    """Independent topic-shift lens: char-ngram KMeans, hold out whole clusters."""
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    Xc = cv.fit_transform(texts)
    cl = MiniBatchKMeans(k, random_state=seed, n_init=5, batch_size=2048).fit_predict(Xc)
    order = np.argsort(-np.bincount(cl, minlength=k))
    grp = {c: i % n_splits for i, c in enumerate(order)}
    g = np.array([grp[c] for c in cl])
    return [(np.where(g != f)[0], np.where(g == f)[0]) for f in range(n_splits)]


def eval_rep(vecfac, cr, wr, texts, Y, folds):
    pred = np.full(len(Y), -1, dtype=int); mask = np.zeros(len(Y), bool); trf = []
    for tr, val in folds:
        vecs = vecfac(cr, wr)
        Xtr = sparse.hstack([v.fit(texts[tr]).transform(texts[tr]) for v in vecs]).tocsr()
        Xev = sparse.hstack([v.transform(texts[val]) for v in vecs]).tocsr()
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
        pred[val] = clf.predict(Xev); mask[val] = True
        trf.append(macro_f1(Y[tr], clf.predict(Xtr)))
    return macro_f1(Y[mask], pred[mask]), float(np.mean(trf))


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("TASK 3 — refinement, TWO-LENS validated (classical, no DL)", flush=True)
    print("=" * 80, flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    Y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()

    lensA = cluster_folds(texts, Y)[0]
    lensB = lensB_folds(texts, Y)
    print(f"train={len(texts)}  lensA folds={len(lensA)}  lensB folds={len(lensB)}\n", flush=True)

    print(f"  {'candidate':<28} {'lensA':>8} {'gapA':>7} {'lensB':>8} {'gapB':>7}", flush=True)
    res = {}
    for name, (fac, cr, wr) in CANDIDATES.items():
        aF1, aTr = eval_rep(fac, cr, wr, texts, Y, lensA)
        bF1, bTr = eval_rep(fac, cr, wr, texts, Y, lensB)
        res[name] = (aF1, bF1)
        print(f"  {name:<28} {aF1:>8.4f} {aTr-aF1:>+7.3f} {bF1:>8.4f} {bTr-bF1:>+7.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    baseA, baseB = res["base char(3,5)+word(1,2)"]
    print("\n  ANTI-WINNER'S-CURSE RULE: accept a refinement only if it beats base on BOTH lenses.", flush=True)
    winners = {n: (a, b) for n, (a, b) in res.items()
               if n != "base char(3,5)+word(1,2)" and a > baseA and b > baseB}
    if winners:
        best = max(winners, key=lambda n: winners[n][0] + winners[n][1])
        a, b = winners[best]
        print(f"  -> PASSES both lenses: {best}  (lensA {a:.4f} vs {baseA:.4f}, lensB {b:.4f} vs {baseB:.4f})", flush=True)
        print(f"     same-family projection (min lens - 0.008) ~= {min(a, b) - 0.008:.3f}  vs baseline 0.72990", flush=True)
        fac, cr, wr = CANDIDATES[best]
    else:
        best = "base char(3,5)+word(1,2)"
        print(f"  -> NO candidate beats base on both lenses. Winner's-curse guard HOLDS; "
              f"do NOT submit a refinement (baseline stays the model).", flush=True)
        fac, cr, wr = CANDIDATES[best]

    # write the chosen candidate's test prediction regardless (for inspection)
    vecs = fac(cr, wr)
    Xtr = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
    Xte = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
    clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, Y)
    pred = clf.predict(Xte).astype(int)
    out = OUT_DIR / "Task3_Refined_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"\n  chosen = {best}", flush=True)
    print(f"  wrote {out}  machine={int(pred.sum())} ({pred.mean():.1%})", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
