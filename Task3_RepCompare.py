"""
Task 3 — Representation comparison (the user's "4-word n-grams / whole sentences"
ideas), judged HONESTLY on the cluster-holdout topic-shift proxy with the
train/val gap shown for each. Classical only (LinearSVC), no deep learning.

Question this answers: does adding richer n-grams (word 3/4-grams, wider char,
sentence-level features) improve the REALISTIC score, or just the optimistic one
while widening the overfit gap? We isolate the REPRESENTATION effect with a single
fast LinearSVC (no stacking / RF), because the stacking conclusion is already in.

For each representation we report:
  vanilla-F1   : stratified 5-fold OOF (optimistic — ignores topic shift)
  cluster-F1   : cluster-holdout OOF (realistic proxy)
  train-F1     : fit-on-fold-train, score-fold-train (memorization level)
  gap          : train-F1 - cluster-F1  (bigger = more overfit -> worse real LB)

Run:  .venv/bin/python Task3_RepCompare.py
"""
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from Task3_Improved_Model import cluster_folds, macro_f1, build_stylo, ROBUST_STYLO_IDX

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = Path("data")
np.random.seed(SEED)


def sentence_feats(texts):
    """Sentence-level structure features (the honest reading of 'whole sentences':
    n-gram-ing entire sentences is degenerate/sparse; sentence STATISTICS are the
    usable signal). Per doc: #sentences, mean/std sentence length (words),
    mean/std words-per-clause (comma splits), fraction of long sentences."""
    out = np.zeros((len(texts), 6), dtype=np.float32)
    for i, t in enumerate(texts):
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        lens = np.array([len(s.split()) for s in sents], dtype=np.float32)
        clauses = [len(re.split(r"[,;:]", s)) for s in sents]
        if len(lens) == 0:
            continue
        out[i, 0] = len(sents)
        out[i, 1] = lens.mean()
        out[i, 2] = lens.std()
        out[i, 3] = np.mean(clauses)
        out[i, 4] = np.std(clauses)
        out[i, 5] = float(np.mean(lens > 30))
    return out


# ---- representation factories (each returns a list of fitted-per-fold vecs) ----
def rep_baseline():   # current baseline
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)]
def rep_word3():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)]
def rep_word4():      # user's 4-word n-grams
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 4), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)]
def rep_charonly():   # shift-robust
    return [TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)]
def rep_widechar():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True)]

REPS = {
    "word(1,2)+char(3,5) [base]": (rep_baseline, False),
    "word(1,3)+char(3,5)":        (rep_word3, False),
    "word(1,4)+char(3,5) [4gram]": (rep_word4, False),
    "char(3,5) only [robust]":    (rep_charonly, False),
    "word(1,2)+char(2,6) wide":   (rep_widechar, False),
    "base + sentence/stylo feats": (rep_baseline, True),  # append dense sentence+stylo
}


def build_X(factory, texts, tr_idx, ev_idx, add_dense, DENSE):
    vecs = factory()
    mats_tr = [v.fit(texts[tr_idx]).transform(texts[tr_idx]) for v in vecs]
    mats_ev = [v.transform(texts[ev_idx]) for v in vecs]
    Xtr = sparse.hstack(mats_tr).tocsr(); Xev = sparse.hstack(mats_ev).tocsr()
    if add_dense:
        sc = StandardScaler().fit(DENSE[tr_idx])
        Xtr = sparse.hstack([Xtr, sparse.csr_matrix(sc.transform(DENSE[tr_idx]))]).tocsr()
        Xev = sparse.hstack([Xev, sparse.csr_matrix(sc.transform(DENSE[ev_idx]))]).tocsr()
    return Xtr, Xev


def eval_rep(name, factory, add_dense, texts, Y, DENSE, van_folds, clus_folds):
    t0 = time.time()
    # vanilla OOF
    van_pred = np.zeros(len(Y), dtype=int)
    for tr, val in van_folds:
        Xtr, Xev = build_X(factory, texts, tr, val, add_dense, DENSE)
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
        van_pred[val] = clf.predict(Xev)
    van_f1 = macro_f1(Y, van_pred)
    # cluster OOF + train-fit gap
    clu_pred = np.full(len(Y), -1, dtype=int); mask = np.zeros(len(Y), bool); tr_f1s = []
    for tr, val in clus_folds:
        Xtr, Xev = build_X(factory, texts, tr, val, add_dense, DENSE)
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
        clu_pred[val] = clf.predict(Xev); mask[val] = True
        tr_f1s.append(macro_f1(Y[tr], clf.predict(Xtr)))
    clu_f1 = macro_f1(Y[mask], clu_pred[mask]); tr_f1 = float(np.mean(tr_f1s))
    print(f"  {name:<30} vanilla={van_f1:.4f}  cluster={clu_f1:.4f}  "
          f"train={tr_f1:.4f}  gap={tr_f1-clu_f1:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return van_f1, clu_f1, tr_f1


def main():
    t0 = time.time()
    print("=" * 92, flush=True)
    print("REPRESENTATION COMPARISON — LinearSVC, judged on the realistic proxy", flush=True)
    print("=" * 92, flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    texts = train["text"].astype(str).to_numpy()
    Y = train["label"].to_numpy(dtype=int)
    DENSE = np.hstack([sentence_feats(texts), build_stylo(texts)[:, ROBUST_STYLO_IDX]])
    print(f"train={len(texts)}  dense sentence+stylo feats={DENSE.shape[1]}", flush=True)

    van_folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(Y)), Y))
    clus_folds, _ = cluster_folds(texts, Y)
    print(f"vanilla folds=5  cluster folds={len(clus_folds)}\n", flush=True)

    print(f"  {'representation':<30} {'':>8} (lower gap = transfers better)\n", flush=True)
    results = {}
    for name, (fac, add_dense) in REPS.items():
        results[name] = eval_rep(name, fac, add_dense, texts, Y, DENSE, van_folds, clus_folds)

    print("\n" + "=" * 92, flush=True)
    print("SUMMARY (sorted by REALISTIC cluster-F1)", flush=True)
    print("=" * 92, flush=True)
    print(f"  {'representation':<30} {'vanilla':>8} {'cluster':>8} {'gap':>7}   real-proj(~-0.008 single)", flush=True)
    for name, (v, c, tf) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:<30} {v:>8.4f} {c:>8.4f} {tf-c:>+7.3f}   ~{c-0.008:.3f}", flush=True)
    print(f"\n  baseline LinearSVC real-LB = 0.72990 (the bar to beat).", flush=True)
    print(f"  Read: higher vanilla with a BIGGER gap = more overfit = worse real LB.", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
