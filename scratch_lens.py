"""
SHARED TWO-LENS HARNESS for the refinement campaign (classical, NO deep learning).
================================================================================
Every campaign agent imports THIS module so all experiments are judged on the
EXACT topic-shift discipline the ledger requires, on IDENTICAL fold indices.

Lens A = word-unigram KMeans k=10 seed=42 -> 5 groups, hold out whole clusters
Lens B = char_wb(3,5) KMeans k=16 seed=2026 -> 5 groups (independent of A)

Anchor to beat: wideB = LinearSVC(C=0.25, balanced) on word(1,3)+char_wb(2,6)
  ledger real Kaggle 0.74477; Lens A 0.7439, Lens B 0.7640.

TWO-LENS RULE: accept a refinement ONLY if it beats wideB on BOTH lenses.
Report Lens A and Lens B macro-F1 for your candidate vs the wideB anchor.

Usage:
    from scratch_lens import (load_data, get_folds, eval_rep, macro_f1,
                              wideB_vecs, ANCHOR)
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()          # cached identical indices
    a = eval_rep(my_vec_factory, texts, Y, foldsA)   # your candidate, Lens A
    b = eval_rep(my_vec_factory, texts, Y, foldsB)   # Lens B
    # beats anchor iff a > ANCHOR['A'] and b > ANCHOR['B']
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
SEED = 42
N_SPLITS = 5
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
FOLD_CACHE = Path("scratch_folds.npz")
ANCHOR_CACHE = Path("scratch_anchor.json")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def load_data():
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    return (tr["text"].astype(str).to_numpy(),
            tr["label"].to_numpy(dtype=int),
            te["text"].astype(str).to_numpy(),
            te["id"].to_numpy())


def _lensA_folds(texts, y, n_clusters=10):
    cv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=5,
                         max_features=20000, sublinear_tf=True)
    Xc = cv.fit_transform(texts)
    cl = MiniBatchKMeans(n_clusters, random_state=SEED, n_init=5,
                         batch_size=2048).fit_predict(Xc)
    order = np.argsort(-np.bincount(cl, minlength=n_clusters))
    grp = np.array([{c: i % N_SPLITS for i, c in enumerate(order)}[c] for c in cl])
    return [(np.where(grp != g)[0], np.where(grp == g)[0]) for g in range(N_SPLITS)
            if len(np.where(grp == g)[0]) and len(np.unique(y[grp == g])) == 2]


def _lensB_folds(texts, y, k=16, seed=2026):
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    Xc = cv.fit_transform(texts)
    cl = MiniBatchKMeans(k, random_state=seed, n_init=5, batch_size=2048).fit_predict(Xc)
    order = np.argsort(-np.bincount(cl, minlength=k))
    grp = np.array([{c: i % N_SPLITS for i, c in enumerate(order)}[c] for c in cl])
    return [(np.where(grp != g)[0], np.where(grp == g)[0]) for g in range(N_SPLITS)
            if len(np.where(grp == g)[0]) and len(np.unique(y[grp == g])) == 2]


def get_folds():
    """Compute once, cache to disk; all agents load IDENTICAL fold indices."""
    if FOLD_CACHE.exists():
        d = np.load(FOLD_CACHE, allow_pickle=True)
        return list(d["A"]), list(d["B"])
    texts, Y, _, _ = load_data()
    A = _lensA_folds(texts, Y)
    B = _lensB_folds(texts, Y)
    np.savez(FOLD_CACHE,
             A=np.array([(tr, val) for tr, val in A], dtype=object),
             B=np.array([(tr, val) for tr, val in B], dtype=object))
    return A, B


def wideB_vecs():
    """The current best (anchor) representation."""
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True)]


def eval_rep(vec_factory, texts, Y, folds, est_factory=None, transductive=False,
             sample_weight_fn=None):
    """Per-fold macro-F1 for a candidate representation + estimator.

    vec_factory():   -> list of fitted-per-fold TfidfVectorizer(s) (or transformers
                        exposing fit/transform). Their outputs are hstacked.
    est_factory():   -> a fresh classifier (default LinearSVC C=0.25 balanced).
    transductive:    fit vectorizers on train+val TEXT (labels only from train).
    sample_weight_fn(tr_texts, val_texts) -> weights for train rows (optional).
    Returns (mean_macro_f1, per_fold_list).
    """
    if est_factory is None:
        est_factory = lambda: LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
    per_fold = []
    for tr, val in folds:
        vecs = vec_factory()
        if transductive:
            allt = np.concatenate([texts[tr], texts[val]])
            mats = [v.fit(allt) for v in vecs]
            Xtr = sparse.hstack([v.transform(texts[tr]) for v in vecs]).tocsr()
            Xev = sparse.hstack([v.transform(texts[val]) for v in vecs]).tocsr()
        else:
            Xtr = sparse.hstack([v.fit(texts[tr]).transform(texts[tr]) for v in vecs]).tocsr()
            Xev = sparse.hstack([v.transform(texts[val]) for v in vecs]).tocsr()
        clf = est_factory()
        w = sample_weight_fn(texts[tr], texts[val]) if sample_weight_fn else None
        try:
            clf.fit(Xtr, Y[tr], sample_weight=w)
        except TypeError:
            clf.fit(Xtr, Y[tr])
        per_fold.append(macro_f1(Y[val], clf.predict(Xev)))
    return float(np.mean(per_fold)), [round(v, 4) for v in per_fold]


# ledger anchor; recomputed and overwritten by main() below on first run
ANCHOR = {"A": 0.7439, "B": 0.7640}
if ANCHOR_CACHE.exists():
    ANCHOR = json.loads(ANCHOR_CACHE.read_text())


if __name__ == "__main__":
    import time
    t0 = time.time()
    texts, Y, _, _ = load_data()
    print(f"train={len(texts)}  pos={Y.mean():.4f}", flush=True)
    foldsA, foldsB = get_folds()
    print(f"lensA folds={len(foldsA)}  lensB folds={len(foldsB)}  ({time.time()-t0:.0f}s)", flush=True)
    a, af = eval_rep(wideB_vecs, texts, Y, foldsA)
    b, bf = eval_rep(wideB_vecs, texts, Y, foldsB)
    print(f"wideB anchor  LensA={a:.4f} {af}\n              LensB={b:.4f} {bf}  ({time.time()-t0:.0f}s)", flush=True)
    ANCHOR_CACHE.write_text(json.dumps({"A": round(a, 4), "B": round(b, 4)}))
    print(f"ledger says LensA~0.7439 LensB~0.7640  -> reproduced={abs(a-0.7439)<0.006 and abs(b-0.7640)<0.006}", flush=True)
    print(f"anchor cached to {ANCHOR_CACHE}  ({time.time()-t0:.0f}s)", flush=True)
