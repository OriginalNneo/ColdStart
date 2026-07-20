"""
STAGE 1 — push transductive shift-recovery past self-training, four-lens gated.
==============================================================================
The +0.081 topic-shift tax is the prize; validated self-training recovered only
~+0.012 of it (and was only ever checked on Lens A/B). This script tests three
transductive levers against the SAME base stack, on ALL FOUR lenses (A,B,C1,C2),
using the ledger's non-circular pool/eval protocol:

  base            RidgeClassifier(0.9,bal) on [1.6*word(1,3) | char_wb(2,6)]   (the 0.752 model)
  selftrain       validated frac0.7 class-balanced pseudo-labels, 3 rounds     (1D reference)
  iw              covariate-shift importance weighting: reweight train by the
                  density ratio P(test-like)/(1-P) from a train-vs-target domain
                  classifier, then fit the weighted stack                       (1A)
  iw_selftrain    iw weights + self-training combined                          (1A x 1D)

Non-circular protocol (identical to scratch_selftrain_tune.py): each held-out
cluster is split into an unlabeled POOL (self-trained on) and a disjoint EVAL set
that is never pseudo-labeled. IW weights use only target-domain FEATURES (pool+eval,
never labels) — the honest analog of "we see all real test features at predict time".

GATE: a lever is a candidate only if min over {A,B,C1,C2} of its mean Δ-vs-base > 0
AND its worst single fold is not badly negative. Classical ML only, no DL.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
CANDS = ["base", "selftrain", "iw", "iw_selftrain"]


def vecs():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                            max_features=300000, sublinear_tf=True)]


def build(texts_tr, others, ws=1.6):
    """Fit stack vectorizers on train text; transform train + each 'others' set."""
    v = vecs()
    Xw = v[0].fit_transform(texts_tr).astype(np.float32)
    Xc = v[1].fit_transform(texts_tr).astype(np.float32)
    Xt = sparse.hstack([Xw * ws, Xc]).tocsr()
    outs = []
    for T in others:
        outs.append(sparse.hstack([v[0].transform(T).astype(np.float32) * ws,
                                    v[1].transform(T).astype(np.float32)]).tocsr())
    return Xt, outs


def clf_():
    return RidgeClassifier(alpha=0.9, class_weight="balanced")


def select(margin, frac, balanced):
    """Top-frac most confident, optionally per-class (balanced)."""
    conf = np.abs(margin); pl = (margin > 0).astype(int)
    if not balanced:
        take = conf >= np.quantile(conf, 1 - frac)
    else:
        take = np.zeros(len(margin), bool)
        for c in (0, 1):
            idx = np.where(pl == c)[0]
            if len(idx) == 0:
                continue
            k = max(1, int(len(idx) * frac))
            take[idx[np.argsort(-conf[idx])[:k]]] = True
    return take, pl


def self_train(Xtr, ytr, Xpool, Xeval, yeval, frac=0.7, rounds=3, balanced=True,
               w_tr=None):
    """Iterative pseudo-labeling of Xpool; optional train-row importance weights w_tr."""
    def fit(X, y, w):
        return clf_().fit(X, y, sample_weight=w)
    clf = fit(Xtr, ytr, w_tr)
    for _ in range(rounds):
        m = clf.decision_function(Xpool)
        take, pl = select(m, frac, balanced)
        Xc = sparse.vstack([Xtr, Xpool[take]]).tocsr()
        yc = np.r_[ytr, pl[take]]
        wc = None if w_tr is None else np.r_[w_tr, np.ones(int(take.sum()))]
        clf = fit(Xc, yc, wc)
    return macro_f1(yeval, clf.predict(Xeval))


def iw_weights(train_texts, target_texts, clip_p=(0.05, 0.95), cap=10.0, seed=13):
    """Density-ratio sample weights: P(target-like)/(1-P) per train doc, from a
    train-vs-target domain classifier on char_wb(3,5). Uses target FEATURES only."""
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    Xtr = cv.fit_transform(train_texts)
    Xta = cv.transform(target_texts)
    X = sparse.vstack([Xtr, Xta]).tocsr()
    z = np.r_[np.zeros(Xtr.shape[0]), np.ones(Xta.shape[0])]
    dc = LogisticRegression(max_iter=300, C=1.0, random_state=seed).fit(X, z)
    p = np.clip(dc.predict_proba(Xtr)[:, 1], *clip_p)
    w = np.clip(p / (1 - p), 0, cap)
    return (w / w.mean()).astype(np.float64), float(p.mean())


def eval_lens(name, folds, texts, Y):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        val = np.array(val); rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        target_idx = val                                  # full held-out cluster = target domain
        Xt, (Xp, Xe) = build(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, ye = Y[tr], Y[eval_idx]
        w, pmean = iw_weights(texts[tr], texts[target_idx])

        base = clf_().fit(Xt, ytr)
        acc["base"].append(macro_f1(ye, base.predict(Xe)))
        acc["selftrain"].append(self_train(Xt, ytr, Xp, Xe, ye))
        acc["iw"].append(macro_f1(ye, clf_().fit(Xt, ytr, sample_weight=w).predict(Xe)))
        acc["iw_selftrain"].append(self_train(Xt, ytr, Xp, Xe, ye, w_tr=w))
        print(f"  [{name}] fold {fi} pmean={pmean:.3f} base={acc['base'][-1]:.4f} "
              f"st={acc['selftrain'][-1]:.4f} iw={acc['iw'][-1]:.4f} "
              f"iwst={acc['iw_selftrain'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    foldsA, foldsB = get_folds()
    foldsC1 = lensC1_folds(texts, Y)
    foldsC2, _ = lensC2_folds(texts, test_texts, Y)
    lenses = [("A", foldsA), ("B", foldsB), ("C1", foldsC1), ("C2", foldsC2)]
    print(f"train={len(texts)} folds " +
          " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)

    res = {n: eval_lens(n, f, texts, Y) for n, f in lenses}

    print("\n============ Δ vs base (mean per lens) ============", flush=True)
    print(f"{'candidate':14s}" + "".join(f"{'L'+n:>14s}" for n, _ in lenses) +
          f"{'min':>10s}{'worst':>10s}", flush=True)
    for c in CANDS:
        if c == "base":
            continue
        means, worst = [], []
        for n, _ in lenses:
            d = res[n][c] - res[n]["base"]
            means.append(d.mean()); worst.append(d.min())
        row = f"{c:14s}" + "".join(f"{m:+13.4f} " for m in means)
        print(row + f"{min(means):+9.4f} {min(worst):+9.4f}", flush=True)

    print("\n--- four-lens gate (min mean-Δ across A/B/C1/C2 must be > 0) ---", flush=True)
    for c in CANDS:
        if c == "base":
            continue
        means = [(res[n][c] - res[n]["base"]).mean() for n, _ in lenses]
        wins = sum(m > 0 for m in means)
        verdict = "PASS" if min(means) > 0 else "fail"
        print(f"  {c:14s} {wins}/4 lenses positive  min={min(means):+.4f}  [{verdict}]", flush=True)

    print("\nper-fold detail:", flush=True)
    for n, _ in lenses:
        print(f"  Lens {n}: " + ", ".join(
            f"{c}={[round(x,4) for x in res[n][c]]}" for c in CANDS), flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
