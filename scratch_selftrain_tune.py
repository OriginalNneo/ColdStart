"""
Tune transductive self-training to MAXIMIZE + STABILIZE topic-shift recovery.
Clean protocol: held-out cluster split into unlabeled pool (self-train) + disjoint
eval (never pseudo-labeled). Sweep confidence fraction, rounds, and class-balanced
pseudo selection. Judge on BOTH lenses by mean Δ and worst-fold Δ (stability).
"""
import time, itertools
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from scratch_lens import load_data, get_folds, macro_f1

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)


def vecs():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                            max_features=300000, sublinear_tf=True)]


def build(texts_tr, others, ws=1.6):
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
    conf = np.abs(margin); pl = (margin > 0).astype(int)
    if not balanced:
        thr = np.quantile(conf, 1 - frac)
        take = conf >= thr
    else:
        take = np.zeros(len(margin), bool)
        for c in (0, 1):
            idx = np.where(pl == c)[0]
            if len(idx) == 0: continue
            k = max(1, int(len(idx) * frac))
            top = idx[np.argsort(-conf[idx])[:k]]
            take[top] = True
    return take, pl


def run_cfg(Xtr, ytr, Xpool, Xeval, yeval, frac, rounds, balanced):
    clf = clf_().fit(Xtr, ytr)
    base = macro_f1(yeval, clf.predict(Xeval))
    for _ in range(rounds):
        m = clf.decision_function(Xpool)
        take, pl = select(m, frac, balanced)
        Xc = sparse.vstack([Xtr, Xpool[take]]).tocsr()
        yc = np.r_[ytr, pl[take]]
        clf = clf_().fit(Xc, yc)
    return base, macro_f1(yeval, clf.predict(Xeval))


def main():
    texts, Y, _, _ = load_data()
    foldsA, foldsB = get_folds()
    CFGS = list(itertools.product([0.3, 0.5, 0.7, 0.9], [1, 3], [False, True]))

    # precompute per-fold matrices ONCE (pool/eval split fixed per fold)
    prep = {}
    for lname, folds in [("A", foldsA), ("B", foldsB)]:
        prep[lname] = []
        for tr, val in folds:
            val = np.array(val); rng.shuffle(val); h = len(val)//2
            pool_idx, eval_idx = val[:h], val[h:]
            Xt, (Xp, Xe) = build(texts[tr], [texts[pool_idx], texts[eval_idx]])
            prep[lname].append((Xt, Y[tr], Xp, Xe, Y[eval_idx]))
        print(f"prepped Lens {lname} ({time.time()-t0:.0f}s)", flush=True)

    results = []
    for frac, rounds, bal in CFGS:
        deltas = {"A": [], "B": []}
        for lname in ("A", "B"):
            for (Xt, yt, Xp, Xe, ye) in prep[lname]:
                base, fin = run_cfg(Xt, yt, Xp, Xe, ye, frac, rounds, bal)
                deltas[lname].append(fin - base)
        alld = deltas["A"] + deltas["B"]
        mean = np.mean(alld); worst = np.min(alld); mA = np.mean(deltas["A"]); mB = np.mean(deltas["B"])
        results.append((frac, rounds, bal, mean, worst, mA, mB))
        print(f"frac={frac} rounds={rounds} bal={int(bal)}  meanΔ={mean:+.4f}  worst={worst:+.4f}  "
              f"A={mA:+.4f} B={mB:+.4f}  (both>0={'Y' if mA>0 and mB>0 else 'n'}) ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== ranked by (both-lens>0, then meanΔ, then worst) ===", flush=True)
    results.sort(key=lambda r: (not (r[5] > 0 and r[6] > 0), -r[3], -r[4]))
    for frac, rounds, bal, mean, worst, mA, mB in results[:8]:
        print(f"  frac={frac} rounds={rounds} bal={int(bal)}  meanΔ={mean:+.4f} worst={worst:+.4f} A={mA:+.4f} B={mB:+.4f}", flush=True)
    b = results[0]
    print(f"\nBEST: frac={b[0]} rounds={b[1]} balanced={bool(b[2])}  meanΔ={b[3]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
