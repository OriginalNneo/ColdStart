"""
Transductive SELF-TRAINING — attack the topic-shift tax (vanilla CV ~0.84 vs LB ~0.75).
=====================================================================================
Idea (classical, no DL): fit the stack on labeled train, pseudo-label the most
CONFIDENT test rows, refit on train + pseudo-labels. Adapts the decision boundary to
the test topic/style distribution. Repeat a few rounds.

Clean non-circular validation on the shift lenses: for each held-out cluster (=the
shifted 'test'), split it into an UNLABELED pool (self-train on it) and a disjoint
EVAL set (never pseudo-labeled). If self-training on the unlabeled pool improves EVAL
over the base model, the technique recovers shift tax and should help the real test.

Also reports vanilla random-5fold CV (headroom) vs the lens/LB reality.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from scratch_lens import load_data, get_folds, macro_f1

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)


def vecs():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                            max_features=300000, sublinear_tf=True)]


def build(texts_tr, texts_other, ws=1.6):
    v = vecs()
    Xw = v[0].fit_transform(texts_tr).astype(np.float32)
    Xc = v[1].fit_transform(texts_tr).astype(np.float32)
    Xt = sparse.hstack([Xw * ws, Xc]).tocsr()
    outs = []
    for T in texts_other:
        Ow = v[0].transform(T).astype(np.float32)
        Oc = v[1].transform(T).astype(np.float32)
        outs.append(sparse.hstack([Ow * ws, Oc]).tocsr())
    return Xt, outs


def stack_clf():
    return RidgeClassifier(alpha=0.9, class_weight="balanced")


def self_train(Xtr, ytr, Xpool, Xeval, yeval, rounds=2, frac=0.5):
    """Base then iterative pseudo-labeling of Xpool; score Xeval each round."""
    clf = stack_clf().fit(Xtr, ytr)
    base = macro_f1(yeval, clf.predict(Xeval))
    scores = [base]
    Xcur, ycur = Xtr, ytr
    used = np.zeros(Xpool.shape[0], dtype=bool)
    for r in range(rounds):
        margin = clf.decision_function(Xpool)          # signed distance
        conf = np.abs(margin)
        thr = np.quantile(conf, 1 - frac)               # top-frac most confident
        take = (conf >= thr)
        pl = (margin > 0).astype(int)
        Xcur = sparse.vstack([Xtr, Xpool[take]]).tocsr()
        ycur = np.r_[ytr, pl[take]]
        clf = stack_clf().fit(Xcur, ycur)
        scores.append(macro_f1(yeval, clf.predict(Xeval)))
    return scores


def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()

    # ---- vanilla random 5-fold CV of the stack (headroom) ----
    print("== vanilla random 5-fold CV of the stack (headroom vs LB 0.752) ==", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    vf = []
    for tr, val in skf.split(texts, Y):
        Xt, (Xv,) = build(texts[tr], [texts[val]])
        clf = stack_clf().fit(Xt, Y[tr])
        vf.append(macro_f1(Y[val], clf.predict(Xv)))
    print(f"  vanilla CV macro-F1 = {np.mean(vf):.4f}  folds={[round(x,4) for x in vf]}", flush=True)
    print(f"  => topic-shift tax vs LB 0.752 ≈ {np.mean(vf)-0.752:+.4f} (the prize)", flush=True)

    # ---- self-training on the shift lenses (clean unlabeled/eval split) ----
    for lname, folds in [("A", foldsA), ("B", foldsB)]:
        print(f"\n== self-training validation on Lens {lname} ==", flush=True)
        agg = None
        for fi, (tr, val) in enumerate(folds):
            val = np.array(val)
            rng.shuffle(val)
            half = len(val) // 2
            pool_idx, eval_idx = val[:half], val[half:]
            Xt, (Xpool, Xeval) = build(texts[tr], [texts[pool_idx], texts[eval_idx]])
            sc = self_train(Xt, Y[tr], Xpool, Xeval, Y[eval_idx], rounds=2, frac=0.5)
            agg = np.array(sc) if agg is None else agg + np.array(sc)
            print(f"  fold {fi}: base={sc[0]:.4f} -> r1={sc[1]:.4f} -> r2={sc[2]:.4f} "
                  f"(Δ={sc[-1]-sc[0]:+.4f}) ({time.time()-t0:.0f}s)", flush=True)
        agg /= len(folds)
        print(f"  Lens {lname} MEAN: base={agg[0]:.4f} r1={agg[1]:.4f} r2={agg[2]:.4f}  "
              f"Δ_final={agg[-1]-agg[0]:+.4f}  {'HELPS' if agg[-1]>agg[0] else 'no gain'}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
