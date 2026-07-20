"""
STAGE 1 (1C) — co-training word-view <-> char-view, four-lens gated.
===================================================================
Hypothesis: single-view self-training deflated on the C2 shift-probe because its
pseudo-labels are noisy under shift. Co-training uses the two NATURAL views we
already have — word(1,3) and char_wb(2,6) — to cross-check pseudo-labels, so only
higher-PRECISION pool rows get labeled. Higher precision -> less shift-probe damage.

Candidates (all vs the same base stack, non-circular pool/eval protocol, 4 lenses):
  base          RidgeClassifier(0.9,bal) on [1.6*word | char]                (the 0.752 model)
  agree_st      agreement-gated self-training: label a pool row only if BOTH views
                agree on its class AND both are in their top-frac confident; refit
                the stack on train + those (few, clean) pseudo-labels             (1C robust)
  cotrain       classic co-training: each view's confident picks teach the OTHER
                view for rounds; final stack refit on train + union of picks       (1C classic)

GATE: min mean-Δ over {A,B,C1,C2} > 0 with non-catastrophic worst fold. No DL.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds
from scratch_stage1_transduce import select

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
CANDS = ["base", "agree_st", "cotrain"]


def clf_():
    return RidgeClassifier(alpha=0.9, class_weight="balanced")


def build_views(texts_tr, others, ws=1.6):
    """Return per-view (word, char) and stacked matrices for train + each 'others'."""
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                         max_features=300000, sublinear_tf=True)
    Xw = vw.fit_transform(texts_tr).astype(np.float32)
    Xc = vc.fit_transform(texts_tr).astype(np.float32)
    Xs = sparse.hstack([Xw * ws, Xc]).tocsr()
    outs = []
    for T in others:
        Ow = vw.transform(T).astype(np.float32)
        Oc = vc.transform(T).astype(np.float32)
        outs.append((Ow, Oc, sparse.hstack([Ow * ws, Oc]).tocsr()))
    return (Xw, Xc, Xs), outs


def agree_st(train, pool, ev, ytr, yev, frac=0.5, rounds=3):
    (Xw, Xc, Xs) = train; (Pw, Pc, Ps) = pool; (_, _, Es) = ev
    cw = clf_().fit(Xw, ytr); cc = clf_().fit(Xc, ytr); cs = clf_().fit(Xs, ytr)
    for _ in range(rounds):
        mw, mc = cw.decision_function(Pw), cc.decision_function(Pc)
        tw, plw = select(mw, frac, True)
        tc, plc = select(mc, frac, True)
        take = tw & tc & (plw == plc)                    # both confident AND agree
        if take.sum() == 0:
            break
        Xs2 = sparse.vstack([Xs, Ps[take]]).tocsr(); ys2 = np.r_[ytr, plw[take]]
        cs = clf_().fit(Xs2, ys2)
        cw = clf_().fit(sparse.vstack([Xw, Pw[take]]).tocsr(), ys2)
        cc = clf_().fit(sparse.vstack([Xc, Pc[take]]).tocsr(), ys2)
    return macro_f1(yev, cs.predict(Es))


def cotrain(train, pool, ev, ytr, yev, frac=0.5, rounds=3):
    (Xw, Xc, Xs) = train; (Pw, Pc, Ps) = pool; (_, _, Es) = ev
    cw = clf_().fit(Xw, ytr); cc = clf_().fit(Xc, ytr)
    addW = np.zeros(Ps.shape[0], bool); addC = np.zeros(Ps.shape[0], bool)
    labW = np.zeros(Ps.shape[0], int); labC = np.zeros(Ps.shape[0], int)
    for _ in range(rounds):
        mw, mc = cw.decision_function(Pw), cc.decision_function(Pc)
        tw, plw = select(mw, frac, True)
        tc, plc = select(mc, frac, True)
        # word view teaches char view, and vice versa
        addC |= tw; labC = np.where(tw, plw, labC)
        addW |= tc; labW = np.where(tc, plc, labW)
        cw = clf_().fit(sparse.vstack([Xw, Pw[addW]]).tocsr(), np.r_[ytr, labW[addW]])
        cc = clf_().fit(sparse.vstack([Xc, Pc[addC]]).tocsr(), np.r_[ytr, labC[addC]])
    # final stack on train + union of taught rows (prefer agreement; else either)
    union = addW | addC
    lab = np.where(addW & addC, labW, np.where(addW, labW, labC))
    Xs2 = sparse.vstack([Xs, Ps[union]]).tocsr(); ys2 = np.r_[ytr, lab[union]]
    cs = clf_().fit(Xs2, ys2)
    return macro_f1(yev, cs.predict(Es))


def eval_lens(name, folds, texts, Y):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        val = np.array(val); rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        train, (pool, ev) = build_views(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        acc["base"].append(macro_f1(yev, clf_().fit(train[2], ytr).predict(ev[2])))
        acc["agree_st"].append(agree_st(train, pool, ev, ytr, yev))
        acc["cotrain"].append(cotrain(train, pool, ev, ytr, yev))
        print(f"  [{name}] fold {fi} base={acc['base'][-1]:.4f} "
              f"agree={acc['agree_st'][-1]:.4f} cotrain={acc['cotrain'][-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    print(f"train={len(texts)} folds " + " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)
    res = {n: eval_lens(n, f, texts, Y) for n, f in lenses}

    print("\n===== Δ vs base (mean per lens; min = four-lens gate) =====", flush=True)
    print(f"{'candidate':12s}{'A':>10s}{'B':>10s}{'C1':>10s}{'C2':>10s}{'min':>10s}{'worst':>10s}  gate",
          flush=True)
    for c in CANDS:
        if c == "base":
            continue
        means = [float((res[n][c] - res[n]["base"]).mean()) for n, _ in lenses]
        worst = min(float((res[n][c] - res[n]["base"]).min()) for n, _ in lenses)
        gate = "PASS" if min(means) > 0 else "fail"
        print(f"{c:12s}" + "".join(f"{m:+10.4f}" for m in means) +
              f"{min(means):+10.4f}{worst:+10.4f}  [{gate}]", flush=True)

    print("\nper-fold detail:", flush=True)
    for n, _ in lenses:
        print(f"  Lens {n}: " + ", ".join(
            f"{c}={[round(x,4) for x in res[n][c]]}" for c in CANDS), flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
