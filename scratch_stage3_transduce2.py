"""
STAGE 3 — stronger transduction on the STRONG bankstylo base (Iter-17), four-lens.
=================================================================================
Self-training was tuned on the WEAK Stage-1 base (frac0.5, 1 round). The base is now
much stronger (0.79080), so pseudo-labels are cleaner and MORE transduction may recover
more of the remaining shift tax (~0.042 left). Two mechanisms:
  - self-training DEPTH: rounds {1,2,3} x frac {0.5,0.7}   (ref = f0.5_r1 = Iter-17)
  - label SPREADING: graph propagation on SVD(fused) over train+pool+eval (distinct mechanism)

Base rep per fold = stack[1.6word|char] + bank(16)x0.02 + stylo(227)x0.04, IW-weighted.
Non-circular pool/eval. Judge on TOPICAL lenses (A/B/C1). Submit only if a config beats
the f0.5_r1 reference by a REAL topical margin (>~0.005), per the Iter-18 winner's-curse lesson.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.semi_supervised import LabelSpreading

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds, build_dense
from scratch_stage2_features import build_stack, build_bank, clf_
from scratch_stage1_transduce import iw_weights, self_train

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
BANK_S, STYLO_S = 0.02, 0.04
ST_CFGS = [(0.5, 1), (0.5, 2), (0.5, 3), (0.7, 1), (0.7, 2)]
CANDS = [f"st_f{f}_r{r}" for f, r in ST_CFGS] + ["labelspread"]
REF = "st_f0.5_r1"


def label_spread(Ftr, ytr, Fp, Fe):
    """Transductive label propagation on SVD(fused) over train+pool+eval."""
    n = Ftr.shape[0] + Fp.shape[0] + Fe.shape[0]
    k = min(200, Ftr.shape[1] - 1)
    Z = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(
        sparse.vstack([Ftr, Fp, Fe]).tocsr())
    y = np.r_[ytr, -np.ones(Fp.shape[0], int), -np.ones(Fe.shape[0], int)]
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=30).fit(Z, y)
    ev = ls.transduction_[Ftr.shape[0] + Fp.shape[0]:]
    return ev


def eval_lens(name, folds, texts, Y, Dsty_all):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        tr = np.asarray(tr); val = np.asarray(val)
        rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        Xtr, (Xp, Xe) = build_stack(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        w, _ = iw_weights(texts[tr], texts[val])
        # bank + stylo legs
        Btr, (Bp, Be) = build_bank(texts[tr], ytr, [texts[pool_idx], texts[eval_idx]])
        sb = StandardScaler().fit(Btr)
        bk = lambda D: sparse.csr_matrix(sb.transform(D) * BANK_S)
        ss = StandardScaler().fit(Dsty_all[tr])
        st = lambda idx: sparse.csr_matrix(ss.transform(Dsty_all[idx]) * STYLO_S)
        Ftr = sparse.hstack([Xtr, bk(Btr), st(tr)]).tocsr()
        Fp = sparse.hstack([Xp, bk(Bp), st(pool_idx)]).tocsr()
        Fe = sparse.hstack([Xe, bk(Be), st(eval_idx)]).tocsr()

        for f, r in ST_CFGS:
            acc[f"st_f{f}_r{r}"].append(
                self_train(Ftr, ytr, Fp, Fe, yev, frac=f, rounds=r, balanced=True, w_tr=w))
        acc["labelspread"].append(macro_f1(yev, label_spread(Ftr, ytr, Fp, Fe)))
        print(f"  [{name}] f{fi} " +
              " ".join(f"{c.replace('st_','').replace('labelspread','LS')}={acc[c][-1]:.4f}" for c in CANDS) +
              f" ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    Dsty_all = build_dense(texts)
    print(f"stylo precomputed {Dsty_all.shape} ({time.time()-t0:.0f}s)", flush=True)
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    print(f"train={len(texts)} folds " + " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)
    res = {n: eval_lens(n, f, texts, Y, Dsty_all) for n, f in lenses}

    print(f"\n===== Δ vs reference ({REF}); topical A/B/C1 primary =====", flush=True)
    print(f"{'candidate':14s}{'A':>9s}{'B':>9s}{'C1':>9s}{'C2':>9s}{'topical_min':>12s}", flush=True)
    for c in CANDS:
        d = [float((res[n][c] - res[n][REF]).mean()) for n, _ in lenses]
        tag = "  <-REF" if c == REF else ("  *TOPICAL-WIN*" if min(d[:3]) > 0.003 else "")
        print(f"{c:14s}" + "".join(f"{x:+9.4f}" for x in d) + f"{min(d[:3]):+12.4f}{tag}", flush=True)

    print("\nabsolute topical mean (A+B+C1)/3 per candidate:", flush=True)
    for c in CANDS:
        tm = np.mean([res[n][c].mean() for n in ("A", "B", "C1")])
        print(f"  {c:14s} {tm:.4f}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
