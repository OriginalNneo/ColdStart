"""
STAGE 2b — expand the topic-invariant leg: add the 227-dim stylo block on top of
the Iter-16 winner (bank_iwst), four-lens gated. Push toward 0.80.
================================================================================
LB just proved (Iter 16 = 0.77913): tiny dense topic-invariant fusion does NOT
deflate, and the TOPICAL lenses (A/B/C1) are the faithful real-test proxy. The
227-dim stylo block (`scratch_agent5_stylo`) had the highest proxy ever measured
but was deferred purely on deflation fear — now retired. Does it ADD on top of
bank_iwst?

Candidates (four lenses, non-circular pool/eval, all with IW + frac0.5 1-round self-train):
  base            plain stack (no leg, no transduction)
  bank_iwst       stack + bank(16)x0.02 + IW + ST                 [Iter-16 = current best 0.77913]
  bankstylo_iwst  stack + bank(16)x0.02 + stylo(227)x{0.01,0.02,0.04} + IW + ST

Judge PRIMARILY on topical lenses A/B/C1 (C2 = conservative floor, per Iter-16 finding).
Reports Δ vs base AND Δ vs bank_iwst (does stylo add incrementally?). Classical, no DL.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds, build_dense
from scratch_stage2_features import build_stack, build_bank, clf_
from scratch_stage1_transduce import iw_weights, self_train

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
BANK_S = 0.02
STYLO_S = [0.01, 0.02, 0.04]
CANDS = ["base", "bank_iwst"] + [f"bankstylo_s{s}" for s in STYLO_S]


def fuse(*blocks):
    return sparse.hstack(blocks).tocsr()


def eval_lens(name, folds, texts, Y, Dsty_all):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        tr = np.asarray(tr); val = np.asarray(val)
        rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        Xtr, (Xp, Xe) = build_stack(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        w, _ = iw_weights(texts[tr], texts[val])

        # bank leg (per-fold LMs)
        Btr, (Bp, Be) = build_bank(texts[tr], ytr, [texts[pool_idx], texts[eval_idx]])
        sb = StandardScaler().fit(Btr)
        bk = lambda D: sparse.csr_matrix(sb.transform(D) * BANK_S)
        Bktr, Bkp, Bke = bk(Btr), bk(Bp), bk(Be)

        # stylo leg (precomputed per-doc, scaled per-fold)
        ss = StandardScaler().fit(Dsty_all[tr])
        st = lambda idx, s: sparse.csr_matrix(ss.transform(Dsty_all[idx]) * s)

        # base
        acc["base"].append(macro_f1(yev, clf_().fit(Xtr, ytr).predict(Xe)))
        # bank_iwst (current best)
        Ftr, Fp, Fe = fuse(Xtr, Bktr), fuse(Xp, Bkp), fuse(Xe, Bke)
        acc["bank_iwst"].append(self_train(Ftr, ytr, Fp, Fe, yev, frac=0.5, rounds=1, w_tr=w))
        # + stylo at each scale
        for s in STYLO_S:
            Gtr = fuse(Xtr, Bktr, st(tr, s))
            Gp = fuse(Xp, Bkp, st(pool_idx, s))
            Ge = fuse(Xe, Bke, st(eval_idx, s))
            acc[f"bankstylo_s{s}"].append(self_train(Gtr, ytr, Gp, Ge, yev, frac=0.5, rounds=1, w_tr=w))
        print(f"  [{name}] f{fi} base={acc['base'][-1]:.4f} bankiwst={acc['bank_iwst'][-1]:.4f} " +
              " ".join(f"sty{s}={acc[f'bankstylo_s{s}'][-1]:.4f}" for s in STYLO_S) +
              f" ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    print("precomputing 227-dim stylo block for all train (once) ...", flush=True)
    Dsty_all = build_dense(texts)
    print(f"  stylo shape={Dsty_all.shape} ({time.time()-t0:.0f}s)", flush=True)
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    print(f"train={len(texts)} bank_s={BANK_S} stylo_s={STYLO_S} folds " +
          " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)
    res = {n: eval_lens(n, f, texts, Y, Dsty_all) for n, f in lenses}

    print("\n===== Δ vs base (topical A/B/C1 primary; C2 = floor) =====", flush=True)
    print(f"{'candidate':16s}{'A':>9s}{'B':>9s}{'C1':>9s}{'C2':>9s}{'min':>9s}{'topical_min':>12s}  gate",
          flush=True)
    means = {}
    for c in CANDS:
        if c == "base":
            continue
        m = [float((res[n][c] - res[n]["base"]).mean()) for n, _ in lenses]
        means[c] = m
        tmin = min(m[:3])                      # topical min (A,B,C1)
        gate = "PASS" if min(m) > 0 else ("PASS*" if tmin > 0 else "fail")
        print(f"{c:16s}" + "".join(f"{x:+9.4f}" for x in m) +
              f"{min(m):+9.4f}{tmin:+12.4f}  [{gate}]", flush=True)

    print("\n--- incremental: does stylo ADD on top of bank_iwst? (Δ vs bank_iwst per lens) ---", flush=True)
    bi = means["bank_iwst"]
    for s in STYLO_S:
        inc = [means[f"bankstylo_s{s}"][i] - bi[i] for i in range(4)]
        print(f"  stylo_s{s}: A{inc[0]:+.4f} B{inc[1]:+.4f} C1{inc[2]:+.4f} C2{inc[3]:+.4f}  "
              f"topical_min {min(inc[:3]):+.4f}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
