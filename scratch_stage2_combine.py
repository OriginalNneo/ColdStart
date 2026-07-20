"""
STAGE 2 payoff — do the two complementary levers STACK? Four-lens gated.
=======================================================================
Iter 12 iw_selftrain (transductive) and Iter 15 llr_s0.02 (topic-invariant feature
bank) have OPPOSITE failure modes — the bank is strongly positive on the exact hard
folds (C1-f4, B-f4) where transduction craters. If the signals are complementary they
should ADD. This tests additivity on all four lenses with the non-circular pool/eval
protocol (transduction needs it).

Candidates (vs base stack, eval-half of each held-out cluster):
  base        RidgeClassifier(0.9,bal) on [1.6*word | char]
  iwst        base + IW weighting + 1 round frac0.5 self-train            (Iter 12)
  bank        base + StandardScaled(16-feat LLR/style bank) * 0.02        (Iter 15)
  bank_iwst   base + bank + IW + self-train                              (the stack)

Reads additivity as: is Δ(bank_iwst) ≈ Δ(bank) + Δ(iwst)? min over 4 lenses gates it.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds
from scratch_stage2_features import build_stack, build_bank, clf_
from scratch_stage1_transduce import iw_weights, self_train

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
BANK_S = 0.02
CANDS = ["base", "iwst", "bank", "bank_iwst"]


def fuse(X, Z, s):
    return sparse.hstack([X, sparse.csr_matrix(Z * s)]).tocsr()


def eval_lens(name, folds, texts, Y):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        tr = np.asarray(tr); val = np.asarray(val)
        rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        Xtr, (Xp, Xe) = build_stack(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        Dtr, (Dp, De) = build_bank(texts[tr], ytr, [texts[pool_idx], texts[eval_idx]])
        scl = StandardScaler().fit(Dtr)
        Ztr, Zp, Ze = scl.transform(Dtr), scl.transform(Dp), scl.transform(De)
        Ftr, Fp, Fe = fuse(Xtr, Ztr, BANK_S), fuse(Xp, Zp, BANK_S), fuse(Xe, Ze, BANK_S)
        w, _ = iw_weights(texts[tr], texts[val])

        acc["base"].append(macro_f1(yev, clf_().fit(Xtr, ytr).predict(Xe)))
        acc["iwst"].append(self_train(Xtr, ytr, Xp, Xe, yev, frac=0.5, rounds=1, w_tr=w))
        acc["bank"].append(macro_f1(yev, clf_().fit(Ftr, ytr).predict(Fe)))
        acc["bank_iwst"].append(self_train(Ftr, ytr, Fp, Fe, yev, frac=0.5, rounds=1, w_tr=w))
        print(f"  [{name}] f{fi} base={acc['base'][-1]:.4f} iwst={acc['iwst'][-1]:.4f} "
              f"bank={acc['bank'][-1]:.4f} both={acc['bank_iwst'][-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    print(f"train={len(texts)} bank_scale={BANK_S} folds " +
          " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)
    res = {n: eval_lens(n, f, texts, Y) for n, f in lenses}

    print("\n===== Δ vs base (mean per lens; min = four-lens gate) =====", flush=True)
    print(f"{'candidate':12s}{'A':>10s}{'B':>10s}{'C1':>10s}{'C2':>10s}{'min':>10s}{'worst':>10s}  gate",
          flush=True)
    D = {}
    for c in CANDS:
        if c == "base":
            continue
        means = [float((res[n][c] - res[n]["base"]).mean()) for n, _ in lenses]
        worst = min(float((res[n][c] - res[n]["base"]).min()) for n, _ in lenses)
        D[c] = means
        gate = "PASS" if min(means) > 0 else "fail"
        print(f"{c:12s}" + "".join(f"{m:+10.4f}" for m in means) +
              f"{min(means):+10.4f}{worst:+10.4f}  [{gate}]", flush=True)

    print("\n--- additivity check: bank_iwst vs (bank + iwst) per lens ---", flush=True)
    for i, (n, _) in enumerate(lenses):
        add = D["bank"][i] + D["iwst"][i]
        got = D["bank_iwst"][i]
        print(f"  Lens {n}: bank {D['bank'][i]:+.4f} + iwst {D['iwst'][i]:+.4f} = {add:+.4f}  "
              f"actual {got:+.4f}  ({'super' if got>add else 'sub'}-additive {got-add:+.4f})", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
