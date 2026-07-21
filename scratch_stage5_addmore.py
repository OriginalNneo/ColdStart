"""
STAGE 5 — does ADDING MORE orthogonal legs keep growing the stack? Four-lens ablation.
=====================================================================================
Answers: (a) precise per-leg contribution on top of Iter-17 (bank+stylo); (b) whether
stacking a SECOND new orthogonal leg (word-shape / casing patterns) on top of the
function-word (fw) leg keeps adding or plateaus.

Legs (all on top of Iter-17 base = stack + bank x0.02 + stylo x0.04; IW + frac0.5 self-train):
  iter17         [ref]
  + fw           function-word skeleton TF-IDF word(1,3)   (Iter-20 syntactic leg)
  + shape        word-SHAPE n-grams: token -> C(apitalized)/w(ord)/U(pper)/0(num)/punct,
                 word(1,3) on the shape sequence = casing/formatting style (NEW, orthogonal)
  + fw + shape   both new legs stacked

If (+fw+shape) > (+fw) by a real topical margin, adding more still helps -> path to 0.80.
If it plateaus at ~fw alone, the orthogonal-feature axis is saturated too. Classical, no DL.
"""
import re, time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds, build_dense
from scratch_stage2_features import build_stack, build_bank, clf_
from scratch_stage1_transduce import iw_weights, self_train

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
BANK_S, STYLO_S, FW_W, SHAPE_W = 0.02, 0.04, 1.0, 1.0
CANDS = ["iter17", "fw", "shape", "fw_shape"]
REF = "iter17"

_tok = re.compile(r"[A-Za-z']+|[.,;:!?()\-]|\d+")
_FW = set(w.lower() for w in ENGLISH_STOP_WORDS)


def skeleton(t):
    out = []
    for m in _tok.findall(t):
        if m[:1].isalpha():
            out.append(m.lower() if m.lower() in _FW else "#")
        else:
            out.append(m)
    return " ".join(out)


def shape(t):
    out = []
    for m in _tok.findall(t):
        if m.isdigit():
            out.append("0")
        elif m.isalpha():
            if m.isupper() and len(m) > 1:
                out.append("U")
            elif m[0].isupper():
                out.append("C")
            else:
                out.append("w")
        else:
            out.append(m)
    return " ".join(out)


def eval_lens(name, folds, texts, Y, Dsty, Skel, Shp):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        tr = np.asarray(tr); val = np.asarray(val)
        rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        Xtr, (Xp, Xe) = build_stack(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        w, _ = iw_weights(texts[tr], texts[val])
        Btr, (Bp, Be) = build_bank(texts[tr], ytr, [texts[pool_idx], texts[eval_idx]])
        sb = StandardScaler().fit(Btr)
        bk = lambda D: sparse.csr_matrix(sb.transform(D) * BANK_S)
        ss = StandardScaler().fit(Dsty[tr])
        stf = lambda idx: sparse.csr_matrix(ss.transform(Dsty[idx]) * STYLO_S)
        I17_tr = sparse.hstack([Xtr, bk(Btr), stf(tr)]).tocsr()
        I17_p = sparse.hstack([Xp, bk(Bp), stf(pool_idx)]).tocsr()
        I17_e = sparse.hstack([Xe, bk(Be), stf(eval_idx)]).tocsr()

        fv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
        fw_tr = fv.fit_transform(Skel[tr]).astype(np.float32) * FW_W
        fw_p = fv.transform(Skel[pool_idx]).astype(np.float32) * FW_W
        fw_e = fv.transform(Skel[eval_idx]).astype(np.float32) * FW_W
        hv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
        sh_tr = hv.fit_transform(Shp[tr]).astype(np.float32) * SHAPE_W
        sh_p = hv.transform(Shp[pool_idx]).astype(np.float32) * SHAPE_W
        sh_e = hv.transform(Shp[eval_idx]).astype(np.float32) * SHAPE_W

        def run(extra_tr, extra_p, extra_e):
            Gt = sparse.hstack([I17_tr] + extra_tr).tocsr()
            Gp = sparse.hstack([I17_p] + extra_p).tocsr()
            Ge = sparse.hstack([I17_e] + extra_e).tocsr()
            return self_train(Gt, ytr, Gp, Ge, yev, frac=0.5, rounds=1, w_tr=w)

        acc["iter17"].append(run([], [], []))
        acc["fw"].append(run([fw_tr], [fw_p], [fw_e]))
        acc["shape"].append(run([sh_tr], [sh_p], [sh_e]))
        acc["fw_shape"].append(run([fw_tr, sh_tr], [fw_p, sh_p], [fw_e, sh_e]))
        print(f"  [{name}] f{fi} " + " ".join(f"{c}={acc[c][-1]:.4f}" for c in CANDS) +
              f" ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    Dsty = build_dense(texts)
    Skel = np.array([skeleton(t) for t in texts], dtype=object)
    Shp = np.array([shape(t) for t in texts], dtype=object)
    print(f"precomputed stylo/skeleton/shape ({time.time()-t0:.0f}s)", flush=True)
    print("shape example:", shape(texts[0])[:90], flush=True)
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    res = {n: eval_lens(n, f, texts, Y, Dsty, Skel, Shp) for n, f in lenses}

    print(f"\n===== Δ vs {REF} (topical A/B/C1 primary) =====", flush=True)
    print(f"{'candidate':10s}{'A':>9s}{'B':>9s}{'C1':>9s}{'C2':>9s}{'topical_min':>12s}", flush=True)
    D = {}
    for c in CANDS:
        d = [float((res[n][c] - res[n][REF]).mean()) for n, _ in lenses]
        D[c] = d
        print(f"{c:10s}" + "".join(f"{x:+9.4f}" for x in d) + f"{min(d[:3]):+12.4f}", flush=True)
    print("\n--- does 2nd leg add on top of fw? (fw_shape vs fw, per lens) ---", flush=True)
    inc = [D["fw_shape"][i] - D["fw"][i] for i in range(4)]
    print(f"  fw_shape − fw:  A{inc[0]:+.4f} B{inc[1]:+.4f} C1{inc[2]:+.4f} C2{inc[3]:+.4f}  "
          f"topical_min {min(inc[:3]):+.4f}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
