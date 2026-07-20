"""
STAGE 4 — function-word / syntactic-skeleton leg: genuinely NEW signal, four-lens.
=================================================================================
Both tunable axes (feature scale Iter-18, transduction depth Iter-19) are saturated.
Need a NEW signal orthogonal to everything we have (char/word content TF-IDF, LLR bank,
stylo COUNT features). Function-word SEQUENCE n-grams capture SYNTAX / word-order —
strongly topic-invariant (no content vocabulary) and a classic authorship/AI-detection
signal. Skeleton = replace every content word with '#', keep function words + punctuation;
TF-IDF word(1,3) on that skeleton = syntactic-pattern features.

Candidates on top of bankstylo (Iter-17, the current best 0.79080), non-circular pool/eval,
IW + frac0.5 1-round self-train:
  bankstylo_iwst   [ref = Iter-17]
  + fw leg x{0.5,1.0,1.5}   (sparse TF-IDF on the function-word skeleton)

Judge on TOPICAL lenses; submit only if topical margin >> proxy noise (~>0.005), per Iter-18/19.
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
BANK_S, STYLO_S = 0.02, 0.04
FW_W = [0.5, 1.0, 1.5]
CANDS = ["bankstylo_iwst"] + [f"fw_w{w}" for w in FW_W]
REF = "bankstylo_iwst"

_tok = re.compile(r"[A-Za-z']+|[.,;:!?()\-]")
_FW = set(w.lower() for w in ENGLISH_STOP_WORDS)


def skeleton(t):
    out = []
    for m in _tok.findall(t):
        if m.isalpha():
            out.append(m.lower() if m.lower() in _FW else "#")
        else:
            out.append(m)
    return " ".join(out)


def eval_lens(name, folds, texts, Y, Dsty_all, Skel_all):
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
        stf = lambda idx: sparse.csr_matrix(ss.transform(Dsty_all[idx]) * STYLO_S)
        BS_tr = sparse.hstack([Xtr, bk(Btr), stf(tr)]).tocsr()
        BS_p = sparse.hstack([Xp, bk(Bp), stf(pool_idx)]).tocsr()
        BS_e = sparse.hstack([Xe, bk(Be), stf(eval_idx)]).tocsr()
        # function-word skeleton leg (fit on train skeletons)
        fv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
        Fw_tr = fv.fit_transform(Skel_all[tr]).astype(np.float32)
        Fw_p = fv.transform(Skel_all[pool_idx]).astype(np.float32)
        Fw_e = fv.transform(Skel_all[eval_idx]).astype(np.float32)

        acc[REF].append(self_train(BS_tr, ytr, BS_p, BS_e, yev, frac=0.5, rounds=1, w_tr=w))
        for wt in FW_W:
            Gtr = sparse.hstack([BS_tr, Fw_tr * wt]).tocsr()
            Gp = sparse.hstack([BS_p, Fw_p * wt]).tocsr()
            Ge = sparse.hstack([BS_e, Fw_e * wt]).tocsr()
            acc[f"fw_w{wt}"].append(self_train(Gtr, ytr, Gp, Ge, yev, frac=0.5, rounds=1, w_tr=w))
        print(f"  [{name}] f{fi} " +
              " ".join(f"{c.replace('bankstylo_iwst','REF')}={acc[c][-1]:.4f}" for c in CANDS) +
              f" ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    Dsty_all = build_dense(texts)
    print(f"stylo {Dsty_all.shape} ({time.time()-t0:.0f}s); building skeletons ...", flush=True)
    Skel_all = np.array([skeleton(t) for t in texts], dtype=object)
    print(f"skeletons built ({time.time()-t0:.0f}s)", flush=True)
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    print(f"train={len(texts)} fw_weights={FW_W} folds " +
          " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)
    res = {n: eval_lens(n, f, texts, Y, Dsty_all, Skel_all) for n, f in lenses}

    print(f"\n===== Δ vs {REF} (Iter-17); topical A/B/C1 primary =====", flush=True)
    print(f"{'candidate':14s}{'A':>9s}{'B':>9s}{'C1':>9s}{'C2':>9s}{'topical_min':>12s}", flush=True)
    for c in CANDS:
        d = [float((res[n][c] - res[n][REF]).mean()) for n, _ in lenses]
        tag = "  <-REF" if c == REF else ("  *WIN*" if min(d[:3]) > 0.003 else "")
        print(f"{c:14s}" + "".join(f"{x:+9.4f}" for x in d) + f"{min(d[:3]):+12.4f}{tag}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
