"""
STAGE 2 — topic-invariant feature bank fused on the linear stack, four-lens gated.
=================================================================================
Iter 4 found n-gram PERPLEXITY / log-likelihood-ratio (LLR) features are the most
TOPIC-ROBUST signal measured (smallest train−cluster gap) but "5 features can't
match a vocabulary" and a naive append DISTORTED the SVM. Fix the magnitude AND the
fusion: a proper StandardScaled, scale-tuned bank of class-conditional LLR + style
features that describe HOW text is written (topic-invariant by construction), added
as a dense leg on the base stack.

Feature bank (all fit on TRAIN ONLY within each fold — no leakage):
  - class-conditional char n-gram (3,4) & word n-gram (1,2) mean log-prob under the
    human-LM and machine-LM, and their LLR difference          (12 feats: 4 orders × 3)
  - gzip compression ratio                                      (1)
  - type/token ratio, repeated-bigram rate, hapax ratio         (3)
Total 16 dense, topic-invariant features. Deliberately NOT the 227-dim stylo block
(that is the deferred high-deflation gamble); this bank is the low-risk new signal.

Candidates (vs same base stack; FULL held-out cluster = eval, matches Iter-9 anchor):
  base        RidgeClassifier(0.9,bal) on [1.6*word | char]
  llr_s0.3/0.6/1.0   base stack + scale·StandardScaled(bank)

GATE: min mean-Δ over {A,B,C1,C2} > 0. Reports C2 (shift-probe) explicitly — dense
legs are the class that historically deflates on real submission, so holding on C2
is the strongest offline evidence available (still not proof; final check = LB).
"""
import time, gzip, re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds

SEED = 42
t0 = time.time()
SCALES = [0.02, 0.03, 0.04]  # small-scale sweet spot: bank helps under shift, distorts if larger
CANDS = ["base"] + [f"llr_s{s}" for s in SCALES]
_word = re.compile(r"[A-Za-z']+")


def clf_():
    return RidgeClassifier(alpha=0.9, class_weight="balanced")


def stack_vecs():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                            max_features=300000, sublinear_tf=True)]


def build_stack(texts_tr, others, ws=1.6):
    v = stack_vecs()
    Xw = v[0].fit_transform(texts_tr).astype(np.float32)
    Xc = v[1].fit_transform(texts_tr).astype(np.float32)
    Xt = sparse.hstack([Xw * ws, Xc]).tocsr()
    outs = [sparse.hstack([v[0].transform(T).astype(np.float32) * ws,
                           v[1].transform(T).astype(np.float32)]).tocsr() for T in others]
    return Xt, outs


def _class_logprob(cv, texts_tr, ytr):
    """Laplace-smoothed class-conditional log-prob vectors for a fitted count vec."""
    Xtr = cv.transform(texts_tr)
    V = Xtr.shape[1]
    logp = {}
    for c in (0, 1):
        cnt = np.asarray(Xtr[ytr == c].sum(axis=0)).ravel()
        logp[c] = np.log((cnt + 1.0) / (cnt.sum() + V))
    return logp


def _doc_meanlogp(cv, texts, logp_c):
    X = cv.transform(texts)
    tot = np.asarray(X.sum(axis=1)).ravel() + 1e-9
    return (X.dot(logp_c) / tot)


def _cheap_stats(texts):
    out = np.zeros((len(texts), 4))
    for i, t in enumerate(texts):
        b = t.encode("utf-8", "ignore")
        gz = len(gzip.compress(b, 4)) / (len(b) + 1e-9)
        w = _word.findall(t.lower())
        n = len(w) + 1e-9
        types = len(set(w))
        bg = list(zip(w, w[1:]))
        rep_bg = 1.0 - (len(set(bg)) / (len(bg) + 1e-9)) if bg else 0.0
        from collections import Counter
        hap = sum(1 for _, c in Counter(w).items() if c == 1) / n
        out[i] = [gz, types / n, rep_bg, hap]
    return out


def build_bank(texts_tr, ytr, others):
    """16 topic-invariant features; LMs fit on train only. Returns dense arrays."""
    specs = [("char", (3, 3), 200000), ("char", (4, 4), 300000),
             ("word", (1, 1), 100000), ("word", (2, 2), 200000)]
    cvs, logps = [], []
    for an, ng, mf in specs:
        cv = CountVectorizer(analyzer=an if an == "word" else "char_wb",
                             ngram_range=ng, min_df=3, max_features=mf)
        cv.fit(texts_tr)
        cvs.append(cv); logps.append(_class_logprob(cv, texts_tr, ytr))

    def feats(texts):
        cols = []
        for cv, lp in zip(cvs, logps):
            lh = _doc_meanlogp(cv, texts, lp[0])   # human
            lm = _doc_meanlogp(cv, texts, lp[1])   # machine
            cols += [lh, lm, lh - lm]              # human lp, machine lp, LLR
        base = np.column_stack(cols)
        return np.hstack([base, _cheap_stats(texts)])

    return feats(texts_tr), [feats(T) for T in others]


def eval_lens(name, folds, texts, Y):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        tr = np.asarray(tr); val = np.asarray(val)
        Xt, (Xv,) = build_stack(texts[tr], [texts[val]])
        ytr, yv = Y[tr], Y[val]
        Dtr, (Dv,) = build_bank(texts[tr], ytr, [texts[val]])
        sc = StandardScaler().fit(Dtr)
        Ztr, Zv = sc.transform(Dtr), sc.transform(Dv)

        acc["base"].append(macro_f1(yv, clf_().fit(Xt, ytr).predict(Xv)))
        for s in SCALES:
            Xt2 = sparse.hstack([Xt, sparse.csr_matrix(Ztr * s)]).tocsr()
            Xv2 = sparse.hstack([Xv, sparse.csr_matrix(Zv * s)]).tocsr()
            acc[f"llr_s{s}"].append(macro_f1(yv, clf_().fit(Xt2, ytr).predict(Xv2)))
        print(f"  [{name}] f{fi} base={acc['base'][-1]:.4f} " +
              " ".join(f"s{s}={acc[f'llr_s{s}'][-1]:.4f}" for s in SCALES) +
              f" ({time.time()-t0:.0f}s)", flush=True)
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

    print("\nabsolute per-lens base vs best scale:", flush=True)
    for n, _ in lenses:
        print(f"  Lens {n}: " + ", ".join(f"{c}={res[n][c].mean():.4f}" for c in CANDS), flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
