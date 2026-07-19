"""
Round 2 — optimize the winning stack + test stylo-on-stack. All 4-lens gated.
============================================================================
Best so far (Iter 9, real 0.75210): Ridge(alpha=0.9,bal) on [1.6*word | char].
Questions:
  (Q1) Is (alpha=0.9, word_scale=1.6) the local optimum, or can we nudge it?
  (Q2) Does the strong stylo signal (+0.018 proxy) ADD on top of the stack,
       or is it redundant with / dominated by the sparse levers?

Capped-wideB rep for speed; shared per-fold vectorization (vectorize once/fold,
fit every candidate on the same matrices). Lenses A,B from scratch_lens; C1
(word(2,3) KMeans) and C2 (adversarial test-similarity) from scratch_lensC_combine.
Winners get re-confirmed uncapped before any submission. Classical ML only.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds, build_dense

SEED = 42
t0 = time.time()

# stack grid around the (0.9, 1.6) optimum
ALPHAS = [0.8, 0.9, 1.0]
WSCALES = [1.5, 1.6, 1.8]
STYLO_SCALES = [0.5, 1.0]   # for stack+stylo


def word_char_vecs():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                            max_features=300000, sublinear_tf=True)]


def build_candidates():
    """name -> (kind, params). kind in {svc_anchor, ridge_stack, svc_stylo, ridge_stack_stylo}."""
    cands = {"anchor": ("svc", 1.0, None, None)}
    for a in ALPHAS:
        for ws in WSCALES:
            cands[f"stack_a{a}_w{ws}"] = ("ridge", ws, a, None)
    cands["stylo_fusion"] = ("svc", 1.0, None, 1.0)             # anchor rep + stylo
    for ss in STYLO_SCALES:
        cands[f"stackstylo_a0.9_w1.6_s{ss}"] = ("ridge", 1.6, 0.9, ss)
    return cands


def fit_eval(kind, ws, alpha, sscale, Xw_tr, Xc_tr, Xw_ev, Xc_ev, Dtr, Dev, ytr, yval):
    blocks_tr = [Xw_tr * ws, Xc_tr]
    blocks_ev = [Xw_ev * ws, Xc_ev]
    if sscale is not None:
        blocks_tr.append(Dtr * sscale)
        blocks_ev.append(Dev * sscale)
    Xt = sparse.hstack(blocks_tr).tocsr()
    Xe = sparse.hstack(blocks_ev).tocsr()
    if kind == "svc":
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
    else:
        clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
    clf.fit(Xt, ytr)
    return macro_f1(yval, clf.predict(Xe))


def eval_lens(name, folds, texts, Y, D, cands):
    acc = {c: [] for c in cands}
    for fi, (tr, val) in enumerate(folds):
        vecs = word_char_vecs()
        Xw_tr = vecs[0].fit_transform(texts[tr]).astype(np.float32)
        Xw_ev = vecs[0].transform(texts[val]).astype(np.float32)
        Xc_tr = vecs[1].fit_transform(texts[tr]).astype(np.float32)
        Xc_ev = vecs[1].transform(texts[val]).astype(np.float32)
        sc = StandardScaler().fit(D[tr])
        Dtr = sparse.csr_matrix(sc.transform(D[tr]).astype(np.float32))
        Dev = sparse.csr_matrix(sc.transform(D[val]).astype(np.float32))
        for c, (kind, ws, alpha, sscale) in cands.items():
            acc[c].append(fit_eval(kind, ws, alpha, sscale, Xw_tr, Xc_tr, Xw_ev, Xc_ev,
                                   Dtr, Dev, Y[tr], Y[val]))
        print(f"  [{name}] fold {fi} done ({time.time()-t0:.0f}s)", flush=True)
    return {c: (float(np.mean(v)), [round(x, 4) for x in v]) for c, v in acc.items()}


def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    foldsC1 = lensC1_folds(texts, Y)
    foldsC2, _ = lensC2_folds(texts, test_texts, Y)
    print(f"train={len(texts)} folds A={len(foldsA)} B={len(foldsB)} C1={len(foldsC1)} C2={len(foldsC2)}", flush=True)
    D = build_dense(texts)
    print(f"stylo D={D.shape} ({time.time()-t0:.0f}s)", flush=True)

    cands = build_candidates()
    lenses = [("A", foldsA), ("B", foldsB), ("C1", foldsC1), ("C2", foldsC2)]
    res = {n: eval_lens(n, f, texts, Y, D, cands) for n, f in lenses}

    anch = {n: res[n]["anchor"][0] for n, _ in lenses}
    print("\n============ RESULTS (Δ vs anchor per lens; min-margin across 4) ============", flush=True)
    print(f"{'candidate':28s}" + "".join(f"{'L'+n:>10s}" for n, _ in lenses) + f"{'min':>9s}", flush=True)
    rows = []
    for c in cands:
        ds = [res[n][c][0] - anch[n] for n, _ in lenses]
        mm = min(ds)
        rows.append((c, ds, mm))
    # anchor first, then by min-margin desc
    rows.sort(key=lambda r: (r[0] != "anchor", -r[2]))
    for c, ds, mm in rows:
        print(f"{c:28s}" + "".join(f"{d:+10.4f}" for d in ds) + f"{mm:+9.4f}"
              + ("  <-- current" if c == "stack_a0.9_w1.6" else ""), flush=True)

    winners = [r for r in rows if r[0] != "anchor" and r[2] > 0]
    print(f"\n4/4-lens passers (min-margin>0): {len(winners)}", flush=True)
    for c, ds, mm in winners[:6]:
        print(f"  {c:28s} min={mm:+.4f} margins={[round(d,4) for d in ds]}", flush=True)
    best = winners[0][0] if winners else "none"
    print(f"\nBEST by min-margin: {best}", flush=True)
    print("\nabsolute macro-F1 per lens (best + current + stylo):", flush=True)
    for c in [best, "stack_a0.9_w1.6", "stylo_fusion", "anchor"]:
        if c in res["A"]:
            print(f"  {c:28s} " + " ".join(f"{n}={res[n][c][0]:.4f}" for n, _ in lenses), flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
