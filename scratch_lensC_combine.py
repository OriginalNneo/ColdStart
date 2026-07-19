"""
Decisive experiment: lever-stacking + robustness on TWO new independent lenses.
==============================================================================
Candidates (all on capped-wideB rep unless noted):
  anchor          LinearSVC(C=0.25, bal)                        [current best family]
  ridge09         RidgeClassifier(alpha=0.9, bal)               [lever 1: estimator geometry]
  word16_svc      LinearSVC on word-block x1.6                  [lever 2: block reweight]
  ridge09_word16  RidgeClassifier(0.9) on word-block x1.6       [STACK: lever1 + lever2]
  stylo_fusion    LinearSVC on [wideB | 1.0*StandardScaled stylo dense]  [track-5, high-deflation]

Lenses:
  A  = existing word-unigram KMeans holdout (scratch_lens)
  B  = existing char_wb(3,5) KMeans holdout (scratch_lens)
  C1 = NEW word(2,3) KMeans holdout  (independent topic basis, seed 7)
  C2 = NEW adversarial test-similarity quantile holdout (mimics real train->test shift):
        classify train-vs-test on char features, rank train by P(test-like),
        hold out each of 5 test-likeness quintiles. Validating on the most
        test-like quintile is the closest in-sample proxy for the real shift.

PASS = candidate beats anchor on a lens. We care WHICH candidates hold across
C1/C2 (robust) vs collapse (the deflation tell). Classical ML only.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_agent5_stylo import _features, NFW, NSTRUCT

SEED = 42
N_SPLITS = 5
t0 = time.time()


def word_char_vecs():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                            max_features=300000, sublinear_tf=True)]


def _group_folds_from_clusters(cl, y, k):
    order = np.argsort(-np.bincount(cl, minlength=k))
    grp = np.array([{c: i % N_SPLITS for i, c in enumerate(order)}[c] for c in cl])
    return [(np.where(grp != g)[0], np.where(grp == g)[0]) for g in range(N_SPLITS)
            if len(np.where(grp == g)[0]) and len(np.unique(y[grp == g])) == 2]


def lensC1_folds(texts, y, k=12, seed=7):
    """Independent topic lens: word(2,3) n-gram KMeans (different basis than A/B)."""
    cv = TfidfVectorizer(analyzer="word", ngram_range=(2, 3), min_df=5,
                         max_features=30000, sublinear_tf=True)
    cl = MiniBatchKMeans(k, random_state=seed, n_init=5, batch_size=2048).fit_predict(
        cv.fit_transform(texts))
    return _group_folds_from_clusters(cl, y, k)


def lensC2_folds(texts, test_texts, y, seed=13):
    """Adversarial test-similarity holdout: rank train by P(test-like), quintile holdout."""
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    Xtr = cv.fit_transform(texts)
    Xte = cv.transform(test_texts)
    X = sparse.vstack([Xtr, Xte]).tocsr()
    z = np.r_[np.zeros(len(texts)), np.ones(len(test_texts))]
    clf = LogisticRegression(max_iter=300, C=1.0, random_state=seed).fit(X, z)
    p = clf.predict_proba(Xtr)[:, 1]           # train docs' test-likeness
    order = np.argsort(p)                       # ascending: least->most test-like
    grp = np.empty(len(texts), dtype=int)
    for q in range(N_SPLITS):
        grp[order[q * len(texts) // N_SPLITS:(q + 1) * len(texts) // N_SPLITS]] = q
    folds = [(np.where(grp != g)[0], np.where(grp == g)[0]) for g in range(N_SPLITS)
             if len(np.unique(y[grp == g])) == 2]
    return folds, p


def build_dense(texts):
    return np.vstack([_features(t) for t in texts]).astype(np.float64)


CANDS = ["anchor", "ridge09", "word16_svc", "ridge09_word16", "stylo_fusion"]


def eval_lens(name, folds, texts, Y, D):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        vecs = word_char_vecs()
        Xw_tr = vecs[0].fit_transform(texts[tr]).astype(np.float32)
        Xw_ev = vecs[0].transform(texts[val]).astype(np.float32)
        Xc_tr = vecs[1].fit_transform(texts[tr]).astype(np.float32)
        Xc_ev = vecs[1].transform(texts[val]).astype(np.float32)
        Xa_tr = sparse.hstack([Xw_tr, Xc_tr]).tocsr()
        Xa_ev = sparse.hstack([Xw_ev, Xc_ev]).tocsr()
        Xw16_tr = sparse.hstack([Xw_tr * 1.6, Xc_tr]).tocsr()
        Xw16_ev = sparse.hstack([Xw_ev * 1.6, Xc_ev]).tocsr()
        sc = StandardScaler().fit(D[tr])
        Dtr = sparse.csr_matrix(sc.transform(D[tr]))
        Dev = sparse.csr_matrix(sc.transform(D[val]))
        Xs_tr = sparse.hstack([Xa_tr, Dtr]).tocsr()
        Xs_ev = sparse.hstack([Xa_ev, Dev]).tocsr()

        fits = {
            "anchor": (LinearSVC(C=0.25, class_weight="balanced", random_state=SEED), Xa_tr, Xa_ev),
            "ridge09": (RidgeClassifier(alpha=0.9, class_weight="balanced"), Xa_tr, Xa_ev),
            "word16_svc": (LinearSVC(C=0.25, class_weight="balanced", random_state=SEED), Xw16_tr, Xw16_ev),
            "ridge09_word16": (RidgeClassifier(alpha=0.9, class_weight="balanced"), Xw16_tr, Xw16_ev),
            "stylo_fusion": (LinearSVC(C=0.25, class_weight="balanced", random_state=SEED), Xs_tr, Xs_ev),
        }
        for c, (clf, xt, xe) in fits.items():
            clf.fit(xt, Y[tr])
            acc[c].append(macro_f1(Y[val], clf.predict(xe)))
        print(f"  [{name}] fold {fi} done ({time.time()-t0:.0f}s)", flush=True)
    return {c: (float(np.mean(v)), [round(x, 4) for x in v]) for c, v in acc.items()}


def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    foldsC1 = lensC1_folds(texts, Y)
    foldsC2, p = lensC2_folds(texts, test_texts, Y)
    print(f"train={len(texts)} folds A={len(foldsA)} B={len(foldsB)} C1={len(foldsC1)} C2={len(foldsC2)}", flush=True)
    print(f"C2 test-likeness p: min={p.min():.3f} med={np.median(p):.3f} max={p.max():.3f} "
          f"mean={p.mean():.3f} (0.5=indistinguishable)", flush=True)
    print("building stylo dense block ...", flush=True)
    D = build_dense(texts)
    print(f"  D shape={D.shape} ({time.time()-t0:.0f}s)", flush=True)

    lenses = [("A", foldsA), ("B", foldsB), ("C1", foldsC1), ("C2", foldsC2)]
    res = {}
    for name, folds in lenses:
        res[name] = eval_lens(name, folds, texts, Y, D)

    print("\n================ RESULTS (macro-F1; Δ vs anchor per lens) ================", flush=True)
    header = f"{'candidate':16s}" + "".join(f"{'L'+n:>18s}" for n, _ in lenses)
    print(header, flush=True)
    anch = {n: res[n]["anchor"][0] for n, _ in lenses}
    for c in CANDS:
        row = f"{c:16s}"
        for n, _ in lenses:
            v = res[n][c][0]
            d = v - anch[n]
            row += f"  {v:.4f}({d:+.4f})"
        print(row, flush=True)

    print("\n--- robustness verdict (how many of the 4 lenses each candidate beats anchor on) ---", flush=True)
    for c in CANDS:
        if c == "anchor":
            continue
        wins = [n for n, _ in lenses if res[n][c][0] > anch[n]]
        margins = [round(res[n][c][0] - anch[n], 4) for n, _ in lenses]
        print(f"  {c:16s} beats anchor on {len(wins)}/4 lenses {wins}  margins={margins}", flush=True)

    print("\nper-fold detail:", flush=True)
    for n, _ in lenses:
        print(f"  Lens {n}: " + ", ".join(f"{c}={res[n][c][1]}" for c in CANDS), flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
