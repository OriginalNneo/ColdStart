"""
Track 1 (NBSVM) — Wang & Manning 2012 NB-SVM on the wideB representation.
Classical ML only. Judged on the two topic-shift lenses via scratch_lens folds.

Design: for each lens, process ONE fold at a time (keeps RSS low), building the
capped word(1,3)+char_wb(2,6) TF-IDF once, then evaluating every candidate on that
same fold's matrices so deltas are fair. NB log-count-ratio r is computed on TRAIN
rows/labels only; val labels never touch feature fitting.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from scratch_lens import load_data, get_folds, macro_f1, ANCHOR

SEED = 42
t0 = time.time()


def word_vec():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                           sublinear_tf=True)


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                           max_features=300000, sublinear_tf=True)


def nb_r(X, y, cols=None, alpha=1.0):
    """NB log-count-ratio on TRAIN rows only. cols=slice restricts computation to a
    block; features outside cols get r=1 (no reweight). Returns dense (n_feat,) vector."""
    n_feat = X.shape[1]
    r = np.ones(n_feat, dtype=np.float64)
    Xu = X if cols is None else X[:, cols]
    p = alpha + np.asarray(Xu[y == 1].sum(axis=0)).ravel()
    q = alpha + np.asarray(Xu[y == 0].sum(axis=0)).ravel()
    rr = np.log((p / p.sum()) / (q / q.sum()))
    if cols is None:
        r = rr
    else:
        r[cols] = rr
    return r


def apply_r(X, r):
    return X.multiply(sparse.csr_matrix(r.reshape(1, -1))).tocsr()


def svc(C):
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def fit_pred_svc(Xtr, ytr, Xev, C):
    clf = svc(C)
    clf.fit(Xtr, ytr)
    return (clf.decision_function(Xev) > 0).astype(int)


def fit_svc_coef(Xtr, ytr, C):
    clf = svc(C)
    clf.fit(Xtr, ytr)
    return clf.coef_[0].copy(), float(clf.intercept_[0])


def pred_interp(Xev, w, b, beta):
    wbar = np.abs(w).mean()
    wb = (1.0 - beta) * wbar + beta * w
    dec = Xev.dot(wb) + b
    return (dec > 0).astype(int)


def fit_pred_logreg(Xtr, ytr, Xev, C=0.25):
    clf = LogisticRegression(solver="liblinear", class_weight="balanced",
                             C=C, random_state=SEED, max_iter=1000)
    clf.fit(Xtr, ytr)
    return clf.predict(Xev)


# candidate name -> per-lens list of fold scores
CAND = [
    "anchor_noNB_svcC0.25",
    "nb_both_svcC0.25_b1.0",
    "nb_both_svcC0.25_b0.5",
    "nb_both_svcC0.25_b0.25",
    "nb_both_svcC0.15",
    "nb_both_svcC0.5",
    "nb_both_logreg_C0.25",
    "nb_word_only_svcC0.25",
    "nb_char_only_svcC0.25",
]


def _rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MB


def run_lens(name, folds, texts, Y, results):
    import gc
    for fi, (tr, val) in enumerate(folds):
        wv, cv = word_vec(), char_vec()
        Xw_tr = wv.fit_transform(texts[tr]).astype(np.float32)
        Xw_ev = wv.transform(texts[val]).astype(np.float32)
        Xc_tr = cv.fit_transform(texts[tr]).astype(np.float32)
        Xc_ev = cv.transform(texts[val]).astype(np.float32)
        nw = Xw_tr.shape[1]
        Xtr = sparse.hstack([Xw_tr, Xc_tr]).tocsr()
        Xev = sparse.hstack([Xw_ev, Xc_ev]).tocsr()
        del Xw_tr, Xw_ev, Xc_tr, Xc_ev; gc.collect()
        ytr, yval = Y[tr], Y[val]
        wcol, ccol = slice(0, nw), slice(nw, Xtr.shape[1])

        # anchor: no NB
        results["anchor_noNB_svcC0.25"].append(
            macro_f1(yval, fit_pred_svc(Xtr, ytr, Xev, 0.25)))

        # --- NB both block (reuse one reweighted pair for many candidates) ---
        r_both = nb_r(Xtr, ytr).astype(np.float32)
        Xtr_b, Xev_b = apply_r(Xtr, r_both), apply_r(Xev, r_both)
        w, b = fit_svc_coef(Xtr_b, ytr, 0.25)
        results["nb_both_svcC0.25_b1.0"].append(macro_f1(yval, pred_interp(Xev_b, w, b, 1.0)))
        results["nb_both_svcC0.25_b0.5"].append(macro_f1(yval, pred_interp(Xev_b, w, b, 0.5)))
        results["nb_both_svcC0.25_b0.25"].append(macro_f1(yval, pred_interp(Xev_b, w, b, 0.25)))
        results["nb_both_svcC0.15"].append(macro_f1(yval, fit_pred_svc(Xtr_b, ytr, Xev_b, 0.15)))
        results["nb_both_svcC0.5"].append(macro_f1(yval, fit_pred_svc(Xtr_b, ytr, Xev_b, 0.5)))
        results["nb_both_logreg_C0.25"].append(macro_f1(yval, fit_pred_logreg(Xtr_b, ytr, Xev_b)))
        del Xtr_b, Xev_b, r_both; gc.collect()

        # --- NB word-block only ---
        r_word = nb_r(Xtr, ytr, cols=wcol).astype(np.float32)
        Xtr_w, Xev_w = apply_r(Xtr, r_word), apply_r(Xev, r_word)
        results["nb_word_only_svcC0.25"].append(macro_f1(yval, fit_pred_svc(Xtr_w, ytr, Xev_w, 0.25)))
        del Xtr_w, Xev_w, r_word; gc.collect()

        # --- NB char-block only ---
        r_char = nb_r(Xtr, ytr, cols=ccol).astype(np.float32)
        Xtr_c, Xev_c = apply_r(Xtr, r_char), apply_r(Xev, r_char)
        results["nb_char_only_svcC0.25"].append(macro_f1(yval, fit_pred_svc(Xtr_c, ytr, Xev_c, 0.25)))
        del Xtr_c, Xev_c, r_char, Xtr, Xev; gc.collect()

        print(f"  [{name}] fold {fi} done ({time.time()-t0:.0f}s) maxRSS={_rss():.0f}MB", flush=True)


def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    print(f"train={len(texts)} pos={Y.mean():.4f} lensA={len(foldsA)} lensB={len(foldsB)}", flush=True)

    resA = {c: [] for c in CAND}
    resB = {c: [] for c in CAND}
    print("=== LENS A ===", flush=True)
    run_lens("A", foldsA, texts, Y, resA)
    print("=== LENS B ===", flush=True)
    run_lens("B", foldsB, texts, Y, resB)

    anchorA = float(np.mean(resA["anchor_noNB_svcC0.25"]))
    anchorB = float(np.mean(resB["anchor_noNB_svcC0.25"]))
    print(f"\nledger ANCHOR (cached) A={ANCHOR['A']} B={ANCHOR['B']}", flush=True)
    print(f"internal reproduced anchor A={anchorA:.4f} B={anchorB:.4f}\n", flush=True)

    print(f"{'candidate':28s} {'LensA':>7s} {'dA':>8s} {'LensB':>7s} {'dB':>8s}  {'PASS':>4s}")
    best = None
    for c in CAND:
        a = float(np.mean(resA[c])); bb = float(np.mean(resB[c]))
        da = a - anchorA; db = bb - anchorB
        passed = (a > anchorA) and (bb > anchorB)
        tag = "PASS" if passed else "fail"
        if c == "anchor_noNB_svcC0.25":
            tag = "----"
        print(f"{c:28s} {a:7.4f} {da:+8.4f} {bb:7.4f} {db:+8.4f}  {tag:>4s}", flush=True)
        if c != "anchor_noNB_svcC0.25" and passed:
            score = min(da, db)
            if best is None or score > best[1]:
                best = (c, score, a, bb, da, db)

    print("\nfoldsA detail:", {c: [round(x, 4) for x in resA[c]] for c in CAND}, flush=True)
    print("foldsB detail:", {c: [round(x, 4) for x in resB[c]] for c in CAND}, flush=True)

    if best is None:
        print("\nNO CANDIDATE PASSES BOTH LENSES. Null result — no prediction written.", flush=True)
        return
    c, score, a, bb, da, db = best
    print(f"\nBEST PASSING: {c}  LensA={a:.4f}({da:+.4f}) LensB={bb:.4f}({db:+.4f})", flush=True)
    write_prediction(c, texts, Y, test_texts, test_ids)


def write_prediction(cand, texts, Y, test_texts, test_ids):
    """Refit the winning candidate on all 20k and predict test."""
    wv, cv = word_vec(), char_vec()
    Xw = wv.fit_transform(texts).astype(np.float32); Xwt = wv.transform(test_texts).astype(np.float32)
    Xc = cv.fit_transform(texts).astype(np.float32); Xct = cv.transform(test_texts).astype(np.float32)
    nw = Xw.shape[1]
    X = sparse.hstack([Xw, Xc]).tocsr(); Xt = sparse.hstack([Xwt, Xct]).tocsr()

    if cand.startswith("nb_word_only"):
        r = nb_r(X, Y, cols=slice(0, nw))
    elif cand.startswith("nb_char_only"):
        r = nb_r(X, Y, cols=slice(nw, X.shape[1]))
    else:
        r = nb_r(X, Y)
    Xr, Xtr = apply_r(X, r), apply_r(Xt, r)

    if "logreg" in cand:
        pred = fit_pred_logreg(Xr, Y, Xtr)
    elif "_b0.5" in cand or "_b0.25" in cand or "_b1.0" in cand:
        beta = 1.0 if "_b1.0" in cand else (0.5 if "_b0.5" in cand else 0.25)
        w, b = fit_svc_coef(Xr, Y, 0.25)
        pred = pred_interp(Xtr, w, b, beta)
    else:
        C = 0.15 if "C0.15" in cand else (0.5 if "C0.5" in cand else 0.25)
        pred = fit_pred_svc(Xr, Y, Xtr, C)

    import pandas as pd
    pd.DataFrame({"id": test_ids, "label": pred.astype(int)}).to_csv(
        "scratch_agent1_pred.csv", index=False)
    print(f"wrote scratch_agent1_pred.csv rows={len(pred)} pos={pred.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
