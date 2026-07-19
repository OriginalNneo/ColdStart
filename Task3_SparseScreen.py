"""
Task 3 — SPARSE-TEXT MODEL SCREEN (agent B, overnight run 2026-07-15).

Goal: find a better SINGLE sparse-text model than LinearSVC under the trusted
5-fold cluster-holdout proxy (Task3_Improved_Model.cluster_folds, SEED=42).
Incumbents on this proxy: BASE (word 1-2 + char_wb 3-5 TF-IDF -> LinearSVC
C=0.25 balanced) = 0.7383; + importance weighting (IW, Task3_PseudoLabel
machinery: transductive vocab + adversarial sample weights) = 0.7523.

Protocol:
  Phase 1  reproduce BASE and IW on all 5 folds (stop if BASE reproduction fails).
  Phase 2  screen every candidate on folds 0 and 1, plain AND with IW machinery
           (IW variant = transductive vocab + adversarial weights, exactly as in
           Task3_PseudoLabel).
  Phase 3  full 5-fold for top-3 configs (their better variant).
  Phase 4  refit overall best on all train rows, write
           predictions/Task3_SparseScreen_Prediction.csv (+ _probs.npy if the
           model yields probabilities).

Run: nohup .venv/bin/python Task3_SparseScreen.py > scratch_sparse_screen.log 2>&1 &
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from Task3_Improved_Model import DATA_DIR, OUT_DIR, SEED, cluster_folds, macro_f1, est_svc
from Task3_PseudoLabel import C_BASE, adversarial_train_weights

REPRO_BASE = 0.7383
REPRO_IW = 0.7523
REPRO_TOL = 0.005
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Representations (identical vectorizer params to the proven baseline)
# ---------------------------------------------------------------------------
VEC_PARAMS = {
    "word": dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    "char35": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    "char25": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
}
REP_PARTS = {"wc": ["word", "char35"], "w": ["word"], "c35": ["char35"], "c25": ["char25"]}


def build_rep(rep, ttr, tval, transductive):
    """Return (Xtr, Xval). transductive=True fits vocab/idf on ttr+tval text
    (no labels), exactly like Task3_PseudoLabel's VOCAB/IW representation."""
    mats_tr, mats_val = [], []
    for part in REP_PARTS[rep]:
        v = TfidfVectorizer(**VEC_PARAMS[part])
        if transductive:
            Xall = v.fit_transform(np.concatenate([ttr, tval]))
            mats_tr.append(Xall[: len(ttr)])
            mats_val.append(Xall[len(ttr):])
        else:
            mats_tr.append(v.fit_transform(ttr))
            mats_val.append(v.transform(tval))
    if len(mats_tr) > 1:
        return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_val).tocsr()
    return mats_tr[0].tocsr(), mats_val[0].tocsr()


# ---------------------------------------------------------------------------
# NBSVM (Wang & Manning 2012) on TF-IDF: log-count-ratio r from BINARIZED
# features, scale X by r, fit LinearSVC / LogisticRegression, interpolate
# w' = b*w_bar + (1-b)*w  (w_bar = mean |w|; b=0 -> plain fit on r-scaled X).
# ---------------------------------------------------------------------------
class NBSVM:
    def __init__(self, base="svc", b=0.0, C=0.25, nb_alpha=1.0):
        self.base, self.b, self.C, self.nb_alpha = base, b, C, nb_alpha

    def fit(self, X, y, sample_weight=None):
        Xb = (X > 0)
        p = np.asarray(Xb[y == 1].sum(axis=0)).ravel() + self.nb_alpha
        q = np.asarray(Xb[y == 0].sum(axis=0)).ravel() + self.nb_alpha
        self.r = np.log(p / p.sum()) - np.log(q / q.sum())
        Xr = X.multiply(self.r).tocsr()
        if self.base == "svc":
            clf = LinearSVC(C=self.C, class_weight="balanced", random_state=SEED,
                            max_iter=3000)
        else:
            clf = LogisticRegression(C=self.C, class_weight="balanced",
                                     solver="liblinear", max_iter=2000,
                                     random_state=SEED)
        clf.fit(Xr, y, sample_weight=sample_weight)
        w = clf.coef_.ravel()
        w_bar = np.abs(w).mean()
        self.w = self.b * w_bar + (1.0 - self.b) * w
        self.bias = float(clf.intercept_[0])
        return self

    def decision_function(self, X):
        return X.multiply(self.r).tocsr() @ self.w + self.bias

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X):  # meaningful for base='lr'
        z = np.clip(self.decision_function(X), -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.c_[1 - p, p]


def char_svc(C):
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=3000)


def lr_wc(C):
    return LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                              max_iter=2000, random_state=SEED)


def sgd(alpha):
    return SGDClassifier(loss="modified_huber", alpha=alpha,
                         class_weight="balanced", random_state=SEED)


CONFIGS = [
    # name, rep, factory, has_probabilities
    ("NBSVM-SVC-b0",   "wc",  lambda: NBSVM("svc", b=0.0,  C=0.25), False),
    ("NBSVM-SVC-b25",  "wc",  lambda: NBSVM("svc", b=0.25, C=0.25), False),
    ("NBSVM-LR-b0",    "wc",  lambda: NBSVM("lr",  b=0.0,  C=1.0),  True),
    ("NBSVM-LR-b25",   "wc",  lambda: NBSVM("lr",  b=0.25, C=1.0),  True),
    ("CNB-a0.1-w",     "w",   lambda: ComplementNB(alpha=0.1),      True),
    ("CNB-a0.3-w",     "w",   lambda: ComplementNB(alpha=0.3),      True),
    ("CNB-a1.0-w",     "w",   lambda: ComplementNB(alpha=1.0),      True),
    ("CNB-a0.1-wc",    "wc",  lambda: ComplementNB(alpha=0.1),      True),
    ("CNB-a0.3-wc",    "wc",  lambda: ComplementNB(alpha=0.3),      True),
    ("CNB-a1.0-wc",    "wc",  lambda: ComplementNB(alpha=1.0),      True),
    ("SGD-mh-a1e-5",   "wc",  lambda: sgd(1e-5),                    True),
    ("SGD-mh-a1e-4",   "wc",  lambda: sgd(1e-4),                    True),
    ("SVC-c25-C0.25",  "c25", lambda: char_svc(0.25),               False),
    ("SVC-c25-C0.5",   "c25", lambda: char_svc(0.5),                False),
    ("SVC-c25-C1",     "c25", lambda: char_svc(1.0),                False),
    ("SVC-c35-C0.25",  "c35", lambda: char_svc(0.25),               False),
    ("SVC-c35-C0.5",   "c35", lambda: char_svc(0.5),                False),
    ("SVC-c35-C1",     "c35", lambda: char_svc(1.0),                False),
    ("LR-C1-wc",       "wc",  lambda: lr_wc(1.0),                   True),
    ("LR-C4-wc",       "wc",  lambda: lr_wc(4.0),                   True),
    ("LR-C16-wc",      "wc",  lambda: lr_wc(16.0),                  True),
]
CONFIG_BY_NAME = {n: (n, r, f, p) for n, r, f, p in CONFIGS}


def fit_model(factory, Xtr, ytr, sw):
    m = factory()
    if sw is not None:
        m.fit(Xtr, ytr, sample_weight=sw)
    else:
        m.fit(Xtr, ytr)
    return m


def eval_fold(fold_k, ttr, tval, ytr, yval, wanted, w_fold):
    """wanted: list of (name, variant) with variant in {'plain','iw'}.
    Returns dict (name, variant) -> macro F1. Builds each representation once
    per variant and frees it afterwards."""
    out = {}
    for variant in ("plain", "iw"):
        todo = [nv for nv in wanted if nv[1] == variant]
        if not todo:
            continue
        transductive = variant == "iw"
        sw = w_fold if transductive else None
        by_rep = {}
        for name, _ in todo:
            by_rep.setdefault(CONFIG_BY_NAME[name][1], []).append(name)
        for rep, names in by_rep.items():
            t = time.time()
            Xtr, Xval = build_rep(rep, ttr, tval, transductive)
            log(f"  fold{fold_k} {variant:5s} rep={rep:3s} built "
                f"{Xtr.shape} in {time.time()-t:.0f}s")
            for name in names:
                t = time.time()
                m = fit_model(CONFIG_BY_NAME[name][2], Xtr, ytr, sw)
                f1 = macro_f1(yval, m.predict(Xval))
                out[(name, variant)] = f1
                log(f"    fold{fold_k} {name:15s} [{variant}] F1={f1:.4f} "
                    f"({time.time()-t:.0f}s)")
            del Xtr, Xval
    return out


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    log(f"train={len(texts)} test={len(test_texts)} pos={y.mean():.3f}")

    folds, _ = cluster_folds(texts, y)
    log(f"cluster-holdout folds: {len(folds)} "
        f"(sizes={[len(v) for _, v in folds]})")

    # ---------------- Phase 1: reproduce BASE and IW on all 5 folds ----------
    log("PHASE 1: reproduce BASE (target 0.7383) and IW (target 0.7523)")
    base_scores, iw_scores, w_cache = [], [], {}
    for k, (tr, val) in enumerate(folds):
        ttr, tval, ytr, yval = texts[tr], texts[val], y[tr], y[val]
        Xtr, Xval = build_rep("wc", ttr, tval, transductive=False)
        clf = est_svc(C_BASE).fit(Xtr, ytr)
        base_scores.append(macro_f1(yval, clf.predict(Xval)))
        del Xtr, Xval
        w_cache[k] = adversarial_train_weights(ttr, tval)
        Xtr, Xval = build_rep("wc", ttr, tval, transductive=True)
        clf = est_svc(C_BASE)
        clf.fit(Xtr, ytr, sample_weight=w_cache[k])
        iw_scores.append(macro_f1(yval, clf.predict(Xval)))
        del Xtr, Xval
        log(f"  fold{k}: BASE={base_scores[-1]:.4f}  IW={iw_scores[-1]:.4f}")
    base_mean, iw_mean = float(np.mean(base_scores)), float(np.mean(iw_scores))
    log(f"REPRODUCTION: BASE mean={base_mean:.4f} (target {REPRO_BASE})  "
        f"IW mean={iw_mean:.4f} (target {REPRO_IW})")
    if abs(base_mean - REPRO_BASE) > REPRO_TOL:
        log("=== REPRO FAILED — STOPPING (BASE mean deviates > 0.005) ===")
        log("=== DONE (FAILURE) ===")
        sys.exit(1)
    log("BASE reproduction OK.")

    # ---------------- Phase 2: screen all candidates on folds 0 and 1 --------
    log("PHASE 2: screening all candidates on folds 0 and 1 (plain + IW)")
    screen = {}
    wanted_all = [(n, v) for n, _, _, _ in CONFIGS for v in ("plain", "iw")]
    for k in (0, 1):
        tr, val = folds[k]
        res = eval_fold(k, texts[tr], texts[val], y[tr], y[val],
                        wanted_all, w_cache[k])
        for key, f1 in res.items():
            screen.setdefault(key, {})[k] = f1

    log("SCREEN TABLE (fold0 / fold1 / mean2):")
    means2 = {}
    for (name, variant), d in sorted(screen.items()):
        m2 = (d[0] + d[1]) / 2
        means2[(name, variant)] = m2
        log(f"  {name:15s} [{variant:5s}]  f0={d[0]:.4f}  f1={d[1]:.4f}  "
            f"mean2={m2:.4f}")

    best_variant = {}
    for name, _, _, _ in CONFIGS:
        v = max(("plain", "iw"), key=lambda vv: means2[(name, vv)])
        best_variant[name] = (v, means2[(name, v)])
    finalists = sorted(best_variant.items(), key=lambda kv: -kv[1][1])[:3]
    log("FINALISTS (top-3 by mean over folds 0,1, best variant): " +
        ", ".join(f"{n}[{v}]={m:.4f}" for n, (v, m) in finalists))

    # ---------------- Phase 3: full 5-fold for finalists ---------------------
    log("PHASE 3: folds 2-4 for finalists")
    final_scores = {n: {0: screen[(n, v)][0], 1: screen[(n, v)][1]}
                    for n, (v, _) in finalists}
    for k in (2, 3, 4):
        tr, val = folds[k]
        wanted = [(n, v) for n, (v, _) in finalists]
        res = eval_fold(k, texts[tr], texts[val], y[tr], y[val],
                        wanted, w_cache[k])
        for (name, _), f1 in res.items():
            final_scores[name][k] = f1

    log("FINALIST 5-FOLD RESULTS (vs BASE %.4f, IW incumbent %.4f):"
        % (base_mean, iw_mean))
    log("  per-fold incumbent IW: " + " ".join(f"{s:.4f}" for s in iw_scores))
    summary = []
    for name, (v, _) in finalists:
        per = [final_scores[name][k] for k in range(5)]
        mean5 = float(np.mean(per))
        summary.append((name, v, per, mean5))
        log(f"  {name:15s} [{v:5s}] " + " ".join(f"{s:.4f}" for s in per) +
            f"  mean={mean5:.4f}  (delta vs IW {mean5-iw_mean:+.4f})")
    summary.sort(key=lambda t: -t[3])
    win_name, win_variant, win_per, win_mean = summary[0]
    log(f"WINNER: {win_name} [{win_variant}]  5-fold={win_mean:.4f}  "
        f"projected LB={win_mean-0.008:.4f}")
    if win_mean <= iw_mean:
        log("NOTE: winner does NOT beat the IW incumbent (0.7523 proxy) — "
            "null result; still writing its prediction file as requested.")

    # ---------------- Phase 4: refit on full train, write predictions --------
    log(f"PHASE 4: refit {win_name} [{win_variant}] on all {len(texts)} rows")
    _, rep, factory, has_proba = CONFIG_BY_NAME[win_name]
    if win_variant == "iw":
        sw = adversarial_train_weights(texts, test_texts)
        Xtr, Xte = build_rep(rep, texts, test_texts, transductive=True)
    else:
        sw = None
        Xtr, Xte = build_rep(rep, texts, test_texts, transductive=False)
    model = fit_model(factory, Xtr, y, sw)
    pred = np.asarray(model.predict(Xte)).astype(int)

    out_csv = OUT_DIR / "Task3_SparseScreen_Prediction.csv"
    pd.DataFrame({"id": test["id"], "label": pred}).to_csv(out_csv, index=False)
    log(f"wrote {out_csv} rows={len(pred)} machine={int(pred.sum())} "
        f"human={int((pred == 0).sum())}")
    if has_proba:
        probs = model.predict_proba(Xte)[:, 1]
        np.save(OUT_DIR / "Task3_SparseScreen_probs.npy", probs)
        log(f"wrote {OUT_DIR/'Task3_SparseScreen_probs.npy'} "
            f"(P(label=1), mean={probs.mean():.4f})")
    else:
        log("winner has no probability output; no probs .npy written")

    # verify CSV
    chk = pd.read_csv(out_csv, dtype={"id": str})
    assert len(chk) == 6999, f"row count {len(chk)}"
    assert (chk["id"].to_numpy() == test["id"].to_numpy()).all(), "id mismatch"
    assert set(chk["label"].unique()) <= {0, 1}, "bad labels"
    log("CSV VERIFIED: 6999 rows, ids match data/test.csv order, labels in {0,1}")

    log(f"total runtime {time.time()-T0:.0f}s")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
