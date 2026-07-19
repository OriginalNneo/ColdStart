"""
Task 3 — Diverse base-model screen + prediction-correlation matrix + multi-layer
stacking (ridge / logistic meta-learner), with SHIFT-AWARE leakage-safe evaluation.
================================================================================
SUTD 50.007 ML project (GenAI academic-abstract detection). Classical ML only.

WHY THIS SCRIPT EXISTS
----------------------
The user asked for: (1) a pool of DIVERSE base models, (2) the correlation matrix
of their out-of-fold (OOF) prediction scores to pick a low-redundancy subset, and
(3) multi-layer stacking with a ridge / logistic meta-learner, judged leakage-safely.

CRITICAL CONTEXT (do not forget)
--------------------------------
Train -> test TOPIC SHIFT. Vanilla stratified CV OVERSTATES the real Kaggle LB by
~0.09 (baseline: vanilla ~0.82, real LB 0.7299). The trustworthy proxy is the
CLUSTER-HOLDOUT (hold out whole topic clusters). BUT: this exact proxy once
OVER-RATED a soft-vote ensemble (proxy 0.79 -> real LB 0.71, BELOW the 0.7299
baseline). So a cluster-holdout win for a STACK is necessary but NOT sufficient;
we report it skeptically and prefer gains that come with a SMALL train/val gap.

The pass/fail criterion (task step 4): does the STACK beat the BEST SINGLE BASE
MODEL on CLUSTER-HOLDOUT (measured on identical folds, apples-to-apples)?

Reuses proven factories from Task3_Improved_Model.py (imported, not copied).

Run:  .venv/bin/python Task3_Stacking.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  RidgeClassifierCV)
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

# Proven, importable building blocks from the incumbent search script.
from Task3_Improved_Model import (
    MultiVec, vec_word_char, vec_char_only,
    build_stylo, ROBUST_STYLO_IDX, cluster_folds, macro_f1,
)

warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5          # stratified OOF folds (base-model OOF matrix)
INNER_SPLITS = 3      # inner OOF folds inside each cluster fold (meta training)
SVD_COMPS = 200       # TruncatedSVD dims for the LightGBM leg
PCA_COMPS = 120       # PCA dims for the dense shift-robust leg
np.random.seed(SEED)


# ============================================================================
# BASE MODELS
# ----------------------------------------------------------------------------
# Each base model is a function fn(tr_idx, ev) -> (score, pred):
#   tr_idx : integer indices into the TRAIN arrays to FIT on (fold-train only)
#   ev     : dict of the EVALUATION representations (a slice of train, or test)
#            keys: 'texts', 'dense', 'stylo'
# All vectorizers / SVD / PCA are refit on tr_idx ONLY  ->  no leakage of
# vocabulary / components from held-out or test rows into any fitting step.
# score = continuous decision score (for correlation + meta-learner input)
# pred  = the model's native hard 0/1 label (for individual macro-F1)
# ============================================================================

def base_svc(tr_idx, ev):
    """(a) PROVEN config: word(1,2)+char_wb(3,5) TF-IDF, LinearSVC C=0.25 balanced."""
    mv = MultiVec(vec_word_char)
    Xtr = mv.fit_transform(TEXTS[tr_idx])
    Xev = mv.transform(ev["texts"])
    clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, Y[tr_idx])
    return clf.decision_function(Xev), clf.predict(Xev)


def base_charlr(tr_idx, ev):
    """(b) char_wb(3,5)-ONLY TF-IDF, LogisticRegression balanced (shift-robust style)."""
    mv = MultiVec(vec_char_only)
    Xtr = mv.fit_transform(TEXTS[tr_idx])
    Xev = mv.transform(ev["texts"])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             solver="liblinear", random_state=SEED)
    clf.fit(Xtr, Y[tr_idx])
    return clf.predict_proba(Xev)[:, 1], clf.predict(Xev)


def _vec_word():
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True)]


def base_mnb(tr_idx, ev):
    """(c) MultinomialNB on word(1,2) TF-IDF (a genuinely different decision rule)."""
    mv = MultiVec(_vec_word)
    Xtr = mv.fit_transform(TEXTS[tr_idx])
    Xev = mv.transform(ev["texts"])
    clf = MultinomialNB(alpha=0.1)
    clf.fit(Xtr, Y[tr_idx])
    return clf.predict_proba(Xev)[:, 1], clf.predict(Xev)


def base_ridge(tr_idx, ev):
    """(d) RidgeClassifier on word(1,2)+char_wb(3,5) TF-IDF (user wants ridge in the mix)."""
    mv = MultiVec(vec_word_char)
    Xtr = mv.fit_transform(TEXTS[tr_idx])
    Xev = mv.transform(ev["texts"])
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, Y[tr_idx])
    return clf.decision_function(Xev), clf.predict(Xev)


def base_lgbm(tr_idx, ev):
    """(e) LightGBM on TruncatedSVD(200) of word+char TF-IDF + robust stylometric feats."""
    mv = MultiVec(vec_word_char)
    Xtr = mv.fit_transform(TEXTS[tr_idx])
    Xev = mv.transform(ev["texts"])
    svd = TruncatedSVD(n_components=SVD_COMPS, random_state=SEED)
    Ztr = np.hstack([svd.fit_transform(Xtr), STYLO[tr_idx]])
    Zev = np.hstack([svd.transform(Xev), ev["stylo"]])
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             class_weight="balanced", random_state=SEED,
                             n_jobs=-1, verbose=-1)
    clf.fit(Ztr, Y[tr_idx])
    return clf.predict_proba(Zev)[:, 1], clf.predict(Zev)


def base_pca(tr_idx, ev):
    """(f) DENSE shift-robust family: StandardScaler->PCA(120)->LogReg on the
    provided 5,000 features. This family scored ABOVE its val estimate on the
    real LB, so it is the most shift-transferable leg for diversity."""
    sc = StandardScaler()
    Dtr = sc.fit_transform(DENSE[tr_idx])
    Dev = sc.transform(ev["dense"])
    pca = PCA(n_components=PCA_COMPS, svd_solver="randomized", random_state=SEED)
    Ptr = pca.fit_transform(Dtr)
    Pev = pca.transform(Dev)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             random_state=SEED)
    clf.fit(Ptr, Y[tr_idx])
    return clf.predict_proba(Pev)[:, 1], clf.predict(Pev)


BASE_MODELS = {
    "SVC_wc":  base_svc,     # proven word+char LinearSVC
    "CharLR":  base_charlr,  # char-only LogReg
    "MNB_w":   base_mnb,     # word MultinomialNB
    "Ridge_wc": base_ridge,  # word+char RidgeClassifier
    "LGBM_svd": base_lgbm,   # LightGBM on SVD + stylo
    "PCA_LR":  base_pca,     # dense PCA LogReg (shift-robust family)
}
MODEL_ORDER = list(BASE_MODELS.keys())


# ---- evaluation-representation helpers -------------------------------------

def ev_train(idx):
    return {"texts": TEXTS[idx], "dense": DENSE[idx], "stylo": STYLO[idx]}


def ev_test():
    return {"texts": TEST_TEXTS, "dense": DENSE_TEST, "stylo": STYLO_TEST}


# ============================================================================
# MAIN
# ============================================================================

def main():
    global TEXTS, Y, DENSE, STYLO, TEST_TEXTS, DENSE_TEST, STYLO_TEST
    t0 = time.time()
    print("=" * 78, flush=True)
    print("TASK 3 — Diverse base screen + correlation matrix + stacking", flush=True)
    print("=" * 78, flush=True)

    # ---- Load & ALIGN on id (guard against silent row-order mismatch) --------
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    TEXTS = train["text"].astype(str).to_numpy()
    Y = train["label"].to_numpy(dtype=int)
    TEST_TEXTS = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()

    tf = pd.read_csv(DATA_DIR / "train_features.csv").set_index("id")
    ttf = pd.read_csv(DATA_DIR / "test_features.csv").set_index("id")
    feat_cols = [c for c in tf.columns if c != "label"]
    # reindex features to EXACTLY the id order of train.csv / test.csv
    DENSE = tf.reindex(train["id"].to_numpy())[feat_cols].to_numpy(dtype=np.float32)
    DENSE_TEST = ttf.reindex(test_ids)[feat_cols].to_numpy(dtype=np.float32)
    assert not np.isnan(DENSE).any() and not np.isnan(DENSE_TEST).any(), "id-align failed"
    print(f"train={len(TEXTS)}  test={len(TEST_TEXTS)}  machine={Y.mean():.1%}  "
          f"dense={DENSE.shape} (aligned on id)", flush=True)

    # ---- stylometric features (deterministic per row -> compute ONCE) --------
    STYLO = build_stylo(TEXTS)[:, ROBUST_STYLO_IDX]
    STYLO_TEST = build_stylo(TEST_TEXTS)[:, ROBUST_STYLO_IDX]

    # ========================================================================
    # STEP 1 — BASE-MODEL OOF via stratified 5-fold  (the OOF matrix)
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[1] BASE-MODEL OUT-OF-FOLD SCORES (stratified 5-fold)", flush=True)
    print("-" * 78, flush=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(Y)), Y))

    oof_score = {n: np.zeros(len(Y)) for n in MODEL_ORDER}
    oof_pred = {n: np.zeros(len(Y), dtype=int) for n in MODEL_ORDER}
    for fi, (tr, val) in enumerate(folds):
        ev = ev_train(val)
        for n in MODEL_ORDER:
            ts = time.time()
            s, p = BASE_MODELS[n](tr, ev)
            oof_score[n][val] = s
            oof_pred[n][val] = p
            print(f"  fold {fi+1}/{N_SPLITS}  {n:<9} ({time.time()-ts:4.0f}s)", flush=True)

    # individual vanilla-CV macro F1 (native predictions)
    van_f1 = {n: macro_f1(Y, oof_pred[n]) for n in MODEL_ORDER}

    # ========================================================================
    # STEP 2 — CORRELATION MATRIX of the OOF scores
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[2] CORRELATION MATRIX of base-model OOF scores (Pearson)", flush=True)
    print("-" * 78, flush=True)
    S = np.column_stack([oof_score[n] for n in MODEL_ORDER])
    corr = np.corrcoef(S, rowvar=False)
    hdr = "           " + "".join(f"{n:>9}" for n in MODEL_ORDER)
    print(hdr, flush=True)
    for i, n in enumerate(MODEL_ORDER):
        row = "".join(f"{corr[i, j]:>9.3f}" for j in range(len(MODEL_ORDER)))
        print(f"  {n:<9}{row}", flush=True)
    print("\n  Individual vanilla-CV macro-F1:", flush=True)
    for n in MODEL_ORDER:
        print(f"    {n:<9} {van_f1[n]:.4f}", flush=True)

    # ---- SELECT a diverse subset: greedy low-correlation, decent F1 ----------
    # Rule: seed with the best individual F1; repeatedly add the candidate that
    # maximizes  (its F1) - (its max correlation to the already-selected set),
    # among candidates with F1 >= floor. Force-include the dense PCA family for
    # representation diversity (it is the only leg that beat its val on the LB).
    print("\n  SELECTION (greedy: high F1, low mutual correlation):", flush=True)
    F1_FLOOR = 0.55
    idx_of = {n: i for i, n in enumerate(MODEL_ORDER)}
    eligible = [n for n in MODEL_ORDER if van_f1[n] >= F1_FLOOR]
    selected = [max(eligible, key=lambda n: van_f1[n])]
    while len(selected) < 4 and len(selected) < len(eligible):
        best, best_score = None, -1e9
        for n in eligible:
            if n in selected:
                continue
            maxc = max(abs(corr[idx_of[n], idx_of[m]]) for m in selected)
            sc = van_f1[n] - maxc          # reward F1, penalize redundancy
            if sc > best_score:
                best, best_score = n, sc
        selected.append(best)
    if "PCA_LR" not in selected and van_f1["PCA_LR"] >= 0.50:
        selected.append("PCA_LR")          # force the shift-robust dense family in
    for n in selected:
        others = [m for m in selected if m != n]
        mc = max((abs(corr[idx_of[n], idx_of[m]]) for m in others), default=0.0)
        print(f"    + {n:<9} F1={van_f1[n]:.4f}  maxcorr-to-others={mc:.3f}", flush=True)
    print(f"    selected subset = {selected}", flush=True)

    # ========================================================================
    # STEP 3+4 — STACKING, evaluated LEAKAGE-SAFELY on the cluster-holdout proxy
    #   Within each outer cluster fold:
    #     - inner stratified OOF on fold-train ONLY  -> fit meta-learners
    #     - refit ALL base models on fold-train      -> score held-out cluster
    #   Base preds AND meta preds are collected on the SAME held clusters, so
    #   base-vs-stack is apples-to-apples on identical rows/folds.
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[3/4] STACKING — leakage-safe cluster-holdout evaluation", flush=True)
    print("-" * 78, flush=True)
    clus, _ = cluster_folds(TEXTS, Y)
    print(f"  cluster folds: {len(clus)} (whole topic clusters held out)", flush=True)

    K = len(selected)
    # accumulators indexed by the held-cluster rows across all outer folds
    ch_base_pred = {n: np.full(len(Y), -1, dtype=int) for n in MODEL_ORDER}
    ch_stack_pred = {"logistic": np.full(len(Y), -1, dtype=int),
                     "ridgecv":  np.full(len(Y), -1, dtype=int)}
    ch_mask = np.zeros(len(Y), dtype=bool)
    # train/val gap tracking for the stack (on cluster folds)
    stack_tr_f1 = {"logistic": [], "ridgecv": []}

    for ci, (tr, val) in enumerate(clus):
        tclock = time.time()
        ch_mask[val] = True
        # --- base models on the held cluster (refit on fold-train) ---
        ev = ev_train(val)
        base_val_score = np.zeros((len(val), K))
        for n in MODEL_ORDER:
            s, p = BASE_MODELS[n](tr, ev)
            ch_base_pred[n][val] = p
            if n in selected:
                base_val_score[:, selected.index(n)] = s

        # --- inner OOF on fold-train ONLY (for meta training) ---
        inner = list(StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                                     random_state=SEED).split(np.zeros(len(tr)), Y[tr]))
        inner_oof = np.zeros((len(tr), K))
        for itr, ival in inner:
            ev_in = ev_train(tr[ival])
            for n in selected:
                s, _ = BASE_MODELS[n](tr[itr], ev_in)
                inner_oof[ival, selected.index(n)] = s

        # standardize meta inputs (fit on inner OOF, apply to held-cluster scores)
        msc = StandardScaler().fit(inner_oof)
        Zin = msc.transform(inner_oof)
        Zval = msc.transform(base_val_score)

        meta_log = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Zin, Y[tr])
        meta_rid = RidgeClassifierCV(alphas=np.logspace(-2, 3, 12)).fit(Zin, Y[tr])
        ch_stack_pred["logistic"][val] = meta_log.predict(Zval)
        ch_stack_pred["ridgecv"][val] = meta_rid.predict(Zval)
        # train-side fit quality (on the inner-OOF the meta saw) for the gap
        stack_tr_f1["logistic"].append(macro_f1(Y[tr], meta_log.predict(Zin)))
        stack_tr_f1["ridgecv"].append(macro_f1(Y[tr], meta_rid.predict(Zin)))
        print(f"  cluster fold {ci+1}/{len(clus)} done ({time.time()-tclock:4.0f}s)", flush=True)

    m = ch_mask
    ch_base_f1 = {n: macro_f1(Y[m], ch_base_pred[n][m]) for n in MODEL_ORDER}
    ch_stack_f1 = {k: macro_f1(Y[m], ch_stack_pred[k][m]) for k in ch_stack_pred}
    best_base = max(selected, key=lambda n: ch_base_f1[n])
    best_base_overall = max(MODEL_ORDER, key=lambda n: ch_base_f1[n])

    # ========================================================================
    # REPORT
    # ========================================================================
    print("\n" + "=" * 78, flush=True)
    print("[R] RESULTS", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'model':<10} {'vanilla-F1':>11} {'cluster-F1':>11}", flush=True)
    for n in MODEL_ORDER:
        tag = " *" if n in selected else "  "
        print(f"  {n:<10} {van_f1[n]:>11.4f} {ch_base_f1[n]:>11.4f}{tag}", flush=True)
    print("  (* = in the stacked subset)", flush=True)

    print(f"\n  Best SINGLE base on cluster-holdout: {best_base_overall} "
          f"= {ch_base_f1[best_base_overall]:.4f}", flush=True)
    print(f"  Best base WITHIN stacked subset:     {best_base} "
          f"= {ch_base_f1[best_base]:.4f}", flush=True)
    print("\n  STACK (meta-learner) cluster-holdout macro-F1 + train/val gap:", flush=True)
    winner_meta, winner_f1 = None, -1
    for k in ("logistic", "ridgecv"):
        tr_mean = float(np.mean(stack_tr_f1[k]))
        gap = tr_mean - ch_stack_f1[k]
        print(f"    meta={k:<9} cluster-F1={ch_stack_f1[k]:.4f}  "
              f"train={tr_mean:.4f}  gap={gap:+.3f}", flush=True)
        if ch_stack_f1[k] > winner_f1:
            winner_meta, winner_f1 = k, ch_stack_f1[k]

    beats_best_base = winner_f1 > ch_base_f1[best_base_overall]
    print(f"\n  Best meta = {winner_meta} (cluster-F1 {winner_f1:.4f}).", flush=True)
    print(f"  Beats best single base ({best_base_overall} {ch_base_f1[best_base_overall]:.4f}) "
          f"on cluster-holdout? {'YES' if beats_best_base else 'NO'}", flush=True)

    # ========================================================================
    # STEP 5 — FINAL refit on all 20k, predict test, write CSV
    #   Meta fits on the FULL-train stratified OOF matrix (step 1), NOT on
    #   full-train self-predictions. Test path: bases refit on all 20k -> meta.
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[5] FINAL REFIT on all 20,000 rows + test prediction", flush=True)
    print("-" * 78, flush=True)
    sel_idx = [MODEL_ORDER.index(n) for n in selected]
    full_oof = S[:, sel_idx]                      # OOF matrix, selected columns
    fsc = StandardScaler().fit(full_oof)
    Zoof = fsc.transform(full_oof)
    if winner_meta == "logistic":
        final_meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Zoof, Y)
    else:
        final_meta = RidgeClassifierCV(alphas=np.logspace(-2, 3, 12)).fit(Zoof, Y)

    all_idx = np.arange(len(Y))
    evt = ev_test()
    test_scores = np.zeros((len(TEST_TEXTS), len(selected)))
    for j, n in enumerate(selected):
        s, _ = BASE_MODELS[n](all_idx, evt)       # base refit on ALL 20k
        test_scores[:, j] = s
        print(f"    refit+predict {n}", flush=True)
    pred = final_meta.predict(fsc.transform(test_scores)).astype(int)

    out = OUT_DIR / "Task3_Stacking_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"\n  wrote {out}  rows={len(pred)}  "
          f"machine={int(pred.sum())} ({pred.mean():.1%})  "
          f"human={int((pred == 0).sum())}", flush=True)
    print(f"\n  total runtime: {time.time()-t0:.0f}s", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
