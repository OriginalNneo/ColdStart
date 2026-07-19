"""
Task 3 — Stacked ensemble: LogReg + LinearSVM + RandomForest  ->  XGBoost meta.
================================================================================
SUTD 50.007 ML project (GenAI academic-abstract detection). Classical ML only
(no deep learning) — SVM / LogReg / RandomForest base learners, XGBoost judge.

WHAT THIS IMPLEMENTS (the user's requested architecture)
-------------------------------------------------------
Step 1  Feature extraction:
          - word(1,2) + char_wb(3,5) TF-IDF n-grams  (sparse; for the linear legs)
          - TruncatedSVD(200) of that TF-IDF + robust stylometric features
            (dense; for the RandomForest leg, which can't use raw sparse n-grams)
Step 2  Base learners, trained independently, each seeing the data differently:
          - LogReg       : calibrated linear probs on sparse TF-IDF
          - LinearSVM    : max-margin boundary on sparse TF-IDF
          - RandomForest : bagged trees on the dense SVD+stylo view
Step 3  Meta-ensemble (STACKING): the 3 base out-of-fold (OOF) scores become the
        input features to an XGBoost "judge" that learns which base to trust when.
        Gaussian NOISE is injected into the meta-features while training the judge
        (a regularizer, as requested) so it can't latch onto tiny OOF quirks.
Step 4  ONE honest test, reported TWO ways:
          (a) vanilla stratified 5-fold  -> the OPTIMISTIC number
          (b) cluster-holdout (hold out whole topic clusters) -> the REALISTIC
              number under this competition's known train->test topic shift.

WHY BOTH NUMBERS (read before trusting any result)
--------------------------------------------------
This dataset has a train->test TOPIC SHIFT. Vanilla CV here overstates the real
Kaggle leaderboard by ~0.09 (baseline: vanilla ~0.82, real LB 0.7299). Five prior
ensemble/stack attempts ALL looked great on vanilla CV (0.79-0.85) and then landed
BELOW the plain LinearSVC baseline (0.7299) on the real leaderboard. So the vanilla
number is shown only to make the inflation visible; the cluster-holdout number
(and the base-vs-stack comparison on IDENTICAL folds) is the one that means
something. A stack "winning" on vanilla CV is the expected trap, not a success.

Reuses proven, leakage-safe primitives from Task3_Improved_Model.py (imported).
Each fold vectorizes the TF-IDF ONCE and shares it across all three legs (~3x
faster than refitting per leg).

Run:  .venv/bin/python Task3_XGBStack.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Proven, importable, leakage-safe building blocks from the incumbent search.
from Task3_Improved_Model import (
    MultiVec, vec_word_char, build_stylo, ROBUST_STYLO_IDX,
    cluster_folds, macro_f1,
)

warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5        # vanilla stratified OOF folds
INNER_SPLITS = 3    # inner OOF folds inside each cluster fold (meta training)
SVD_COMPS = 200     # TruncatedSVD dims for the RandomForest leg
NOISE_SD = 0.10     # Gaussian noise sd added to (standardized) meta-features
MODEL_ORDER = ["LogReg", "LinearSVM", "RandomForest"]
K = len(MODEL_ORDER)
np.random.seed(SEED)

# Shared RNG for reproducible noise injection (deterministic, resume-safe).
RNG = np.random.RandomState(SEED)


# ============================================================================
# THREE BASE LEARNERS, sharing one TF-IDF fit per fold.
#   fit_legs(tr_idx, ev_idx=None, ev_texts/ev_stylo for test) -> per-leg
#   (score, hard_pred) on the eval rows. Vectorizer / SVD refit on tr_idx ONLY.
#   score = continuous decision score (meta input); pred = native 0/1 label.
# ============================================================================

def fit_legs(tr_idx, ev_texts, ev_stylo):
    """Fit LogReg + LinearSVM + RandomForest on tr_idx, score the eval rows.
    Returns {name: (score, pred)}. TF-IDF is built ONCE and shared."""
    mv = MultiVec(vec_word_char)
    Xtr = mv.fit(TEXTS[tr_idx]).transform(TEXTS[tr_idx])
    Xev = mv.transform(ev_texts)
    ytr = Y[tr_idx]

    out = {}

    # LogReg — calibrated linear probabilities on sparse TF-IDF
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                            solver="liblinear", random_state=SEED).fit(Xtr, ytr)
    out["LogReg"] = (lr.predict_proba(Xev)[:, 1], lr.predict(Xev))

    # LinearSVM — max-margin boundary (proven C=0.25 balanced)
    sv = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, ytr)
    out["LinearSVM"] = (sv.decision_function(Xev), sv.predict(Xev))

    # RandomForest — bagged trees on the dense SVD(TF-IDF)+stylo view
    svd = TruncatedSVD(n_components=SVD_COMPS, random_state=SEED)
    Ztr = np.hstack([svd.fit_transform(Xtr), STYLO[tr_idx]])
    Zev = np.hstack([svd.transform(Xev), ev_stylo])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED).fit(Ztr, ytr)
    out["RandomForest"] = (rf.predict_proba(Zev)[:, 1], rf.predict(Zev))
    return out


def make_meta(seed=SEED):
    """XGBoost judge over the 3 base scores. Shallow + subsampled: regularized on
    purpose so it corrects base blind spots without memorizing OOF noise."""
    return XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=seed,
    )


def fit_meta(meta_train_X, meta_train_y):
    """Standardize the 3 meta-features, inject Gaussian noise (regularizer),
    fit the XGBoost judge. Returns (scaler, fitted_meta)."""
    sc = StandardScaler().fit(meta_train_X)
    Z = sc.transform(meta_train_X)
    Z_noisy = Z + RNG.normal(0.0, NOISE_SD, size=Z.shape)   # <-- induced noise
    meta = make_meta().fit(Z_noisy, meta_train_y)
    return sc, meta


# ============================================================================
def main():
    global TEXTS, Y, STYLO, TEST_TEXTS, STYLO_TEST
    t0 = time.time()
    print("=" * 78, flush=True)
    print("TASK 3 — LogReg + LinearSVM + RandomForest  ->  XGBoost stack", flush=True)
    print("=" * 78, flush=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    TEXTS = train["text"].astype(str).to_numpy()
    Y = train["label"].to_numpy(dtype=int)
    TEST_TEXTS = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    STYLO = build_stylo(TEXTS)[:, ROBUST_STYLO_IDX]
    STYLO_TEST = build_stylo(TEST_TEXTS)[:, ROBUST_STYLO_IDX]
    print(f"train={len(TEXTS)}  test={len(TEST_TEXTS)}  machine={Y.mean():.1%}  "
          f"stylo={STYLO.shape}", flush=True)

    # ========================================================================
    # [1] VANILLA stratified 5-fold OOF  -> OPTIMISTIC number
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[1] VANILLA stratified 5-fold OOF (optimistic — ignores topic shift)", flush=True)
    print("-" * 78, flush=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(Y)), Y))

    oof_score = np.zeros((len(Y), K))
    oof_pred = {n: np.zeros(len(Y), dtype=int) for n in MODEL_ORDER}
    for fi, (tr, val) in enumerate(folds):
        ts = time.time()
        res = fit_legs(tr, TEXTS[val], STYLO[val])
        for j, n in enumerate(MODEL_ORDER):
            oof_score[val, j] = res[n][0]
            oof_pred[n][val] = res[n][1]
        print(f"  fold {fi+1}/{N_SPLITS}  all 3 legs ({time.time()-ts:4.0f}s)", flush=True)
    van_base_f1 = {n: macro_f1(Y, oof_pred[n]) for n in MODEL_ORDER}

    # base-score correlations (how much diversity the judge has to work with)
    corr = np.corrcoef(oof_score, rowvar=False)
    print("\n  OOF-score correlation (lower = more diverse -> better stacking):", flush=True)
    print("               " + "".join(f"{n:>14}" for n in MODEL_ORDER), flush=True)
    for i, n in enumerate(MODEL_ORDER):
        print(f"  {n:<14}" + "".join(f"{corr[i,j]:>14.3f}" for j in range(K)), flush=True)

    # leakage-safe vanilla stack: 2nd stratified split OVER the OOF matrix so the
    # judge never scores a row it trained on.
    stack_oof_pred = np.zeros(len(Y), dtype=int)
    for tr, val in skf.split(oof_score, Y):
        sc, meta = fit_meta(oof_score[tr], Y[tr])
        stack_oof_pred[val] = meta.predict(sc.transform(oof_score[val]))
    van_stack_f1 = macro_f1(Y, stack_oof_pred)

    print("\n  vanilla macro-F1:", flush=True)
    for n in MODEL_ORDER:
        print(f"    {n:<14} {van_base_f1[n]:.4f}", flush=True)
    print(f"    {'XGB-STACK':<14} {van_stack_f1:.4f}", flush=True)

    # ========================================================================
    # [2] CLUSTER-HOLDOUT  -> REALISTIC number (leakage-safe, apples-to-apples)
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[2] CLUSTER-HOLDOUT (hold out whole topic clusters — realistic proxy)", flush=True)
    print("-" * 78, flush=True)
    clus, _ = cluster_folds(TEXTS, Y)
    print(f"  {len(clus)} outer folds (each holds out unseen topics)", flush=True)

    ch_base_pred = {n: np.full(len(Y), -1, dtype=int) for n in MODEL_ORDER}
    ch_stack_pred = np.full(len(Y), -1, dtype=int)
    ch_mask = np.zeros(len(Y), dtype=bool)
    stack_train_f1 = []

    for ci, (tr, val) in enumerate(clus):
        tclock = time.time()
        ch_mask[val] = True

        # base legs: refit on fold-train, score the held-out clusters
        res = fit_legs(tr, TEXTS[val], STYLO[val])
        base_val_score = np.zeros((len(val), K))
        for j, n in enumerate(MODEL_ORDER):
            base_val_score[:, j] = res[n][0]
            ch_base_pred[n][val] = res[n][1]

        # inner OOF on fold-train ONLY -> meta-feature matrix the judge trains on
        inner = list(StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                                     random_state=SEED).split(np.zeros(len(tr)), Y[tr]))
        inner_oof = np.zeros((len(tr), K))
        for itr, ival in inner:
            r = fit_legs(tr[itr], TEXTS[tr[ival]], STYLO[tr[ival]])
            for j, n in enumerate(MODEL_ORDER):
                inner_oof[ival, j] = r[n][0]

        sc, meta = fit_meta(inner_oof, Y[tr])
        ch_stack_pred[val] = meta.predict(sc.transform(base_val_score))
        stack_train_f1.append(macro_f1(Y[tr], meta.predict(sc.transform(inner_oof))))
        print(f"  cluster fold {ci+1}/{len(clus)} done ({time.time()-tclock:4.0f}s)", flush=True)

    m = ch_mask
    ch_base_f1 = {n: macro_f1(Y[m], ch_base_pred[n][m]) for n in MODEL_ORDER}
    ch_stack_f1 = macro_f1(Y[m], ch_stack_pred[m])
    best_base = max(MODEL_ORDER, key=lambda n: ch_base_f1[n])
    stack_gap = float(np.mean(stack_train_f1)) - ch_stack_f1

    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "=" * 78, flush=True)
    print("RESULTS  —  vanilla (optimistic)  vs  cluster-holdout (realistic)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'model':<14} {'vanilla-F1':>11} {'cluster-F1':>11}", flush=True)
    for n in MODEL_ORDER:
        print(f"  {n:<14} {van_base_f1[n]:>11.4f} {ch_base_f1[n]:>11.4f}", flush=True)
    print(f"  {'XGB-STACK':<14} {van_stack_f1:>11.4f} {ch_stack_f1:>11.4f}", flush=True)

    print(f"\n  Best single base (cluster-holdout): {best_base} = {ch_base_f1[best_base]:.4f}", flush=True)
    print(f"  XGB-STACK (cluster-holdout):        {ch_stack_f1:.4f}  "
          f"(train/val gap {stack_gap:+.3f})", flush=True)
    beats = ch_stack_f1 > ch_base_f1[best_base]
    print(f"  Stack beats best single base on the REALISTIC proxy? "
          f"{'YES' if beats else 'NO'}", flush=True)
    print(f"\n  Vanilla->cluster drop for the stack: "
          f"{van_stack_f1 - ch_stack_f1:+.3f}  (this gap = the topic-shift tax; "
          f"the LinearSVC baseline's own real-LB is 0.72990)", flush=True)

    # ========================================================================
    # [3] FINAL refit on all 20k -> test prediction CSV (NOT auto-submitted)
    # ========================================================================
    print("\n" + "-" * 78, flush=True)
    print("[3] FINAL refit on all 20,000 rows + test prediction", flush=True)
    print("-" * 78, flush=True)
    sc_f, meta_f = fit_meta(oof_score, Y)     # judge fits on the full OOF matrix
    all_idx = np.arange(len(Y))
    res = fit_legs(all_idx, TEST_TEXTS, STYLO_TEST)
    test_scores = np.column_stack([res[n][0] for n in MODEL_ORDER])
    pred = meta_f.predict(sc_f.transform(test_scores)).astype(int)

    out = OUT_DIR / "Task3_XGBStack_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"\n  wrote {out}  rows={len(pred)}  "
          f"machine={int(pred.sum())} ({pred.mean():.1%})  "
          f"human={int((pred == 0).sum())}", flush=True)
    print("  (NOT submitted to Kaggle — local test artifact only)", flush=True)
    print(f"\n  total runtime: {time.time()-t0:.0f}s", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
