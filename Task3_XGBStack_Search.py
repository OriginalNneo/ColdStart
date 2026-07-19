"""
Task 3 — OFFLINE hyperparameter search for the LogReg+SVM+RF -> XGBoost stack.
================================================================================
Implements the user's "adjust a param, if worse try the opposite direction, if
worse on both ends find another local minimum" loop as automated COORDINATE
DESCENT on the XGBoost meta-learner, judged by the CLUSTER-HOLDOUT proxy (the
realistic topic-shift proxy — vanilla CV is not trustworthy here).

Two-phase for speed:
  Phase A (~30 min, once): compute the expensive base-leg OOF scores and CACHE
    them (scratch_xgbstack_cache.npz). Each proxy evaluation afterward is just
    re-fitting the cheap 3-input XGBoost judge on cached scores -> milliseconds.
  Phase B (minutes): coordinate-descent hill-climb over the meta hyperparameters
    AND the noise level, from the current defaults. Then a leg-subset search
    (free from the cache) — dropping the shift-fragile RF/stylo leg is the lever
    that actually changes generalization under shift, not the meta knobs.

HONESTY NOTE: the cluster proxy OVER-rates stacks vs the real Kaggle LB by
~0.075-0.08 (documented 3x). Any offline "win" here must be discounted that much.
This search maximizes the proxy; the point is to see whether ANY meta/leg setting
is materially different, not to trust the absolute number.

Run:  .venv/bin/python Task3_XGBStack_Search.py
"""

import time
import itertools
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

from Task3_Improved_Model import (
    MultiVec, vec_word_char, build_stylo, ROBUST_STYLO_IDX,
    cluster_folds, macro_f1,
)

warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
OUT_DIR.mkdir(exist_ok=True)
CACHE = Path("scratch_xgbstack_cache.npz")

N_SPLITS = 5
INNER_SPLITS = 3
SVD_COMPS = 200
MODEL_ORDER = ["LogReg", "LinearSVM", "RandomForest"]
K = len(MODEL_ORDER)
np.random.seed(SEED)


def fit_legs(TEXTS, Y, STYLO, tr_idx, ev_texts, ev_stylo):
    """LogReg + LinearSVM + RandomForest on tr_idx; score eval rows.
    Returns (score_matrix[n_ev,3], pred_matrix[n_ev,3]). TF-IDF shared."""
    mv = MultiVec(vec_word_char)
    Xtr = mv.fit(TEXTS[tr_idx]).transform(TEXTS[tr_idx])
    Xev = mv.transform(ev_texts)
    ytr = Y[tr_idx]
    S = np.zeros((len(ev_texts), K)); P = np.zeros((len(ev_texts), K), dtype=int)

    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                            solver="liblinear", random_state=SEED).fit(Xtr, ytr)
    S[:, 0] = lr.predict_proba(Xev)[:, 1]; P[:, 0] = lr.predict(Xev)
    sv = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, ytr)
    S[:, 1] = sv.decision_function(Xev); P[:, 1] = sv.predict(Xev)
    svd = TruncatedSVD(n_components=SVD_COMPS, random_state=SEED)
    Ztr = np.hstack([svd.fit_transform(Xtr), STYLO[tr_idx]])
    Zev = np.hstack([svd.transform(Xev), ev_stylo])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED).fit(Ztr, ytr)
    S[:, 2] = rf.predict_proba(Zev)[:, 1]; P[:, 2] = rf.predict(Zev)
    return S, P


# ---------------------------------------------------------------------------
# PHASE A — compute & cache base-leg OOF (expensive, once)
# ---------------------------------------------------------------------------
def build_cache():
    t0 = time.time()
    print("[A] computing base-leg OOF (this is the slow part, ~30 min)...", flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    TEXTS = train["text"].astype(str).to_numpy()
    Y = train["label"].to_numpy(dtype=int)
    TEST_TEXTS = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    STYLO = build_stylo(TEXTS)[:, ROBUST_STYLO_IDX]
    STYLO_TEST = build_stylo(TEST_TEXTS)[:, ROBUST_STYLO_IDX]

    # vanilla 5-fold OOF (full-train meta-training matrix + optimistic ref)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_score = np.zeros((len(Y), K))
    oof_pred = np.zeros((len(Y), K), dtype=int)
    for fi, (tr, val) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        S, P = fit_legs(TEXTS, Y, STYLO, tr, TEXTS[val], STYLO[val])
        oof_score[val] = S; oof_pred[val] = P
        print(f"    vanilla fold {fi+1}/{N_SPLITS} ({time.time()-t0:.0f}s)", flush=True)

    # cluster-holdout fold caches: per fold store inner_oof, ytr, val_score, yval
    clus, _ = cluster_folds(TEXTS, Y)
    fold_inner_oof, fold_ytr, fold_val_score, fold_yval, fold_val_pred = [], [], [], [], []
    for ci, (tr, val) in enumerate(clus):
        S_val, P_val = fit_legs(TEXTS, Y, STYLO, tr, TEXTS[val], STYLO[val])
        inner = list(StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                     random_state=SEED).split(np.zeros(len(tr)), Y[tr]))
        inner_oof = np.zeros((len(tr), K))
        for itr, ival in inner:
            s, _ = fit_legs(TEXTS, Y, STYLO, tr[itr], TEXTS[tr[ival]], STYLO[tr[ival]])
            inner_oof[ival] = s
        fold_inner_oof.append(inner_oof); fold_ytr.append(Y[tr])
        fold_val_score.append(S_val); fold_yval.append(Y[val]); fold_val_pred.append(P_val)
        print(f"    cluster fold {ci+1}/{len(clus)} ({time.time()-t0:.0f}s)", flush=True)

    # test base scores (bases refit on all 20k)
    all_idx = np.arange(len(Y))
    test_score, _ = fit_legs(TEXTS, Y, STYLO, all_idx, TEST_TEXTS, STYLO_TEST)

    np.savez(CACHE, allow_pickle=True,
             oof_score=oof_score, oof_pred=oof_pred, Y=Y,
             fold_inner_oof=np.array(fold_inner_oof, dtype=object),
             fold_ytr=np.array(fold_ytr, dtype=object),
             fold_val_score=np.array(fold_val_score, dtype=object),
             fold_yval=np.array(fold_yval, dtype=object),
             fold_val_pred=np.array(fold_val_pred, dtype=object),
             test_score=test_score, test_ids=test_ids)
    print(f"[A] cached -> {CACHE} ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# PHASE B — cheap proxy evaluation over cached OOF
# ---------------------------------------------------------------------------
def proxy_f1(C, cols, params, noise_sd):
    """Cluster-holdout macro-F1 of the XGBoost judge with given meta params,
    noise level, and active leg columns. Deterministic noise per call."""
    rng = np.random.RandomState(SEED)
    yv_all, pv_all = [], []
    for f in range(len(C["fold_inner_oof"])):
        io = C["fold_inner_oof"][f][:, cols]
        vs = C["fold_val_score"][f][:, cols]
        ytr = C["fold_ytr"][f]; yval = C["fold_yval"][f]
        sc = StandardScaler().fit(io)
        Z = sc.transform(io) + rng.normal(0.0, noise_sd, (len(io), len(cols)))
        meta = XGBClassifier(**params, objective="binary:logistic",
                             eval_metric="logloss", tree_method="hist",
                             n_jobs=-1, random_state=SEED).fit(Z, ytr)
        pv_all.append(meta.predict(sc.transform(vs))); yv_all.append(yval)
    return macro_f1(np.concatenate(yv_all), np.concatenate(pv_all))


BASE_PARAMS = dict(n_estimators=250, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   min_child_weight=3)
GRID = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "n_estimators": [100, 250, 400, 600],
    "subsample": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
}
NOISE_GRID = [0.0, 0.05, 0.10, 0.20, 0.30]


def search():
    C = dict(np.load(CACHE, allow_pickle=True))
    cols_all = [0, 1, 2]

    # reference: current defaults, all 3 legs
    base_f1 = proxy_f1(C, cols_all, BASE_PARAMS, 0.10)
    print("\n" + "=" * 74, flush=True)
    print(f"[B] START (all 3 legs, defaults, noise=0.10): cluster-F1 = {base_f1:.4f}", flush=True)
    print("=" * 74, flush=True)

    # ---- coordinate descent over meta params + noise (hill-climb) ----
    cur = dict(BASE_PARAMS); cur_noise = 0.10; cur_f1 = base_f1
    for it in range(2):
        print(f"\n-- coordinate-descent pass {it+1} --", flush=True)
        for pname, vals in GRID.items():
            trials = []
            for v in vals:
                p = dict(cur); p[pname] = v
                trials.append((v, proxy_f1(C, cols_all, p, cur_noise)))
            bv, bf = max(trials, key=lambda t: t[1])
            marker = " <= move" if bf > cur_f1 + 1e-9 else ""
            print(f"   {pname:<16} {[(v, round(f,4)) for v,f in trials]}"
                  f"  best={bv}({bf:.4f}){marker}", flush=True)
            if bf > cur_f1 + 1e-9:
                cur[pname] = bv; cur_f1 = bf
        # noise dimension
        trials = [(nz, proxy_f1(C, cols_all, cur, nz)) for nz in NOISE_GRID]
        bv, bf = max(trials, key=lambda t: t[1])
        print(f"   {'noise_sd':<16} {[(v, round(f,4)) for v,f in trials]}"
              f"  best={bv}({bf:.4f})", flush=True)
        if bf > cur_f1 + 1e-9:
            cur_noise = bv; cur_f1 = bf
    print(f"\n   hill-climb best (all 3 legs): {cur_f1:.4f}  params={cur} noise={cur_noise}", flush=True)

    # ---- leg-subset search (the real shift lever; free from cache) ----
    print(f"\n-- leg-subset search (which base learners to feed the judge) --", flush=True)
    names = np.array(MODEL_ORDER)
    subset_best = (cols_all, cur, cur_noise, cur_f1)
    for r in (2, 3):
        for cols in itertools.combinations(range(K), r):
            cols = list(cols)
            # quick hill-climb of just max_depth + noise on this subset
            best = (-1, None, None)
            for md in [2, 3, 4]:
                for nz in [0.0, 0.10, 0.20]:
                    p = dict(cur); p["max_depth"] = md
                    f = proxy_f1(C, cols, p, nz)
                    if f > best[0]:
                        best = (f, md, nz)
            f, md, nz = best
            tag = "+".join(names[cols])
            star = ""
            if f > subset_best[3] + 1e-9:
                p = dict(cur); p["max_depth"] = md
                subset_best = (cols, p, nz, f); star = " <= NEW BEST"
            print(f"   legs=[{tag:<28}] best-F1={f:.4f} (max_depth={md}, noise={nz}){star}", flush=True)

    cols, params, noise, f1 = subset_best
    print("\n" + "=" * 74, flush=True)
    print(f"[B] OVERALL BEST cluster-F1 = {f1:.4f}", flush=True)
    print(f"    legs   = {[MODEL_ORDER[c] for c in cols]}", flush=True)
    print(f"    params = {params}", flush=True)
    print(f"    noise  = {noise}", flush=True)
    print(f"    vs start {base_f1:.4f}  (delta {f1-base_f1:+.4f})", flush=True)
    print(f"    REAL-LB projection (proxy - 0.078 stack/tree deflation) ~= {f1-0.078:.3f}", flush=True)
    print(f"    (LinearSVC baseline real-LB = 0.72990)", flush=True)
    print("=" * 74, flush=True)

    # ---- write the best candidate's test prediction ----
    rng = np.random.RandomState(SEED)
    oof = C["oof_score"][:, cols]; Y = C["Y"]
    sc = StandardScaler().fit(oof)
    Z = sc.transform(oof) + rng.normal(0.0, noise, (len(oof), len(cols)))
    final = XGBClassifier(**params, objective="binary:logistic",
                          eval_metric="logloss", tree_method="hist",
                          n_jobs=-1, random_state=SEED).fit(Z, Y)
    tst = C["test_score"][:, cols]
    pred = final.predict(sc.transform(tst)).astype(int)
    out = OUT_DIR / "Task3_XGBStack_Best_Prediction.csv"
    pd.DataFrame({"id": C["test_ids"], "label": pred}).to_csv(out, index=False)
    print(f"  wrote {out}  machine={int(pred.sum())} ({pred.mean():.1%})  "
          f"(NOT submitted)", flush=True)


def main():
    if not CACHE.exists():
        build_cache()
    else:
        print(f"[A] using cached base OOF -> {CACHE}", flush=True)
    search()


if __name__ == "__main__":
    main()
