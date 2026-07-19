"""
Task 3 — TREE / BOOSTING model track (XGBoost, LightGBM, RandomForest)
=======================================================================
Goal: squeeze the maximum out of tree models over several dense
representations, with SHIFT-AWARE selection (see Task3_Improved_Model.py).

Motivation: dense-feature models were the only family that scored ABOVE
their validation estimates on the real Kaggle LB (PCA+KNN), while sparse
text models collapsed ~0.09 under the train->test topic shift. Trees on
dense features may therefore transfer better.

REPRESENTATIONS
---------------
  svd100/svd300/svd500 : TruncatedSVD of word(1-2)+char_wb(3-5) TF-IDF
                         built from RAW TEXT, fit transductively on
                         train+test (legitimate: unsupervised, and it makes
                         CV mirror deployment where test text is available).
                         One k=500 fit; smaller k are prefix slices (SVD
                         components are nested top singular vectors).
  feat5000             : the provided 5,000 precomputed TF-IDF features.
  fsvd300              : TruncatedSVD-300 of the provided 5,000 features
                         (fit on stacked train+test).
  stylo10              : the 10 shift-robust stylometric features
                         (ROBUST_STYLO_IDX subset of build_stylo).
  svd300+stylo         : concat of svd300 and stylo10.
  fsvd300+stylo        : concat of fsvd300 and stylo10.

PROTOCOL
--------
  Stage 1  coarse LightGBM grid on EVERY representation, vanilla
           stratified 5-fold (fast lens, in-distribution only).
  Stage 2  refine LightGBM around the best; XGBoost grid on the top reps;
           RandomForest baseline on the top rep.
  Stage 3  confirm the best config of every (model, representation) on the
           CLUSTER-HOLDOUT harness (topic KMeans, hold out whole clusters)
           — the validated leaderboard proxy (it scored the 0.7299-LB
           baseline at 0.7383). SELECTION IS BY CLUSTER-HOLDOUT MEAN.
  Final    refit winner on all 20,000 rows, predict the 6,999 test rows,
           write predictions/Task3_TreeModels_Prediction.csv (id,label).

Class balance 62.5/37.5 handled via class_weight='balanced' (LGBM/RF) and
scale_pos_weight = n_neg/n_pos (XGB).

Run:  .venv/bin/python Task3_TreeModels.py          (full, ~30-45 min)
      FAST=1 .venv/bin/python Task3_TreeModels.py   (pipeline smoke test)
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from Task3_Improved_Model import (
    macro_f1, cluster_folds, stratified_folds, build_stylo,
    ROBUST_STYLO_IDX, vec_word_char, MultiVec,
)

warnings.filterwarnings("ignore")

SEED = 42
FAST = os.environ.get("FAST", "0") == "1"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "predictions"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "Task3_TreeModels_Prediction.csv"

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_lgbm(p):
    return lgb.LGBMClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
        **p)


def make_xgb(p, spw):
    return xgb.XGBClassifier(
        tree_method="hist", eval_metric="logloss", scale_pos_weight=spw,
        random_state=SEED, n_jobs=-1, **p)


def make_rf(p):
    return RandomForestClassifier(
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1, **p)


def cv_eval(factory, X, y, folds):
    """Return (val_mean, val_std, train_mean) macro-F1 over folds."""
    vals, trs = [], []
    for tr, va in folds:
        m = factory()
        m.fit(X[tr], y[tr])
        vals.append(macro_f1(y[va], m.predict(X[va])))
        trs.append(macro_f1(y[tr], m.predict(X[tr])))
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(trs))


def fmt(p):
    return " ".join(f"{k}={v}" for k, v in sorted(p.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"TASK 3 — TREE MODELS (FAST={FAST})  seed={SEED}")
    print("=" * 78)

    # ---- Load & align -----------------------------------------------------
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    spw = n_neg / n_pos  # XGBoost scale_pos_weight
    print(f"train={len(y)}  test={len(test_ids)}  pos={n_pos} neg={n_neg}  "
          f"scale_pos_weight={spw:.4f}")

    # Provided 5,000 TF-IDF features, aligned to train.csv/test.csv by id.
    trf = pd.read_csv(DATA_DIR / "train_features.csv")
    tef = pd.read_csv(DATA_DIR / "test_features.csv")
    feat_cols = [c for c in trf.columns if c not in ("id", "label")]
    trf = trf.set_index("id").loc[train["id"]]  # enforce identical row order
    tef = tef.set_index("id").loc[test["id"]]
    assert (trf["label"].to_numpy(dtype=int) == y).all(), "id-merge mismatch"
    F_tr = trf[feat_cols].to_numpy(dtype=np.float32)
    F_te = tef[feat_cols].to_numpy(dtype=np.float32)
    print(f"provided features: train{F_tr.shape} test{F_te.shape}")

    # ---- Representations --------------------------------------------------
    print("\n[1] building representations ...")
    reps = {}  # name -> (Xtr, Xte)

    # stylometrics (robust subset)
    t = time.time()
    S_tr = build_stylo(texts)[:, ROBUST_STYLO_IDX].astype(np.float32)
    S_te = build_stylo(test_texts)[:, ROBUST_STYLO_IDX].astype(np.float32)
    reps["stylo10"] = (S_tr, S_te)
    print(f"  stylo10 {S_tr.shape[1]}d ({time.time()-t:.0f}s)")

    # word+char TF-IDF -> SVD (transductive fit on train+test)
    t = time.time()
    mv = MultiVec(vec_word_char)
    X_all = mv.fit_transform(np.concatenate([texts, test_texts]))
    print(f"  tfidf word+char: {X_all.shape} ({time.time()-t:.0f}s)")
    t = time.time()
    k_max = 100 if FAST else 500
    svd = TruncatedSVD(n_components=k_max, random_state=SEED)
    Z_all = svd.fit_transform(X_all).astype(np.float32)
    print(f"  SVD-{k_max}: explained var {svd.explained_variance_ratio_.sum():.3f} "
          f"({time.time()-t:.0f}s)")
    Z_tr, Z_te = Z_all[:len(y)], Z_all[len(y):]
    ks = [100] if FAST else [100, 300, 500]
    for k in ks:
        reps[f"svd{k}"] = (Z_tr[:, :k], Z_te[:, :k])
    del X_all, Z_all

    # provided 5000 features: direct + SVD-300
    if not FAST:
        reps["feat5000"] = (F_tr, F_te)
    t = time.time()
    kf = 100 if FAST else 300
    fsvd = TruncatedSVD(n_components=kf, random_state=SEED)
    FZ_all = fsvd.fit_transform(np.vstack([F_tr, F_te])).astype(np.float32)
    FZ_tr, FZ_te = FZ_all[:len(y)], FZ_all[len(y):]
    reps[f"fsvd{kf}"] = (FZ_tr, FZ_te)
    print(f"  fsvd{kf}: explained var {fsvd.explained_variance_ratio_.sum():.3f} "
          f"({time.time()-t:.0f}s)")

    # combos
    kc = ks[-1] if FAST else 300
    reps[f"svd{kc}+stylo"] = (np.hstack([Z_tr[:, :kc], S_tr]),
                              np.hstack([Z_te[:, :kc], S_te]))
    reps[f"fsvd{kf}+stylo"] = (np.hstack([FZ_tr, S_tr]),
                               np.hstack([FZ_te, S_te]))

    print("  reps:", {n: v[0].shape[1] for n, v in reps.items()})

    # ---- Harnesses (shared folds) -----------------------------------------
    print("\n[2] building shared folds ...")
    strat = stratified_folds(y)
    clus, cl_labels = cluster_folds(texts, y)
    if FAST:
        strat, clus = strat[:3], clus[:3]
    print(f"  stratified={len(strat)} folds  cluster-holdout={len(clus)} folds "
          f"(cluster sizes {np.bincount(cl_labels).tolist()})")

    # ---- Stage 1: coarse LightGBM on every representation ------------------
    print("\n[3] STAGE 1 — coarse LightGBM grid, vanilla stratified CV")
    coarse = [
        dict(n_estimators=300, learning_rate=0.1, num_leaves=31,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(n_estimators=300, learning_rate=0.1, num_leaves=63,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    ]
    if FAST:
        coarse = coarse[:2]

    stage1 = {}  # rep -> (best_params, val, std, train)
    for rep, (Xtr, _) in reps.items():
        grid = coarse[:2] if rep == "feat5000" else coarse  # cap cost on 5000d
        best = None
        for p in grid:
            t = time.time()
            v, s, tr = cv_eval(lambda: make_lgbm(p), Xtr, y, strat)
            print(f"  LGBM {rep:<14} {fmt(p):<95} "
                  f"val={v:.4f}+/-{s:.3f} train={tr:.4f} ({time.time()-t:.0f}s)")
            if best is None or v > best[1]:
                best = (p, v, s, tr)
        stage1[rep] = best
        print(f"  >> best {rep}: val={best[1]:.4f}")

    rank1 = sorted(stage1.items(), key=lambda kv: -kv[1][1])
    top_reps = [r for r, _ in rank1[:2]]
    print(f"\n  top representations by vanilla CV: {top_reps}")

    # ---- Stage 2: refine LGBM on top reps; XGB on top reps; RF baseline ----
    print("\n[4] STAGE 2 — refinement")
    stage2 = []  # (model, rep, params, val, std, train)

    for rep in top_reps:
        Xtr = reps[rep][0]
        bp = dict(stage1[rep][0])
        refine = []
        for lam in [0.1, 5.0]:
            q = dict(bp); q["reg_lambda"] = lam; refine.append(q)
        for cs in [0.6, 1.0]:
            q = dict(bp); q["colsample_bytree"] = cs; refine.append(q)
        for mcs in [5, 50]:
            q = dict(bp); q["min_child_samples"] = mcs; refine.append(q)
        q = dict(bp); q["n_estimators"] = 1200; q["learning_rate"] = 0.03
        refine.append(q)
        if FAST:
            refine = refine[:2]
        best = stage1[rep]
        for p in refine:
            t = time.time()
            v, s, tr = cv_eval(lambda: make_lgbm(p), Xtr, y, strat)
            print(f"  LGBM {rep:<14} {fmt(p):<95} "
                  f"val={v:.4f}+/-{s:.3f} train={tr:.4f} ({time.time()-t:.0f}s)")
            if v > best[1]:
                best = (p, v, s, tr)
        stage2.append(("LGBM", rep, *best))

        xgrid = [
            dict(n_estimators=400, learning_rate=0.1, max_depth=4,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
            dict(n_estimators=400, learning_rate=0.1, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
            dict(n_estimators=800, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
            dict(n_estimators=800, learning_rate=0.05, max_depth=8,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        ]
        if FAST:
            xgrid = xgrid[:1]
        xbest = None
        for p in xgrid:
            t = time.time()
            v, s, tr = cv_eval(lambda: make_xgb(p, spw), Xtr, y, strat)
            print(f"  XGB  {rep:<14} {fmt(p):<95} "
                  f"val={v:.4f}+/-{s:.3f} train={tr:.4f} ({time.time()-t:.0f}s)")
            if xbest is None or v > xbest[1]:
                xbest = (p, v, s, tr)
        stage2.append(("XGB", rep, *xbest))

    # RandomForest comparison baseline on the single best rep
    rf_rep = top_reps[0]
    rf_grid = [
        dict(n_estimators=600, max_features="sqrt", min_samples_leaf=1),
        dict(n_estimators=600, max_features=0.2, min_samples_leaf=3),
    ]
    if FAST:
        rf_grid = rf_grid[:1]
    rbest = None
    for p in rf_grid:
        t = time.time()
        v, s, tr = cv_eval(lambda: make_rf(p), reps[rf_rep][0], y, strat)
        print(f"  RF   {rf_rep:<14} {fmt(p):<95} "
              f"val={v:.4f}+/-{s:.3f} train={tr:.4f} ({time.time()-t:.0f}s)")
        if rbest is None or v > rbest[1]:
            rbest = (p, v, s, tr)
    stage2.append(("RF", rf_rep, *rbest))

    # non-top reps also enter the final table with their stage-1 LGBM best
    for rep, best in stage1.items():
        if rep not in top_reps:
            stage2.append(("LGBM", rep, *best))

    # ---- Stage 3: cluster-holdout confirmation (SELECTION LENS) ------------
    print("\n[5] STAGE 3 — cluster-holdout (topic-shift proxy) for every candidate")
    table = []
    for model, rep, p, v, s, tr in stage2:
        Xtr = reps[rep][0]
        if model == "LGBM":
            fac = lambda p=p: make_lgbm(p)
        elif model == "XGB":
            fac = lambda p=p: make_xgb(p, spw)
        else:
            fac = lambda p=p: make_rf(p)
        t = time.time()
        cv_, cs_, ct_ = cv_eval(fac, Xtr, y, clus)
        print(f"  {model:<5}{rep:<14} strat={v:.4f}  clus={cv_:.4f}+/-{cs_:.3f} "
              f"clus_train={ct_:.4f} gap={ct_-cv_:+.3f} ({time.time()-t:.0f}s)")
        table.append(dict(model=model, rep=rep, params=p,
                          strat=v, strat_std=s, strat_train=tr,
                          clus=cv_, clus_std=cs_, clus_train=ct_))

    # ---- Decision -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("[6] FINAL TABLE (sorted by CLUSTER-HOLDOUT mean — the selection lens)")
    print("=" * 78)
    table.sort(key=lambda r: -r["clus"])
    print(f"  {'model':<6}{'rep':<15}{'STRAT val':>11}{'CLUS val':>11}"
          f"{'CLUS train':>12}{'gap':>8}")
    for r in table:
        print(f"  {r['model']:<6}{r['rep']:<15}"
              f"{r['strat']:>11.4f}{r['clus']:>11.4f}"
              f"{r['clus_train']:>12.4f}{r['clus_train']-r['clus']:>+8.3f}")

    win = table[0]
    print(f"\n  WINNER: {win['model']} on {win['rep']}")
    print(f"  params: {fmt(win['params'])}")
    print(f"  vanilla-CV {win['strat']:.4f}+/-{win['strat_std']:.3f} "
          f"(train {win['strat_train']:.4f}, gap {win['strat_train']-win['strat']:+.3f})")
    print(f"  cluster   {win['clus']:.4f}+/-{win['clus_std']:.3f} "
          f"(train {win['clus_train']:.4f}, gap {win['clus_train']-win['clus']:+.3f})")

    # ---- Refit on all 20k and predict test ---------------------------------
    print("\n[7] refit winner on full train, predict test")
    Xtr, Xte = reps[win["rep"]]
    if win["model"] == "LGBM":
        final = make_lgbm(win["params"])
    elif win["model"] == "XGB":
        final = make_xgb(win["params"], spw)
    else:
        final = make_rf(win["params"])
    final.fit(Xtr, y)
    pred = final.predict(Xte).astype(int)
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV}")
    print(f"  rows={len(pred)}  machine={int(pred.sum())} "
          f"({pred.mean():.1%})  human={int((pred==0).sum())}")
    assert len(pred) == 6999 or FAST

    print(f"\n  total runtime: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
