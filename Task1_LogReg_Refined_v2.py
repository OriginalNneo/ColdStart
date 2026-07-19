"""
Task 1 — Logistic Regression from scratch: REFINEMENT ROUND 2.

Round 1 (Task1_LogReg_Refined.py) already ruled out: L2 grid, class weighting,
Adam + column standardization, lr decay, longer training, threshold tuning —
all inside the ~0.022 seed-to-seed noise band. This round tests the levers
round 1 did NOT touch:

  A. Row L2-normalization: the provided 5000 features are a truncated TF-IDF
     matrix whose row norms range 0.38-1.00 (normalized BEFORE truncation).
     Re-normalizing each row to unit norm removes that residual scale
     variation — the standard preprocessing for TF-IDF + linear models.
  B. log1p transform (sublinear scaling) followed by row normalization.
  C. Multi-seed probability averaging (train 3 models with different shuffle
     seeds, average their sigmoid outputs) — pure variance reduction, still
     100% hand-rolled logistic regression.

Same compliance rules: model math is all NumPy; sklearn only for
train_test_split. Same honesty rules: any candidate must beat the baseline
on ALL 5 seeds AND by mean >= 0.01 before LogReg_Prediction.csv is touched.

Run: .venv/bin/python Task1_LogReg_Refined_v2.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # data splitting only

SEED = 42
DATA_DIR = Path("data")
BASELINE_F1 = 0.7318
IMPROVEMENT_MARGIN = 0.01
BASE_CFG = dict(bs=512, epochs=500, lr=10.0)


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    f1s = []
    for c in (0, 1):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train(X, y, bs, epochs, lr, seed=SEED):
    """Plain mini-batch GD logistic regression (round-1 winner config)."""
    rng = np.random.default_rng(seed)
    m, n = X.shape
    y = y.reshape(m, 1).astype(X.dtype)
    w = np.zeros((n, 1), dtype=X.dtype)
    b = 0.0
    for _ in range(epochs):
        perm = rng.permutation(m)
        X_sh, y_sh = X[perm], y[perm]
        for i in range(0, m, bs):
            xb = X_sh[i:i + bs]
            yb = y_sh[i:i + bs]
            err = sigmoid(xb @ w + b) - yb
            w -= lr * (xb.T @ err) / xb.shape[0]
            b -= lr * float(np.sum(err)) / xb.shape[0]
    return w, b


def predict_proba(X, w, b):
    return sigmoid(X @ w + b).ravel()


# ---- feature transforms (fit-free, per-row, so no train/val leakage) ----
def row_l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (X / norms).astype(X.dtype)


def log1p_l2norm(X):
    return row_l2norm(np.log1p(X))


TRANSFORMS = {
    "raw": lambda X: X,
    "l2norm": row_l2norm,
    "log1p+l2norm": log1p_l2norm,
}


def load():
    tr = pd.read_csv(DATA_DIR / "train_features.csv")
    te = pd.read_csv(DATA_DIR / "test_features.csv")
    cols = [c for c in tr.columns if c not in ("id", "label")]
    X = tr[cols].to_numpy(dtype=np.float32)
    y = tr["label"].to_numpy(dtype=int)
    Xtest = te[cols].to_numpy(dtype=np.float32)
    return X, y, Xtest, te["id"].to_numpy()


def make_split(X, y, seed):
    tr_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def eval_single(Xt, yt, Xv, yv, lr, seed):
    w, b = train(Xt, yt, bs=BASE_CFG["bs"], epochs=BASE_CFG["epochs"],
                 lr=lr, seed=seed)
    return macro_f1(yv, (predict_proba(Xv, w, b) >= 0.5).astype(int))


def eval_ensemble(Xt, yt, Xv, yv, lr, seeds):
    """Average sigmoid probabilities over models trained with different
    shuffle seeds (init is zeros, so seeds only change batch order)."""
    p = np.zeros(len(yv))
    for s in seeds:
        w, b = train(Xt, yt, bs=BASE_CFG["bs"], epochs=BASE_CFG["epochs"],
                     lr=lr, seed=s)
        p += predict_proba(Xv, w, b)
    p /= len(seeds)
    return macro_f1(yv, (p >= 0.5).astype(int))


def main():
    t_start = time.time()
    X, y, Xtest, test_ids = load()
    print(f"X: {X.shape} | baseline val F1 = {BASELINE_F1:.4f}\n")

    # ---- Phase 1: screen transforms x lr on the seed-42 split ----
    print("Phase 1 — screen (seed-42 split, thr=0.5):")
    X_tr, y_tr, X_val, y_val = make_split(X, y, SEED)
    screen = []
    for tname, tfn in TRANSFORMS.items():
        Xt, Xv = tfn(X_tr), tfn(X_val)
        lrs = [10.0] if tname == "raw" else [5.0, 10.0, 20.0]
        for lr in lrs:
            t0 = time.time()
            f1 = eval_single(Xt, y_tr, Xv, y_val, lr, SEED)
            screen.append((tname, lr, f1))
            print(f"  {tname:14s} lr={lr:5.1f}  val F1={f1:.4f} "
                  f"(Δ{f1-BASELINE_F1:+.4f})  {time.time()-t0:.0f}s")

    # best non-raw candidate from the screen
    cand = max((s for s in screen if s[0] != "raw"), key=lambda s: s[2])
    cand_name, cand_lr, cand_screen_f1 = cand
    print(f"\nBest new-transform candidate: {cand_name} lr={cand_lr} "
          f"(screen F1={cand_screen_f1:.4f})")

    # ---- Phase 2: 5-seed robustness — baseline vs candidate vs 3-seed
    # ensemble of the candidate ----
    print("\nPhase 2 — 5-seed robustness:")
    seeds = [42, 7, 123, 2024, 99]
    cand_tfn = TRANSFORMS[cand_name]
    base_s, cand_s, ens_s = [], [], []
    for s in seeds:
        Xtr_s, ytr_s, Xval_s, yval_s = make_split(X, y, s)
        bf = eval_single(Xtr_s, ytr_s, Xval_s, yval_s, BASE_CFG["lr"], s)
        Xt, Xv = cand_tfn(Xtr_s), cand_tfn(Xval_s)
        cf = eval_single(Xt, ytr_s, Xv, yval_s, cand_lr, s)
        ef = eval_ensemble(Xt, ytr_s, Xv, yval_s, cand_lr,
                           seeds=[s, s + 1000, s + 2000])
        base_s.append(bf); cand_s.append(cf); ens_s.append(ef)
        print(f"  seed {s:4d}: baseline={bf:.4f}  {cand_name}={cf:.4f} "
              f"(Δ{cf-bf:+.4f})  ensemble={ef:.4f} (Δ{ef-bf:+.4f})")

    bm, cm, em = map(lambda a: float(np.mean(a)), (base_s, cand_s, ens_s))
    print(f"\n  baseline mean={bm:.4f} (spread {np.ptp(base_s):.4f})")
    print(f"  {cand_name:14s} mean={cm:.4f} (Δ{cm-bm:+.4f})")
    print(f"  ensemble       mean={em:.4f} (Δ{em-bm:+.4f})")

    cand_ok = all(c > b for b, c in zip(base_s, cand_s)) and cm - bm >= IMPROVEMENT_MARGIN
    ens_ok = all(e > b for b, e in zip(base_s, ens_s)) and em - bm >= IMPROVEMENT_MARGIN

    print("\n" + "=" * 70)
    if ens_ok and em >= cm:
        winner, use_ens = f"3-seed ensemble of {cand_name} lr={cand_lr}", True
    elif cand_ok:
        winner, use_ens = f"{cand_name} lr={cand_lr}", False
    else:
        winner, use_ens = None, False

    if winner:
        print(f"GENUINE IMPROVEMENT: {winner}. Refitting on all {len(y)} rows "
              f"and overwriting LogReg_Prediction.csv.")
        Xf, Xtf = cand_tfn(X), cand_tfn(Xtest)
        train_seeds = [SEED, SEED + 1000, SEED + 2000] if use_ens else [SEED]
        p = np.zeros(len(test_ids))
        for s in train_seeds:
            w, b = train(Xf, y, bs=BASE_CFG["bs"], epochs=BASE_CFG["epochs"],
                         lr=cand_lr, seed=s)
            p += predict_proba(Xtf, w, b)
        p /= len(train_seeds)
        out = pd.DataFrame({"id": test_ids, "label": (p >= 0.5).astype(int)})
        out.to_csv("LogReg_Prediction.csv", index=False)
        print(f"WROTE LogReg_Prediction.csv ({len(out)} rows, "
              f"positives={int(out['label'].sum())})")
    else:
        print("NO genuine improvement over the baseline by a non-trivial margin.")
        print("LogReg_Prediction.csv LEFT UNCHANGED.")
    print("=" * 70)
    print(f"\nTotal runtime {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
