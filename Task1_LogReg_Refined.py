"""
Task 1 — Logistic Regression from scratch: REFINEMENT EXPERIMENTS.

Goal: try to genuinely beat the notebook baseline of val macro F1 = 0.7318
(bs=512, epochs=500, lr=10.0, threshold=0.5, stratified 90/10 split, SEED=42)
while staying 100% compliant: NO sklearn LogisticRegression / SGDClassifier /
any pre-built classifier for the actual model. Everything for the model
(sigmoid, loss, gradients, L2, class weighting, Adam, standardization, macro F1)
is hand-rolled with NumPy. sklearn is used ONLY for train_test_split (a data
splitting utility, not a logistic-regression package).

Run: .venv/bin/python Task1_LogReg_Refined.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # data splitting only

SEED = 42
DATA_DIR = Path("data")
BASELINE_F1 = 0.7318
IMPROVEMENT_MARGIN = 0.01  # min non-trivial margin over baseline to overwrite

# --------------------------------------------------------------------------
# Hand-rolled metric
# --------------------------------------------------------------------------
def macro_f1(y_true, y_pred):
    """Macro-averaged F1 computed by hand from per-class TP/FP/FN."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    f1s = []
    for c in (0, 1):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# --------------------------------------------------------------------------
# Model primitives (all NumPy, no ML library)
# --------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def loss(y, y_hat, w=None, l2=0.0, sw=None):
    """(Optionally class-weighted, L2-regularized) log loss."""
    eps = 1e-9
    y_hat = np.clip(y_hat, eps, 1 - eps)
    per = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    if sw is not None:
        val = float(np.sum(sw * per) / np.sum(sw))
    else:
        val = float(np.mean(per))
    if w is not None and l2 > 0.0:
        val += float(l2 * np.sum(w * w) / (2 * len(y)))
    return val


def gradients(X, y, y_hat, w=None, l2=0.0, sw=None):
    """dw, db for (weighted) log loss with optional L2 on weights (not bias)."""
    m = X.shape[0]
    err = y_hat - y
    if sw is not None:
        err = err * sw
        denom = np.sum(sw)
    else:
        denom = m
    dw = X.T @ err / denom
    db = float(np.sum(err) / denom)
    if w is not None and l2 > 0.0:
        dw = dw + (l2 / m) * w
    return dw, db


def standardize_fit(X):
    """Mean/std from TRAIN ONLY. std floored to avoid divide-by-zero."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(X.dtype), sd.astype(X.dtype)


def apply_std(X, mu, sd):
    return (X - mu) / sd


def train(X, y, bs, epochs, lr, l2=0.0, class_weight=False,
          optimizer="sgd", lr_decay=0.0, seed=SEED, verbose=False):
    """Mini-batch gradient descent with optional L2, class weighting, Adam,
    and exponential lr decay. Returns (w, b, losses)."""
    rng = np.random.default_rng(seed)
    m, n = X.shape
    y = y.reshape(m, 1).astype(X.dtype)
    w = np.zeros((n, 1), dtype=X.dtype)
    b = 0.0

    # class weights: inverse frequency, normalized so mean weight = 1
    sample_w = None
    if class_weight:
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        w_pos = m / (2.0 * n_pos)
        w_neg = m / (2.0 * n_neg)
        sample_w = np.where(y == 1, w_pos, w_neg).astype(X.dtype)

    # Adam state
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = 0.0; vb = 0.0
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    t = 0

    losses = []
    for epoch in range(epochs):
        cur_lr = lr / (1.0 + lr_decay * epoch) if lr_decay > 0 else lr
        perm = rng.permutation(m)
        X_sh, y_sh = X[perm], y[perm]
        sw_sh = sample_w[perm] if sample_w is not None else None
        for i in range(0, m, bs):
            xb = X_sh[i:i + bs]
            yb = y_sh[i:i + bs]
            swb = sw_sh[i:i + bs] if sw_sh is not None else None
            y_hat = sigmoid(xb @ w + b)
            dw, db = gradients(xb, yb, y_hat, w=w, l2=l2, sw=swb)
            if optimizer == "adam":
                t += 1
                mw = beta1 * mw + (1 - beta1) * dw
                vw = beta2 * vw + (1 - beta2) * (dw * dw)
                mb = beta1 * mb + (1 - beta1) * db
                vb = beta2 * vb + (1 - beta2) * (db * db)
                mw_h = mw / (1 - beta1 ** t); vw_h = vw / (1 - beta2 ** t)
                mb_h = mb / (1 - beta1 ** t); vb_h = vb / (1 - beta2 ** t)
                w -= cur_lr * mw_h / (np.sqrt(vw_h) + adam_eps)
                b -= cur_lr * mb_h / (np.sqrt(vb_h) + adam_eps)
            else:
                w -= cur_lr * dw
                b -= cur_lr * db
        full = sigmoid(X @ w + b)
        losses.append(loss(y, full, w=w, l2=l2, sw=sample_w))
        if verbose and (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={losses[-1]:.4f}")
    return w, b, losses


def predict_label(X, w, b, thr=0.5):
    return (sigmoid(X @ w + b).ravel() >= thr).astype(int)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load():
    tr = pd.read_csv(DATA_DIR / "train_features.csv")
    te = pd.read_csv(DATA_DIR / "test_features.csv")
    cols = [c for c in tr.columns if c not in ("id", "label")]
    X = tr[cols].to_numpy(dtype=np.float32)
    y = tr["label"].to_numpy(dtype=int)
    Xtest = te[cols].to_numpy(dtype=np.float32)
    test_ids = te["id"].to_numpy()
    return X, y, Xtest, test_ids


def make_split(X, y, seed=SEED):
    tr_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# --------------------------------------------------------------------------
# One end-to-end run on a given split, returns val macro F1 (threshold 0.5)
# --------------------------------------------------------------------------
def run_config(cfg, X_tr, y_tr, X_val, y_val, seed=SEED, verbose=False):
    if cfg.get("standardize"):
        mu, sd = standardize_fit(X_tr)
        Xt = apply_std(X_tr, mu, sd)
        Xv = apply_std(X_val, mu, sd)
    else:
        Xt, Xv = X_tr, X_val
    w, b, losses = train(
        Xt, y_tr, bs=cfg["bs"], epochs=cfg["epochs"], lr=cfg["lr"],
        l2=cfg.get("l2", 0.0), class_weight=cfg.get("class_weight", False),
        optimizer=cfg.get("optimizer", "sgd"), lr_decay=cfg.get("lr_decay", 0.0),
        seed=seed, verbose=verbose)
    val_pred = predict_label(Xv, w, b, thr=0.5)
    return macro_f1(y_val, val_pred), losses[-1], (w, b)


def threshold_tuned_f1(cfg, X_tr, y_tr, X_val, y_val, seed=SEED):
    """HONEST threshold tuning. Inner-split X_tr (85/15), pick the macro-F1
    optimal threshold on the inner holdout ONLY, refit on full X_tr, then
    evaluate on X_val at that fixed threshold. Threshold never sees X_val.
    Returns (val_f1_tuned, chosen_threshold)."""
    itr, ihold = train_test_split(np.arange(len(y_tr)), test_size=0.15,
                                  stratify=y_tr, random_state=seed)
    w, b, _ = train(X_tr[itr], y_tr[itr], bs=cfg["bs"], epochs=cfg["epochs"],
                    lr=cfg["lr"], l2=cfg.get("l2", 0.0),
                    class_weight=cfg.get("class_weight", False), seed=seed)
    p_hold = sigmoid(X_tr[ihold] @ w + b).ravel()
    best_thr, best_hf = 0.5, -1.0
    for thr in np.linspace(0.20, 0.80, 61):
        hf = macro_f1(y_tr[ihold], (p_hold >= thr).astype(int))
        if hf > best_hf:
            best_hf, best_thr = hf, float(thr)
    # refit on full X_tr, apply fixed threshold to val
    w, b, _ = train(X_tr, y_tr, bs=cfg["bs"], epochs=cfg["epochs"], lr=cfg["lr"],
                    l2=cfg.get("l2", 0.0),
                    class_weight=cfg.get("class_weight", False), seed=seed)
    p_val = sigmoid(X_val @ w + b).ravel()
    return macro_f1(y_val, (p_val >= best_thr).astype(int)), best_thr


def main():
    t_start = time.time()
    print("Loading data...")
    X, y, Xtest, test_ids = load()
    print(f"X: {X.shape} | class balance {np.bincount(y)} "
          f"({100*np.mean(y):.1f}% machine)")
    X_tr, y_tr, X_val, y_val = make_split(X, y, seed=SEED)
    print(f"train {len(y_tr)} | val {len(y_val)}  (SEED={SEED})\n")

    print(f"BASELINE (notebook): val macro F1 = {BASELINE_F1:.4f} "
          f"[bs=512, epochs=500, lr=10.0, no reg, thr=0.5]\n")

    # NOTE: earlier run showed standardize+adam (lr re-tuned to 0.01) converges
    # to the same loss ~0.30 but generalizes WORSE (val F1 ~0.70-0.71) on sparse
    # TF-IDF, and coarse L2 (>=0.05) collapses the model. Effective end-of-train
    # weight shrinkage ~= exp(-lr*l2*epochs/bs) = exp(-9.77*l2), so useful L2 must
    # be small. Grid below centers on that regime.
    experiments = [
        ("baseline-reproduce",
         dict(bs=512, epochs=500, lr=10.0)),
        ("L2=0.0005",
         dict(bs=512, epochs=500, lr=10.0, l2=0.0005)),
        ("L2=0.001",
         dict(bs=512, epochs=500, lr=10.0, l2=0.001)),
        ("L2=0.002",
         dict(bs=512, epochs=500, lr=10.0, l2=0.002)),
        ("L2=0.005",
         dict(bs=512, epochs=500, lr=10.0, l2=0.005)),
        ("L2=0.01",
         dict(bs=512, epochs=500, lr=10.0, l2=0.01)),
        ("L2=0.02",
         dict(bs=512, epochs=500, lr=10.0, l2=0.02)),
        ("class_weight",
         dict(bs=512, epochs=500, lr=10.0, class_weight=True)),
        ("class_weight+L2=0.005",
         dict(bs=512, epochs=500, lr=10.0, l2=0.005, class_weight=True)),
        ("epochs=800",
         dict(bs=512, epochs=800, lr=10.0)),
        ("lr_decay",
         dict(bs=512, epochs=500, lr=10.0, lr_decay=0.001)),
    ]

    results = []
    for name, cfg in experiments:
        t0 = time.time()
        f1, final_loss, _ = run_config(cfg, X_tr, y_tr, X_val, y_val, seed=SEED)
        dt = time.time() - t0
        delta = f1 - BASELINE_F1
        flag = "  <-- beats baseline" if delta >= IMPROVEMENT_MARGIN else ""
        results.append((name, cfg, f1, delta))
        print(f"{name:28s} val F1={f1:.4f}  (Δ{delta:+.4f})  "
              f"loss={final_loss:.4f}  {dt:.1f}s{flag}")

    # pick best single-split config
    results.sort(key=lambda r: r[2], reverse=True)
    best_name, best_cfg, best_f1, best_delta = results[0]
    print(f"\nBest single-split config: {best_name}  val F1={best_f1:.4f}  "
          f"(Δ{best_delta:+.4f})")

    # ---- Multi-seed robustness check vs baseline ----
    # Two candidate levers evaluated per seed (each with its OWN split & its OWN
    # per-seed threshold, so nothing leaks): (a) best grid config at thr=0.5,
    # (b) baseline model with an honestly-tuned threshold.
    print("\nMulti-seed check (5 splits) — is any gain consistent, not luck?")
    seeds = [42, 7, 123, 2024, 99]
    base_cfg = dict(bs=512, epochs=500, lr=10.0)
    base_scores, best_scores, thr_scores = [], [], []
    for s in seeds:
        Xtr_s, ytr_s, Xval_s, yval_s = make_split(X, y, seed=s)
        bf, _, _ = run_config(base_cfg, Xtr_s, ytr_s, Xval_s, yval_s, seed=s)
        cf, _, _ = run_config(best_cfg, Xtr_s, ytr_s, Xval_s, yval_s, seed=s)
        tf, thr = threshold_tuned_f1(base_cfg, Xtr_s, ytr_s, Xval_s, yval_s, seed=s)
        base_scores.append(bf); best_scores.append(cf); thr_scores.append(tf)
        print(f"  seed {s:4d}: baseline={bf:.4f}  best-grid={cf:.4f} (Δ{cf-bf:+.4f})"
              f"  thr-tuned={tf:.4f} (Δ{tf-bf:+.4f}, thr={thr:.2f})")
    base_mean = float(np.mean(base_scores))
    best_mean = float(np.mean(best_scores))
    thr_mean = float(np.mean(thr_scores))
    print(f"\n  baseline  mean={base_mean:.4f} (spread {np.ptp(base_scores):.4f})")
    print(f"  best-grid mean={best_mean:.4f} (Δ{best_mean-base_mean:+.4f})")
    print(f"  thr-tuned mean={thr_mean:.4f} (Δ{thr_mean-base_mean:+.4f})")

    # choose the better of the two candidate levers
    grid_ok = (all(c - b > 0 for b, c in zip(base_scores, best_scores))
               and best_mean - base_mean >= IMPROVEMENT_MARGIN
               and best_f1 >= BASELINE_F1 + IMPROVEMENT_MARGIN)
    thr_ok = (all(t - b > 0 for b, t in zip(base_scores, thr_scores))
              and thr_mean - base_mean >= IMPROVEMENT_MARGIN)
    if thr_ok and thr_mean >= best_mean:
        mean_delta = thr_mean - base_mean
        consistent = True
        winner = "threshold-tuned baseline"
        winner_kind = "threshold"
    elif grid_ok:
        mean_delta = best_mean - base_mean
        consistent = True
        winner = f"grid config '{best_name}'"
        winner_kind = "grid"
    else:
        mean_delta = max(best_mean, thr_mean) - base_mean
        consistent = False
        winner = "none"
        winner_kind = "none"
    print(f"\n  Winning lever: {winner} | mean Δ over 5 seeds = {mean_delta:+.4f}")

    # ---- Decision ----
    genuine = consistent and mean_delta >= IMPROVEMENT_MARGIN and winner_kind != "none"

    print("\n" + "=" * 70)
    if genuine and winner_kind == "grid":
        print(f"GENUINE IMPROVEMENT ({winner}). Refitting on ALL 20000 rows "
              f"and overwriting LogReg_Prediction.csv.")
        w, b, _ = train(
            X, y, bs=best_cfg["bs"], epochs=best_cfg["epochs"],
            lr=best_cfg["lr"], l2=best_cfg.get("l2", 0.0),
            class_weight=best_cfg.get("class_weight", False),
            optimizer=best_cfg.get("optimizer", "sgd"),
            lr_decay=best_cfg.get("lr_decay", 0.0), seed=SEED)
        test_pred = predict_label(Xtest, w, b, thr=0.5)
        out = pd.DataFrame({"id": test_ids, "label": test_pred.astype(int)})
        out.to_csv("LogReg_Prediction.csv", index=False)
        print(f"WROTE LogReg_Prediction.csv ({len(out)} rows, "
              f"positives={int(out['label'].sum())})")
    elif genuine and winner_kind == "threshold":
        print(f"GENUINE IMPROVEMENT ({winner}). Deriving threshold on an inner "
              f"split of full train, refitting on ALL 20000 rows.")
        itr, ihold = train_test_split(np.arange(len(y)), test_size=0.15,
                                      stratify=y, random_state=SEED)
        w, b, _ = train(X[itr], y[itr], bs=base_cfg["bs"], epochs=base_cfg["epochs"],
                        lr=base_cfg["lr"], seed=SEED)
        p_hold = sigmoid(X[ihold] @ w + b).ravel()
        best_thr, best_hf = 0.5, -1.0
        for thr in np.linspace(0.20, 0.80, 61):
            hf = macro_f1(y[ihold], (p_hold >= thr).astype(int))
            if hf > best_hf:
                best_hf, best_thr = hf, float(thr)
        w, b, _ = train(X, y, bs=base_cfg["bs"], epochs=base_cfg["epochs"],
                        lr=base_cfg["lr"], seed=SEED)
        test_pred = (sigmoid(Xtest @ w + b).ravel() >= best_thr).astype(int)
        out = pd.DataFrame({"id": test_ids, "label": test_pred.astype(int)})
        out.to_csv("LogReg_Prediction.csv", index=False)
        print(f"WROTE LogReg_Prediction.csv (thr={best_thr:.2f}, {len(out)} rows, "
              f"positives={int(out['label'].sum())})")
    else:
        print("NO genuine improvement over 0.7318 by a non-trivial margin.")
        print("LogReg_Prediction.csv LEFT UNCHANGED.")
    print("=" * 70)
    print(f"\nTotal runtime {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
