"""
GenAI Text Detection — Gradient-Boosted Decision Trees, built from scratch.
==========================================================================
No scikit-learn, no XGBoost/LightGBM/CatBoost. Only NumPy for array math and
Pandas for reading CSVs. This implements the *actual* algorithm those
libraries run under the hood:

  1. A regression tree (`DecisionTreeRegressor`) that greedily picks the
     (feature, threshold) split maximising the second-order loss reduction.
  2. A gradient-boosting classifier that starts at the class log-odds prior
     and, each round, fits one tree to the NEGATIVE GRADIENT of the logistic
     loss (residual y - sigmoid(F)), then adds lr * tree to the running
     margin F. A sigmoid of the final margin gives P(machine | x).

Why gradient boosting on THESE features:
- Trees model non-linear interactions the linear SVM/NB baselines can't
  (e.g. "long words AND low comma-density" jointly signalling machine text).
- But trees are known to choke on 100K-dim sparse TF-IDF (prior RF/XGB runs
  scored 0.70-0.75, well below the 0.82 linear model). So we feed them a
  MODERATE, DENSE matrix: hand-crafted stylometric features + a few hundred
  frequent-word counts — the regime where split-finding is actually useful.

Speed notes: split-finding is fully vectorised. Each feature column is
argsort-ed ONCE up front (globally); at every node we gather that global
order through a membership mask and evaluate all candidate thresholds with
cumulative sums — no per-node re-sorting, no Python loop over thresholds.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the already-correct, hand-rolled helpers from the Naive Bayes module.
# Its __main__ guard means importing does NOT run its pipeline.
from ML_Project_NaiveBayes_Scratch import (
    extract_features, build_feature_matrix, FEATURE_NAMES,
    stratified_split, stratified_kfold, macro_f1,
    tokenize, build_vocabulary, build_count_matrix,
    WORD_RE, SENT_SPLIT_RE,
)

SEED = 42
rng = np.random.default_rng(SEED)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("predictions")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. EXTRA STYLOMETRIC FEATURES (extend the imported 14)
# ============================================================================
# Trees split on interactions, so a few extra *variance / structure* signals
# the base 14 don't capture are cheap and worth adding.

EXTRA_FEATURE_NAMES = [
    "sentence_len_std",     # rhythm regularity — AI text is often more uniform
    "unique_bigram_ratio",  # phrase-level diversity (distinct pairs / total pairs)
    "avg_paragraph_len",    # words per newline-delimited block
    "exclaim_question_ratio",  # emphatic punctuation per word (human-leaning)
]


def extract_extra_features(text):
    words = WORD_RE.findall(text)
    n_words = len(words)
    if n_words == 0:
        return np.zeros(len(EXTRA_FEATURE_NAMES), dtype=np.float64)

    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    sent_word_counts = np.array(
        [len(WORD_RE.findall(s)) for s in sentences], dtype=np.float64
    ) if sentences else np.array([n_words], dtype=np.float64)

    lower = [w.lower() for w in words]
    bigrams = [f"{lower[i]}_{lower[i+1]}" for i in range(len(lower) - 1)]
    unique_bigram_ratio = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    paragraphs = [p for p in text.split("\n") if p.strip()]
    n_para = max(1, len(paragraphs))
    exq = text.count("!") + text.count("?")

    return np.array([
        sent_word_counts.std(),
        unique_bigram_ratio,
        n_words / n_para,
        exq / n_words,
    ], dtype=np.float64)


# ============================================================================
# 2. DECISION TREE REGRESSOR — from scratch, second-order (XGBoost-style)
# ============================================================================
#
# We fit each tree to the logistic loss's gradient g = sigmoid(F) - y and
# hessian h = sigmoid(F)*(1-sigmoid(F)). Choosing the leaf value that
# minimises the 2nd-order Taylor expansion of the loss gives the Newton step
#   leaf = -sum(g) / (sum(h) + lambda)
# and the corresponding split gain (the standard XGBoost gain). This is a
# strict improvement over plain mean-of-residuals leaves because it accounts
# for how confident (h large) each point already is. `reg_lambda` (L2 on leaf
# weights) both regularises and stops the hessian, which shrinks toward 0 for
# already-confident points, from blowing leaf values up.

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_leaf=20, reg_lambda=1.0,
                 feature_subset=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.feature_subset = feature_subset  # column indices this tree may split on
        self.tree = None

    def fit(self, X, g, h, col_sorted):
        # col_sorted[f] = row indices sorting the WHOLE training set by feature f
        # (precomputed once by the booster, reused for every tree — the key
        # optimisation that removes per-node sorting).
        n = X.shape[0]
        feats = (self.feature_subset if self.feature_subset is not None
                 else np.arange(X.shape[1]))
        root_mask = np.ones(n, dtype=bool)
        self.tree = self._build(X, g, h, col_sorted, root_mask, feats, depth=0)
        return self

    def _leaf_value(self, g, h, mask):
        G = g[mask].sum()
        H = h[mask].sum()
        return -G / (H + self.reg_lambda)

    def _best_split(self, X, g, h, col_sorted, mask, feats):
        n_node = mask.sum()
        if n_node < 2 * self.min_samples_leaf:
            return None
        G_tot = g[mask].sum()
        H_tot = h[mask].sum()
        parent_score = (G_tot * G_tot) / (H_tot + self.reg_lambda)

        best = None
        best_gain = 0.0
        for f in feats:
            order_f = col_sorted[f]
            order = order_f[mask[order_f]]          # node rows, sorted by feature f
            vals = X[order, f]
            gs = np.cumsum(g[order])
            hs = np.cumsum(h[order])
            GL = gs[:-1]                              # left = first i+1 rows
            HL = hs[:-1]
            GR = G_tot - GL
            HR = H_tot - HL

            gain = (GL * GL) / (HL + self.reg_lambda) \
                 + (GR * GR) / (HR + self.reg_lambda) - parent_score

            # Only split between DISTINCT adjacent values, and only where both
            # children keep >= min_samples_leaf rows (BoW columns are tie-heavy).
            counts_left = np.arange(1, n_node)
            valid = (vals[:-1] != vals[1:]) \
                & (counts_left >= self.min_samples_leaf) \
                & (n_node - counts_left >= self.min_samples_leaf)
            if not valid.any():
                continue
            gain_valid = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain_valid))
            if gain_valid[i] > best_gain:
                thr = 0.5 * (vals[i] + vals[i + 1])
                best_gain = gain_valid[i]
                best = (f, thr)
        return best

    def _build(self, X, g, h, col_sorted, mask, feats, depth):
        if depth >= self.max_depth:
            return {"leaf": self._leaf_value(g, h, mask)}
        split = self._best_split(X, g, h, col_sorted, mask, feats)
        if split is None:
            return {"leaf": self._leaf_value(g, h, mask)}
        f, thr = split
        left_mask = mask & (X[:, f] <= thr)
        right_mask = mask & (X[:, f] > thr)
        return {
            "feature": f, "threshold": thr,
            "left": self._build(X, g, h, col_sorted, left_mask, feats, depth + 1),
            "right": self._build(X, g, h, col_sorted, right_mask, feats, depth + 1),
        }

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=np.float64)
        self._predict_node(self.tree, X, np.arange(X.shape[0]), out)
        return out

    def _predict_node(self, node, X, idx, out):
        # Route index groups down the tree with boolean masks (vectorised —
        # no per-row Python traversal).
        if "leaf" in node:
            out[idx] = node["leaf"]
            return
        go_left = X[idx, node["feature"]] <= node["threshold"]
        self._predict_node(node["left"], X, idx[go_left], out)
        self._predict_node(node["right"], X, idx[~go_left], out)


# ============================================================================
# 3. GRADIENT BOOSTING CLASSIFIER — from scratch
# ============================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=20, reg_lambda=1.0, colsample=1.0, seed=SEED):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.colsample = colsample  # fraction of features each tree may split on
        self.seed = seed
        self.trees = []

    def _precompute_sorted(self, X):
        # One global argsort per column, reused by every tree/node.
        return [np.argsort(X[:, f], kind="stable") for f in range(X.shape[1])]

    def fit(self, X, y, X_val=None, y_val=None, eval_every=0, verbose=False):
        n, n_feat = X.shape
        y = y.astype(np.float64)
        # Initialise the margin at the log-odds of the class prior — the
        # constant that already minimises logistic loss before any tree.
        p_pos = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.init_margin = np.log(p_pos / (1 - p_pos))

        col_sorted = self._precompute_sorted(X)
        F = np.full(n, self.init_margin, dtype=np.float64)
        F_val = (np.full(X_val.shape[0], self.init_margin)
                 if X_val is not None else None)

        local_rng = np.random.default_rng(self.seed)
        k = max(1, int(round(self.colsample * n_feat)))
        self.trees = []
        self.val_history = []  # (round, train_f1, val_f1)

        for m in range(self.n_estimators):
            p = sigmoid(F)
            g = p - y            # gradient of logistic loss
            h = p * (1.0 - p)    # hessian; negative gradient (residual) = y - p
            feats = (np.arange(n_feat) if k >= n_feat
                     else local_rng.choice(n_feat, size=k, replace=False))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda, feature_subset=feats,
            ).fit(X, g, h, col_sorted)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            if F_val is not None:
                F_val += self.learning_rate * tree.predict(X_val)

            if eval_every and ((m + 1) % eval_every == 0 or m == self.n_estimators - 1):
                tr_f1 = macro_f1(y.astype(int), (sigmoid(F) >= 0.5).astype(int))
                if F_val is not None:
                    val_f1 = macro_f1(y_val, (sigmoid(F_val) >= 0.5).astype(int))
                    self.val_history.append((m + 1, tr_f1, val_f1))
                    if verbose:
                        print(f"    round {m+1:4d}  train F1={tr_f1:.4f}  "
                              f"val F1={val_f1:.4f}  gap={tr_f1-val_f1:+.4f}")
        return self

    def decision_margin(self, X, n_trees=None):
        n_trees = len(self.trees) if n_trees is None else n_trees
        F = np.full(X.shape[0], self.init_margin, dtype=np.float64)
        for tree in self.trees[:n_trees]:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X, n_trees=None):
        return sigmoid(self.decision_margin(X, n_trees))

    def predict(self, X, threshold=0.5, n_trees=None):
        return (self.predict_proba(X, n_trees) >= threshold).astype(int)


# ============================================================================
# 4. FEATURE MATRIX ASSEMBLY (stylometric + moderate bag-of-words)
# ============================================================================

def build_full_matrix(texts, vocab):
    base = build_feature_matrix(texts)                       # 14 stylometric
    extra = np.vstack([extract_extra_features(t) for t in texts])  # +4
    counts = build_count_matrix(texts, vocab).astype(np.float64)   # +|vocab| BoW
    return np.hstack([base, extra, counts])


def scan_threshold(y_true, p, lo=0.30, hi=0.71, step=0.02):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi, step):
        f1 = macro_f1(y_true, (p >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("GenAI TEXT DETECTION — Gradient Boosting (from scratch)")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    y_all = train_df["label"].to_numpy(dtype=int)
    test_ids = test_df["id"].to_numpy()
    train_texts = train_df["text"].values
    test_texts = test_df["text"].values
    print(f"  train: {len(train_df)} rows | test: {len(test_df)} rows")
    print(f"  class balance: human={np.sum(y_all==0)} machine={np.sum(y_all==1)}")

    tr_idx, val_idx = stratified_split(y_all, val_frac=0.1, seed=SEED)
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]

    print("\n[2/7] Building vocabulary (top BoW words) from TRAIN split only...")
    MAX_VOCAB = 200  # moderate & dense — trees' sweet spot, not 5000-dim sparse
    vocab = build_vocabulary(train_texts[tr_idx], MAX_VOCAB)
    print(f"  vocab: {len(vocab)} words")

    print("\n[3/7] Assembling feature matrices (stylometric + BoW)...")
    t0 = time.time()
    X_all = build_full_matrix(train_texts, vocab)
    X_test = build_full_matrix(test_texts, vocab)
    all_feature_names = FEATURE_NAMES + EXTRA_FEATURE_NAMES + \
        [f"w:{w}" for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    print(f"  matrix shape: {X_all.shape} ({len(all_feature_names)} features) "
          f"in {time.time()-t0:.1f}s")
    X_tr, X_val = X_all[tr_idx], X_all[val_idx]

    # --- Sanity timing: one tree, then one full boosting round (task's flagged risk)
    print("\n[4/7] Timing sanity check (one tree, one boosting round)...")
    p0 = np.full(len(y_tr), y_tr.mean())
    g0, h0 = p0 - y_tr, p0 * (1 - p0)
    col_sorted_tr = [np.argsort(X_tr[:, f], kind="stable") for f in range(X_tr.shape[1])]
    t0 = time.time()
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=20).fit(
        X_tr, g0, h0, col_sorted_tr)
    print(f"  one depth-3 tree (all {X_tr.shape[1]} feats): {time.time()-t0:.2f}s")

    # --- Hyperparameter grid (run for real; track train vs val for overfitting).
    # All four requested knobs are tuned: n_estimators via per-round checkpoints,
    # plus learning_rate x max_depth x min_samples_leaf as a real grid.
    print("\n[5/7] Tuning lr x max_depth x min_samples_leaf — real experiments...")
    print("      colsample=0.6 per tree; reg_lambda=1.0")
    MAX_ROUNDS = 250
    grid = [(lr, depth, msl)
            for lr in (0.1, 0.3) for depth in (3, 4) for msl in (20, 50, 100)]
    best = {"val_f1": -1.0}
    for lr, depth, msl in grid:
        gb = GradientBoostingClassifier(
            n_estimators=MAX_ROUNDS, learning_rate=lr, max_depth=depth,
            min_samples_leaf=msl, reg_lambda=1.0, colsample=0.6, seed=SEED)
        t0 = time.time()
        gb.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, eval_every=25, verbose=False)
        # Pick the round with best val F1 (early-stopping-style, guards overfit).
        best_round, tr_at, val_at = max(gb.val_history, key=lambda r: r[2])
        print(f"  lr={lr} depth={depth} min_leaf={msl:3d}: best@round={best_round:3d} "
              f"train F1={tr_at:.4f} val F1={val_at:.4f} gap={tr_at-val_at:+.4f} "
              f"({time.time()-t0:.1f}s)")
        if val_at > best["val_f1"]:
            best = {"lr": lr, "depth": depth, "min_leaf": msl,
                    "n_estimators": best_round, "val_f1": val_at, "train_f1": tr_at}

    print(f"\n  Best config: lr={best['lr']} depth={best['depth']} "
          f"min_leaf={best['min_leaf']} n_estimators={best['n_estimators']}  "
          f"holdout val macro F1={best['val_f1']:.4f} (train={best['train_f1']:.4f}, "
          f"gap={best['train_f1']-best['val_f1']:+.4f})")

    # --- Cross-validated estimate of the CHOSEN config. The single 90/10 holdout
    # above is a noisy 2000-sample number, and it is selection-biased (max over
    # grid x checkpoints x thresholds on ONE val set). A 5-fold mean +/- spread is
    # the honest headline: it tells us whether ~0.83 is real or holdout luck.
    print("\n  Cross-validating the chosen config (5-fold, threshold=0.50)...")
    fold_f1s = []
    for fi, (k_tr, k_val) in enumerate(stratified_kfold(y_all, k=5, seed=SEED)):
        gb_cv = GradientBoostingClassifier(
            n_estimators=best["n_estimators"], learning_rate=best["lr"],
            max_depth=best["depth"], min_samples_leaf=best["min_leaf"],
            reg_lambda=1.0, colsample=0.6, seed=SEED).fit(X_all[k_tr], y_all[k_tr])
        f1 = macro_f1(y_all[k_val], gb_cv.predict(X_all[k_val]))
        fold_f1s.append(f1)
        print(f"    fold {fi+1}: val macro F1 = {f1:.4f}")
    cv_mean, cv_std = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    print(f"  5-fold macro F1 = {cv_mean:.4f} +/- {cv_std:.4f}")

    # --- Threshold tuning at the chosen config
    gb_best = GradientBoostingClassifier(
        n_estimators=best["n_estimators"], learning_rate=best["lr"],
        max_depth=best["depth"], min_samples_leaf=best["min_leaf"], reg_lambda=1.0,
        colsample=0.6, seed=SEED).fit(X_tr, y_tr)
    p_val = gb_best.predict_proba(X_val)
    thr, thr_f1 = scan_threshold(y_val, p_val)
    print(f"  threshold scan: best threshold={thr:.2f} -> val macro F1={thr_f1:.4f} "
          f"(vs {best['val_f1']:.4f} at 0.50)")
    final_val_f1 = max(thr_f1, best["val_f1"])
    if thr_f1 < best["val_f1"]:
        thr = 0.5

    # --- Refit on full 20K train, predict test
    print("\n[6/7] Refitting best config on full 20K training set...")
    t0 = time.time()
    gb_final = GradientBoostingClassifier(
        n_estimators=best["n_estimators"], learning_rate=best["lr"],
        max_depth=best["depth"], min_samples_leaf=best["min_leaf"], reg_lambda=1.0,
        colsample=0.6, seed=SEED).fit(X_all, y_all)
    print(f"  refit {best['n_estimators']} trees on {len(y_all)} rows "
          f"in {time.time()-t0:.1f}s")
    y_test_pred = gb_final.predict(X_test, threshold=thr)

    print("\n[7/7] Saving test predictions...")
    out_path = OUTPUT_DIR / "GradientBoosting_Scratch_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": y_test_pred.astype(int)}).to_csv(
        out_path, index=False)
    print(f"  saved: {out_path} ({len(y_test_pred)} rows, "
          f"machine={int(y_test_pred.sum())}, human={int((y_test_pred==0).sum())})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Features: {X_all.shape[1]} dense "
          f"(14 stylometric + {len(EXTRA_FEATURE_NAMES)} extra + {len(vocab)} BoW)")
    print(f"Best config: learning_rate={best['lr']}, max_depth={best['depth']}, "
          f"n_estimators={best['n_estimators']}, min_samples_leaf={best['min_leaf']}, "
          f"reg_lambda=1.0, colsample=0.6")
    print(f"Decision threshold: {thr:.2f}")
    print(f"Train macro F1: {best['train_f1']:.4f}  |  "
          f"Single-holdout val macro F1: {final_val_f1:.4f} (selection-biased)")
    print(f"HONEST HEADLINE — 5-fold CV macro F1: {cv_mean:.4f} +/- {cv_std:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
