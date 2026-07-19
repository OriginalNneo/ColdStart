"""
GenAI Text Detection — hand-rolled TF-IDF + hand-rolled linear classifier.
==========================================================================
No scikit-learn, no CountVectorizer/TfidfVectorizer, no pre-built classifier.
Only NumPy for the maths, scipy.sparse purely as a SPARSE STORAGE container
(a data structure, not an ML algorithm) so a ~600K-dim vocabulary stays
tractable in memory, and Pandas for reading CSVs. Tokenization via `re`.

Goal: reproduce the sklearn baseline's ~0.82 validation macro F1 (word 1-2
grams + char n-grams, sublinear TF-IDF, LinearSVC) entirely by hand.

Why this representation:
- Char n-grams were the single biggest lever in the sklearn winner: they
  capture punctuation / spacing / affix STYLE patterns (" the", "ing ",
  ", the") that generalise across topics far better than raw content words.
- Sublinear TF (1+log tf): a word appearing 10x is not 10x as informative;
  log-damping stops long documents' repeated tokens from dominating.
- Smoothed IDF (log((1+N)/(1+df))+1): down-weights ubiquitous tokens, and
  the +1 smoothing avoids divide-by-zero / never lets IDF hit exactly 0.
- L2 row-normalisation: makes every document a unit vector, so document
  LENGTH stops being a feature and — crucially — bounds the gradient scale,
  which keeps hand-rolled gradient descent well-conditioned (the main risk
  with a from-scratch optimiser on high-dim TF-IDF is under-convergence).
"""

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Reuse the already-correct, from-scratch split / metric helpers.
from ML_Project_NaiveBayes_Scratch import stratified_split, macro_f1

SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("predictions")
OUTPUT_DIR.mkdir(exist_ok=True)

WORD_RE = re.compile(r"[a-z0-9']+")  # applied to already-lowercased text


# ============================================================================
# 1. TOKENIZATION  (word 1-2 grams  +  char_wb 3-5 grams)
# ============================================================================

CHAR_NS = (3, 4, 5)  # char n-gram sizes


def word_tokens(words):
    """Word unigrams + adjacent bigrams. Bigrams capture phrase-level habits
    (transitions/hedging: "in conclusion", "it is") a single word can't."""
    toks = list(words)
    toks.extend(f"{words[i]}\x1f{words[i+1]}" for i in range(len(words) - 1))
    return toks


def char_tokens(text):
    """char_wb-style: n-grams taken WITHIN whitespace-delimited chunks, each
    padded with spaces. Padding turns word edges into features (" th", "ed ")
    — where a lot of the human/AI style difference actually lives. Splitting
    on whitespace (not WORD_RE) keeps punctuation attached, so ", " / "). "
    style habits survive."""
    toks = []
    for chunk in text.split():
        padded = f" {chunk} "
        L = len(padded)
        for n in CHAR_NS:
            for i in range(L - n + 1):
                toks.append(padded[i:i + n])
    return toks


def tokenize(text):
    """Return (word_token_list, char_token_list) for one raw document."""
    low = text.lower()
    words = WORD_RE.findall(low)
    return word_tokens(words), char_tokens(low)


# ============================================================================
# 2. VOCABULARY + IDF  (fit on TRAIN ONLY — no test/val leakage)
# ============================================================================

def build_vocab_and_idf(texts, min_df=2, max_word=120_000, max_char=520_000):
    """One pass to accumulate document frequencies for word- and char-tokens
    separately, prune anything rarer than min_df (kills train-specific noise
    tokens and keeps the matrix tractable), then map survivors to a single
    contiguous index space [words | chars] and derive smoothed IDF.

    Caps are generous safety valves, not the primary lever — min_df is."""
    N = len(texts)
    word_df, char_df = Counter(), Counter()
    for text in texts:
        wt, ct = tokenize(text)
        word_df.update(set(wt))   # set() => DOCUMENT frequency, not term freq
        char_df.update(set(ct))

    def keep(counter, cap):
        # min_df prune, then (if still over cap) keep the most frequent.
        items = [(t, d) for t, d in counter.items() if d >= min_df]
        if len(items) > cap:
            items.sort(key=lambda x: -x[1])
            items = items[:cap]
        return items

    word_items = keep(word_df, max_word)
    char_items = keep(char_df, max_char)

    vocab, dfs = {}, []
    for t, d in word_items:
        vocab[("w", t)] = len(vocab); dfs.append(d)
    n_word = len(vocab)
    for t, d in char_items:
        vocab[("c", t)] = len(vocab); dfs.append(d)

    dfs = np.asarray(dfs, dtype=np.float64)
    # Smoothed IDF (sklearn's smooth_idf form): pretend one extra doc holds
    # every term, so df is never 0 and IDF never blows up.
    idf = np.log((1.0 + N) / (1.0 + dfs)) + 1.0
    return vocab, idf, n_word


# ============================================================================
# 3. VECTORIZE  ->  sublinear TF-IDF, L2-normalised, scipy.sparse CSR
# ============================================================================

def vectorize(texts, vocab, idf):
    """Build a CSR matrix by hand: accumulate (doc, feature)->count, apply
    sublinear TF (1+log tf), multiply by IDF, L2-normalise each row.

    scipy.sparse is used ONLY to hold the result and to do X @ w later — it
    is storage + basic linear algebra, never a fitted ML routine."""
    indptr = [0]
    indices, data = [], []
    for text in texts:
        wt, ct = tokenize(text)
        counts = {}
        for tok in wt:
            j = vocab.get(("w", tok))
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for tok in ct:
            j = vocab.get(("c", tok))
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if counts:
            js = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
            tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
            tf = 1.0 + np.log(tf)              # sublinear TF
            vals = tf * idf[js]                 # TF-IDF
            norm = np.sqrt((vals * vals).sum())  # L2 row normalisation
            if norm > 0:
                vals /= norm
            indices.append(js)
            data.append(vals)
        indptr.append(indptr[-1] + len(counts))
    X = sp.csr_matrix(
        (np.concatenate(data) if data else np.zeros(0),
         np.concatenate(indices) if indices else np.zeros(0, np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(texts), len(vocab)),
    )
    return X


# ============================================================================
# 4. LINEAR CLASSIFIER — from scratch, mini-batch (sub)gradient descent
# ============================================================================
#
# Two losses, one code path:
#   loss='logistic' -> log loss, gradient (p - y)
#   loss='hinge'    -> linear SVM, subgradient on margin violations (this is
#                      what the 0.82 sklearn LinearSVC optimised, so it's the
#                      most faithful reproduction).
# L2 regularisation (reg) shrinks weights -> less overfitting to the 600K
# train-specific features. Class weighting up-weights the minority (human)
# class inversely to its frequency so the 62.5/37.5 imbalance doesn't push
# the boundary toward always predicting "machine" — macro F1 punishes that.

def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


class LinearClassifierScratch:
    def __init__(self, loss="hinge", reg=2e-4, lr=30.0, epochs=40, decay=0.1,
                 batch_size=512, class_weight=True, seed=SEED, verbose=False):
        self.loss = loss
        self.reg = reg          # L2 strength
        self.lr = lr            # base learning rate
        self.decay = decay      # LR decay: lr_t = lr / (1 + decay*epoch)
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.seed = seed
        self.verbose = verbose

    def _sample_weights(self, y):
        if not self.class_weight:
            return np.ones(len(y))
        # sklearn "balanced": n / (2 * n_c). Up-weights the rarer class.
        n = len(y)
        w = np.empty(n)
        for c in (0, 1):
            w[y == c] = n / (2.0 * np.sum(y == c))
        return w

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        sw = self._sample_weights(y)
        y_pm = np.where(y == 1, 1.0, -1.0)  # +/-1 targets for hinge

        # Because every row is L2-unit-normalised the per-sample gradient has
        # a small, uniform scale, so one decaying LR trains both losses
        # stably. (Pegasos' aggressive 1/(reg*t) step diverged at reg~2e-4 —
        # the ill-conditioning the row-normalisation was meant to tame.)
        for epoch in range(self.epochs):
            lr = self.lr / (1.0 + self.decay * epoch)
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                bidx = order[start:start + self.batch_size]
                Xb = X[bidx]
                swb = sw[bidx]
                scores = Xb @ self.w + self.b

                if self.loss == "logistic":
                    p = _sigmoid(scores)
                    resid = swb * (p - y[bidx])            # dL/dscore
                else:  # hinge — subgradient on margin violations
                    ypm = y_pm[bidx]
                    viol = (ypm * scores < 1).astype(np.float64)
                    resid = -swb * ypm * viol

                grad_w = (Xb.T @ resid) / len(bidx) + self.reg * self.w
                grad_b = resid.mean()  # bias is not regularised
                self.w -= lr * grad_w
                self.b -= lr * grad_b

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                msg = f"    epoch {epoch:2d}  train F1={macro_f1(y, self.predict(X)):.4f}"
                if X_val is not None:
                    msg += f"  val F1={macro_f1(y_val, self.predict(X_val)):.4f}"
                print(msg)
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X, threshold=0.0):
        return (self.decision_function(X) >= threshold).astype(int)


def best_threshold(scores, y_true, grid=None):
    """Pick the decision threshold on the raw score that maximises macro F1
    (works for both hinge and logistic — threshold tuning is not LR-only)."""
    if grid is None:
        lo, hi = np.percentile(scores, 5), np.percentile(scores, 95)
        grid = np.linspace(lo, hi, 61)
    best_t, best_f1 = 0.0, -1.0
    for t in grid:
        f1 = macro_f1(y_true, (scores >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 72)
    print("GenAI TEXT DETECTION — hand-rolled TF-IDF + linear classifier")
    print("=" * 72)

    print("\n[1/7] Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y_all = train["label"].to_numpy(dtype=int)
    texts_all = train["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    test_texts = test["text"].astype(str).to_numpy()
    print(f"  train={len(train)}  test={len(test)}  "
          f"human={np.sum(y_all==0)}  machine={np.sum(y_all==1)}")

    # ----- stratified 90/10 split; vocab+idf fit on TRAIN split only --------
    tr_idx, val_idx = stratified_split(y_all, val_frac=0.1, seed=SEED)
    texts_tr, y_tr = texts_all[tr_idx], y_all[tr_idx]
    texts_val, y_val = texts_all[val_idx], y_all[val_idx]
    print(f"  split: train={len(tr_idx)}  val={len(val_idx)}")

    # min_df=1 reproduces the baseline's ~640K-feature scale (val 0.8236);
    # min_df=2 halves the vocab for ~0.001 less F1 if memory is tight.
    MIN_DF = 1
    print(f"\n[2/7] Building vocabulary + IDF on TRAIN split (min_df={MIN_DF})...")
    t0 = time.time()
    vocab, idf, n_word = build_vocab_and_idf(texts_tr, min_df=MIN_DF)
    print(f"  vocab={len(vocab)}  (word={n_word}  char={len(vocab)-n_word})  "
          f"in {time.time()-t0:.1f}s")

    print("\n[3/7] Vectorizing (sublinear TF-IDF, L2-normalised, sparse)...")
    t0 = time.time()
    X_tr = vectorize(texts_tr, vocab, idf)
    X_val = vectorize(texts_val, vocab, idf)
    print(f"  X_tr={X_tr.shape} nnz={X_tr.nnz}  X_val={X_val.shape}  "
          f"in {time.time()-t0:.1f}s  (~{X_tr.data.nbytes/1e6:.0f}MB data)")

    # ----- tune loss + reg on the split ------------------------------------
    print("\n[4/7] Tuning classifier (train vs val macro F1 per config)...")
    print("      lr=50, 80 epochs. Low reg (~3e-5) needed: it lets train F1 climb")
    print("      to ~0.91 (capacity), which is what pulled val up to the baseline.")
    LR, EP = 50.0, 80
    configs = []
    for loss in ("hinge", "logistic"):
        for reg in (3e-5, 1e-4, 3e-4):
            clf = LinearClassifierScratch(loss=loss, reg=reg, lr=LR,
                                          epochs=EP, verbose=False).fit(X_tr, y_tr)
            s_tr = clf.decision_function(X_tr)
            s_val = clf.decision_function(X_val)
            tr_f1 = macro_f1(y_tr, (s_tr >= 0).astype(int))
            thr, val_f1 = best_threshold(s_val, y_val)
            configs.append((loss, reg, thr, tr_f1, val_f1))
            print(f"  {loss:8s} reg={reg:<7} train F1={tr_f1:.4f}  "
                  f"val F1(thr={thr:+.3f})={val_f1:.4f}")

    # Hard-select LOGISTIC for the shipped model even if hinge nudges ahead on
    # val. On this 2000-sample split the two tie to within noise (~0.001), but
    # hinge is decisively worse on the two things that decide the real test:
    #   (1) its subgradient OSCILLATES late (train F1 swings ~0.79<->0.92), so
    #       the final-epoch weights we must ship (the full-20K refit has no
    #       held-out set to early-stop on) are not a settled state;
    #   (2) its best threshold (~ -1.2) is a property of one unstable epoch's
    #       margin scale and does NOT transfer to the refit model, which
    #       skewed the test prediction to ~90% machine. Logistic's threshold
    #       sits at ~ -0.08 (prob ~0.48), right at the natural boundary, so it
    #       transfers cleanly. Logistic converges smoothly to train F1 ~0.91.
    log_configs = [c for c in configs if c[0] == "logistic"]
    best = max(log_configs, key=lambda c: c[4])
    b_loss, b_reg, _, _, b_val = best
    hinge_best = max((c for c in configs if c[0] == "hinge"), key=lambda c: c[4])
    print(f"\n  Bake-off: best hinge val F1={hinge_best[4]:.4f} (reg={hinge_best[1]}) "
          f"vs best logistic val F1={b_val:.4f} (reg={b_reg}).")
    print(f"  Selecting LOGISTIC (reg={b_reg}): tie on val, but stable + calibrated")
    print(f"  threshold that transfers to the full-data refit (hinge's does not).")

    # ----- refit best config with a verbose per-epoch convergence trace -----
    # Printing TRAIN F1 alongside val is the key diagnostic: if train plateaus
    # low, we're under-converged (need more epochs / higher lr) rather than
    # out of signal. Here train climbs well above val, so val is real headroom.
    print(f"\n[5/7] Refitting best config with convergence trace...")
    clf = LinearClassifierScratch(loss=b_loss, reg=b_reg, lr=LR, epochs=EP,
                                  verbose=True).fit(X_tr, y_tr, X_val, y_val)
    s_val = clf.decision_function(X_val)
    b_thr, val_f1 = best_threshold(s_val, y_val)
    tr_f1 = macro_f1(y_tr, clf.predict(X_tr))
    print(f"  FINAL split: train F1={tr_f1:.4f}  val F1={val_f1:.4f}  "
          f"(threshold={b_thr:+.3f}, gap={tr_f1-val_f1:+.4f})")

    # ----- refit vocab+idf+model on FULL 20K, predict test -----------------
    print("\n[6/7] Refitting on FULL 20K train (vocab+idf+model), predicting test...")
    t0 = time.time()
    vocab_f, idf_f, n_word_f = build_vocab_and_idf(texts_all, min_df=MIN_DF)
    X_full = vectorize(texts_all, vocab_f, idf_f)
    X_test = vectorize(test_texts, vocab_f, idf_f)
    print(f"  full vocab={len(vocab_f)}  X_full={X_full.shape}  X_test={X_test.shape}  "
          f"in {time.time()-t0:.1f}s")
    final = LinearClassifierScratch(loss=b_loss, reg=b_reg, lr=LR, epochs=EP,
                                    verbose=False).fit(X_full, y_all)
    # Reuse the threshold tuned on the held-out split (unbiased on full-train
    # scores it would just re-fit the training set).
    y_test = (final.decision_function(X_test) >= b_thr).astype(int)

    print("\n[7/7] Saving predictions...")
    out = OUTPUT_DIR / "TFIDF_Scratch_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": y_test}).to_csv(out, index=False)
    print(f"  Saved {out}  (rows={len(y_test)}  "
          f"machine={int(y_test.sum())}  human={int((y_test==0).sum())})")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Representation : word 1-2grams + char_wb 3-5grams, sublinear")
    print(f"                   TF-IDF, smoothed IDF, L2-normalised (from scratch)")
    print(f"  Best model     : {b_loss}  reg={b_reg}  threshold={b_thr:+.3f}")
    print(f"  Validation F1  : {val_f1:.4f}   (train F1={tr_f1:.4f})")
    print(f"  sklearn baseline: 0.8229 val macro F1")
    print(f"  Total runtime  : {time.time()-t_start:.0f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
