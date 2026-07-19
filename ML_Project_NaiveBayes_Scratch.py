"""
GenAI Text Detection — Gaussian Naive Bayes, built entirely from scratch.
==========================================================================
No scikit-learn. No pre-built vectorizers, classifiers, splitters, or
metrics. Only NumPy for array math and Pandas for reading CSVs.

Philosophy: instead of hundreds of thousands of sparse TF-IDF dimensions,
use a small number of hand-crafted STYLE features per document (word
length, vocabulary complexity, sentence structure, punctuation habits).

Why this can work:
- AI-generated text tends to favour longer, more "formal" vocabulary
  and more uniform sentence structure than human writing.
- A handful of well-chosen scalar features per document is far less
  prone to overfitting the training vocabulary than a 640K-dim sparse
  matrix — which matters because we saw the sparse TF-IDF model's score
  drop badly from validation (0.82) to the Kaggle leaderboard (0.73),
  a classic sign of overfitting to train-specific vocabulary.
- Gaussian Naive Bayes gives a genuine class probability P(machine | x),
  which is what we want when the loss (macro F1) is not simple accuracy —
  we can tune the decision threshold instead of hard-coding 0.5.
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("predictions")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. FEATURE ENGINEERING — small, hand-crafted, interpretable
# ============================================================================

# A short, hand-picked list of common English function words. Deliberately
# NOT using any library's stopword list — this is our own small set, good
# enough to separate "content words" (where vocabulary complexity shows up)
# from "function words" (which every writer, human or AI, must use).
STOPWORDS = set("""
the a an and or but if then so of in on at to for with as is are was were
be been being have has had do does did this that these those it its i you
he she we they them his her their our your my not no yes can will would
could should may might must from by about into over under again further
""".split())

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT_RE = re.compile(r"[.!?]+")
VOWEL_GROUP_RE = re.compile(r"[aeiouyAEIOUY]+")

LONG_WORD_LEN = 7        # "big word" threshold for AI-style vocabulary
VERY_LONG_WORD_LEN = 10  # stricter threshold


def count_syllables(word):
    """Rough syllable estimate: count vowel groups, floor at 1."""
    return max(1, len(VOWEL_GROUP_RE.findall(word)))


def extract_features(text):
    """Turn one raw document into a fixed-length vector of style features."""
    words = WORD_RE.findall(text)
    n_words = len(words)
    n_chars = len(text)

    if n_words == 0:
        return np.zeros(14, dtype=np.float64)

    word_lens = np.array([len(w) for w in words], dtype=np.float64)
    lower_words = [w.lower() for w in words]
    is_stop = np.array([w in STOPWORDS for w in lower_words])
    content_lens = word_lens[~is_stop]

    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    n_sent = max(1, len(sentences))

    syllables = np.array([count_syllables(w) for w in words], dtype=np.float64)

    unique_words = len(set(lower_words))
    punct_chars = sum(1 for c in text if c in ".,;:!?-—()\"'")
    digit_chars = sum(1 for c in text if c.isdigit())
    upper_chars = sum(1 for c in text if c.isupper())
    comma_count = text.count(",")

    features = np.array([
        word_lens.mean(),                                   # 0 avg word length
        word_lens.std(),                                     # 1 word length variety
        np.mean(word_lens > LONG_WORD_LEN),                   # 2 "big word" ratio
        np.mean(word_lens > VERY_LONG_WORD_LEN),              # 3 "very big word" ratio
        content_lens.mean() if len(content_lens) else 0.0,    # 4 avg CONTENT word length
        n_words / n_sent,                                     # 5 avg sentence length (words)
        unique_words / n_words,                               # 6 type-token ratio (diversity)
        syllables.mean(),                                     # 7 avg syllables per word
        np.mean(is_stop),                                     # 8 stopword ratio
        punct_chars / max(1, n_chars),                        # 9 punctuation density
        comma_count / n_words,                                # 10 commas per word (clause complexity)
        digit_chars / max(1, n_chars),                        # 11 digit density
        upper_chars / max(1, n_chars),                        # 12 uppercase density
        n_words / n_sent > 25,                                 # 13 "long sentence" flag (bool as float)
    ], dtype=np.float64)

    return features


FEATURE_NAMES = [
    "avg_word_len", "word_len_std", "big_word_ratio", "very_big_word_ratio",
    "avg_content_word_len", "avg_sentence_len", "type_token_ratio",
    "avg_syllables_per_word", "stopword_ratio", "punct_density",
    "commas_per_word", "digit_density", "upper_density", "long_sentence_flag",
]


def build_feature_matrix(texts):
    t0 = time.time()
    X = np.vstack([extract_features(t) for t in texts])
    print(f"  extracted {X.shape[1]} features for {X.shape[0]} docs in {time.time()-t0:.1f}s")
    return X


# ============================================================================
# 2. FROM-SCRATCH TRAIN/VALIDATION SPLIT (stratified, no sklearn)
# ============================================================================

def stratified_split(y, val_frac=0.1, seed=SEED):
    """Shuffle each class's indices separately, then carve off val_frac
    from each — preserves class balance in both splits without sklearn."""
    local_rng = np.random.default_rng(seed)
    tr_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        local_rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr_idx = np.concatenate(tr_idx)
    val_idx = np.concatenate(val_idx)
    local_rng.shuffle(tr_idx)
    local_rng.shuffle(val_idx)
    return tr_idx, val_idx


def stratified_kfold(y, k=5, seed=SEED):
    """k-fold splits, each class distributed evenly across folds. More
    robust than one 90/10 holdout: every hyperparameter choice below is
    scored as a MEAN over k folds instead of one noisy 2000-sample estimate,
    which matters when we're comparing configs that differ by ~0.01 F1."""
    local_rng = np.random.default_rng(seed)
    class_folds = {}
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        local_rng.shuffle(idx)
        class_folds[c] = np.array_split(idx, k)

    splits = []
    for fold in range(k):
        val_idx = np.concatenate([class_folds[c][fold] for c in class_folds])
        tr_idx = np.concatenate([
            class_folds[c][f] for c in class_folds for f in range(k) if f != fold
        ])
        local_rng.shuffle(tr_idx)
        local_rng.shuffle(val_idx)
        splits.append((tr_idx, val_idx))
    return splits


# ============================================================================
# 3. FROM-SCRATCH METRICS (no sklearn.metrics)
# ============================================================================

def macro_f1(y_true, y_pred):
    """Macro F1 = average of per-class F1, computed by hand."""
    f1s = []
    for c in (0, 1):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# ============================================================================
# 4. GAUSSIAN NAIVE BAYES — from scratch
# ============================================================================
#
# Bayes' theorem:  P(class | x) ∝ P(class) * P(x | class)
# Naive assumption: features are conditionally independent given the class,
# so  P(x | class) = product_i P(x_i | class).
# Gaussian assumption: each P(x_i | class) is a 1-D Normal distribution
# fitted separately per (feature, class) using the training data's mean
# and variance for that feature restricted to that class.
#
# We work in LOG space throughout (log-sum instead of product-of-many-small-
# numbers) purely for numerical stability — this is standard practice, not
# a "pre-built ML" shortcut.

class GaussianNaiveBayesScratch:
    def __init__(self, var_smoothing=1e-2):
        # var_smoothing: added to every feature's variance. Prevents a
        # near-zero variance from making the Gaussian spike to infinity for
        # points near the mean (a severe overfitting failure mode) and from
        # making it collapse to zero for points slightly off the mean.
        # This is the model's main hyperparameter.
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.mean_ = np.zeros((len(self.classes_), n_features))
        self.var_ = np.zeros((len(self.classes_), n_features))
        self.log_prior_ = np.zeros(len(self.classes_))

        # Smoothing scaled to each feature's own variance so one fixed
        # epsilon doesn't over- or under-smooth features on different scales.
        global_var = X.var(axis=0)
        eps = self.var_smoothing * global_var.mean()

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.mean_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + eps
            self.log_prior_[i] = np.log(len(Xc) / len(X))
        return self

    def _log_gaussian(self, X, mean, var):
        # log N(x; mu, sigma^2) = -0.5*log(2*pi*sigma^2) - (x-mu)^2/(2*sigma^2)
        return -0.5 * np.log(2 * np.pi * var) - ((X - mean) ** 2) / (2 * var)

    def predict_log_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            log_likelihood = self._log_gaussian(X, self.mean_[i], self.var_[i]).sum(axis=1)
            log_probs[:, i] = log_likelihood + self.log_prior_[i]
        # normalise (log-sum-exp) so we get genuine log-posteriors
        log_norm = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        return log_probs - log_norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X, threshold=0.5):
        """threshold applies to P(class=1 | x) — tunable for macro F1
        under class imbalance, instead of a hard-coded argmax at 0.5."""
        p_machine = self.predict_proba(X)[:, list(self.classes_).index(1)]
        return (p_machine >= threshold).astype(int)


# ============================================================================
# 4b. WORD-FREQUENCY NAIVE BAYES — from scratch
# ============================================================================
#
# The style features above only capture AGGREGATE statistics (one number
# per document). They plateau around 0.57 macro F1: real signal, but too
# coarse. This model instead learns a per-WORD probability under each
# class — e.g. it can directly learn "the word 'furthermore' is 3x more
# common in machine text" rather than only knowing the document's *average*
# word length. Still fully hand-rolled: a bag-of-words count matrix + a
# Multinomial Naive Bayes likelihood.
#
#   P(word_i | class) = (count(word_i, class) + alpha) /
#                        (total_words_in_class + alpha * vocab_size)
#
# alpha is Lidstone/Laplace smoothing — the direct analogue of
# var_smoothing above, and vocab_size is the other knob: too small
# underfits (throws away real signal), too large overfits (memorises
# train-specific words that won't appear the same way in test — exactly
# the failure mode that hurt the sparse TF-IDF model on the leaderboard).

def tokenize(text, ngram_range=(1, 1), word_filter=None):
    """Turn raw text into a token stream. ngram_range=(1,2) adds adjacent
    word-pairs ("in_conclusion") alongside unigrams — phrase-level habits
    (hedging, transitions) that a single word can't capture. word_filter is
    an optional predicate applied to UNIGRAMS only (e.g. keep only stopwords,
    or only non-stopwords) — bigrams always pass through unfiltered since
    splitting a phrase mid-word breaks its meaning."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    tokens = []
    if ngram_range[0] <= 1 <= ngram_range[1]:
        tokens.extend(w for w in words if word_filter is None or word_filter(w))
    if ngram_range[1] >= 2:
        tokens.extend(f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1))
    return tokens


def build_vocabulary(texts, max_vocab, ngram_range=(1, 1), word_filter=None):
    """Rank tokens in the TRAINING corpus by frequency, keep the top
    max_vocab. Only ever called on train text to avoid leaking test-set
    vocabulary into feature selection."""
    from collections import Counter
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text, ngram_range, word_filter))
    most_common = [w for w, _ in counts.most_common(max_vocab)]
    return {w: i for i, w in enumerate(most_common)}


def build_count_matrix(texts, vocab, ngram_range=(1, 1), word_filter=None):
    """Bag-of-words count matrix, shape (n_docs, len(vocab)). uint16 keeps
    memory small — no document uses any single token >65535 times."""
    X = np.zeros((len(texts), len(vocab)), dtype=np.uint16)
    for row, text in enumerate(texts):
        for tok in tokenize(text, ngram_range, word_filter):
            idx = vocab.get(tok)
            if idx is not None:
                X[row, idx] += 1
    return X


class MultinomialNaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Lidstone smoothing strength

    def fit(self, X_counts, y):
        self.classes_ = np.unique(y)
        vocab_size = X_counts.shape[1]
        self.log_prob_ = np.zeros((len(self.classes_), vocab_size))
        self.log_prior_ = np.zeros(len(self.classes_))

        for i, c in enumerate(self.classes_):
            Xc = X_counts[y == c].astype(np.float64)
            word_totals = Xc.sum(axis=0)
            class_total = word_totals.sum()
            self.log_prob_[i] = np.log(
                (word_totals + self.alpha) / (class_total + self.alpha * vocab_size)
            )
            self.log_prior_[i] = np.log(len(Xc) / len(X_counts))
        return self

    def predict_log_proba(self, X_counts):
        log_probs = X_counts.astype(np.float64) @ self.log_prob_.T + self.log_prior_
        log_norm = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        return log_probs - log_norm

    def predict_proba(self, X_counts):
        return np.exp(self.predict_log_proba(X_counts))

    def predict(self, X_counts, threshold=0.5):
        p_machine = self.predict_proba(X_counts)[:, list(self.classes_).index(1)]
        return (p_machine >= threshold).astype(int)

    def most_diagnostic_words(self, vocab, top_n=15):
        """log-odds of each word favouring class 1 (machine) vs class 0 (human)
        — a from-scratch look at exactly which words drive the decision."""
        log_odds = self.log_prob_[1] - self.log_prob_[0]
        idx_to_word = {i: w for w, i in vocab.items()}
        order = np.argsort(-log_odds)
        machine_words = [(idx_to_word[i], log_odds[i]) for i in order[:top_n]]
        human_words = [(idx_to_word[i], log_odds[i]) for i in order[::-1][:top_n]]
        return machine_words, human_words


class HybridNaiveBayesScratch:
    """Combines the Gaussian style-feature likelihood and the Multinomial
    word-count likelihood under ONE shared posterior. Valid because Naive
    Bayes only assumes features are conditionally independent given the
    class — nothing requires them to share a distribution family, so we
    simply add both families' log-likelihoods before applying the (single,
    shared) class prior once."""

    def __init__(self, var_smoothing, alpha, style_weight=0.5):
        # style_weight: how much of the combined evidence comes from the
        # style features vs the word-frequency features (word_weight =
        # 1 - style_weight). Plain addition of two log-likelihoods implicitly
        # assumes both families are equally informative and equally
        # calibrated in scale — that's not guaranteed, so this is exposed as
        # a tunable hyperparameter rather than assumed.
        self.gnb = GaussianNaiveBayesScratch(var_smoothing=var_smoothing)
        self.mnb = MultinomialNaiveBayesScratch(alpha=alpha)
        self.style_weight = style_weight

    def fit(self, X_style, X_counts, y):
        self.gnb.fit(X_style, y)
        self.mnb.fit(X_counts, y)
        self.classes_ = self.gnb.classes_
        return self

    @staticmethod
    def _logit(p, eps=1e-9):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def predict_proba(self, X_style, X_counts):
        # Earlier attempt summed raw (un-normalised) log-likelihoods scaled
        # by 1/n_features. That silently crushed the word-frequency signal
        # to near-zero whenever vocab_size was large (the class prior then
        # dominated every prediction — a bug caught by 5-fold CV collapsing
        # to the "always predict majority class" macro F1 of 0.3847 at every
        # weight). The fix: let each sub-model produce its OWN properly
        # normalised posterior probability first, then pool the two
        # probabilities via a weighted average in LOG-ODDS space (a
        # logarithmic opinion pool). This is scale-invariant by
        # construction — each p is already in [0, 1] regardless of how many
        # features fed into it.
        idx1 = list(self.classes_).index(1)
        p_style = self.gnb.predict_proba(X_style)[:, idx1]
        p_word = self.mnb.predict_proba(X_counts)[:, idx1]

        combined_logit = (self.style_weight * self._logit(p_style)
                           + (1 - self.style_weight) * self._logit(p_word))
        p_machine = 1.0 / (1.0 + np.exp(-combined_logit))
        return np.column_stack([1 - p_machine, p_machine])

    def predict(self, X_style, X_counts, threshold=0.5):
        p_machine = self.predict_proba(X_style, X_counts)[:, list(self.classes_).index(1)]
        return (p_machine >= threshold).astype(int)


# ============================================================================
# 5. FEATURE REFINEMENT — rank features by class-separation power
# ============================================================================

def feature_separation_scores(X, y):
    """Standardized mean difference between classes, per feature — a simple,
    from-scratch stand-in for an ANOVA F-score. Higher = more separating."""
    mu0, mu1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    pooled_std = np.sqrt((X[y == 0].var(axis=0) + X[y == 1].var(axis=0)) / 2) + 1e-9
    return np.abs(mu1 - mu0) / pooled_std


def select_decorrelated_features(X, y, max_corr=0.6):
    """Greedily pick features by separation power, skipping any feature that
    is highly correlated with one already chosen.

    Naive Bayes assumes features are conditionally independent given the
    class. Feeding it several near-duplicate features (e.g. avg_word_len,
    big_word_ratio, avg_syllables_per_word — all just "word complexity"
    restated) breaks that assumption: the model treats one real signal as
    several independent pieces of evidence and over-weights it. Enforcing a
    correlation cap keeps each selected feature contributing NEW information.
    """
    scores = feature_separation_scores(X, y)
    corr = np.corrcoef(X.T)
    order = np.argsort(-scores)

    selected = []
    for idx in order:
        if all(abs(corr[idx, j]) <= max_corr for j in selected):
            selected.append(idx)
    return selected


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("GenAI TEXT DETECTION — Gaussian Naive Bayes (from scratch)")
    print("=" * 70)

    print("\n[1/9] Loading data...")
    train_text = pd.read_csv(DATA_DIR / "train.csv")   # id, text, label
    test_text = pd.read_csv(DATA_DIR / "test.csv")      # id, text
    y_all = train_text["label"].to_numpy(dtype=int)
    test_ids = test_text["id"].to_numpy()
    print(f"  train: {len(train_text)} rows | test: {len(test_text)} rows")
    print(f"  class balance: human={np.sum(y_all==0)} machine={np.sum(y_all==1)}")

    print("\n[2/9] Extracting hand-crafted style features...")
    X_all = build_feature_matrix(train_text["text"].values)
    X_test = build_feature_matrix(test_text["text"].values)

    tr_idx, val_idx = stratified_split(y_all, val_frac=0.1, seed=SEED)
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"  train split: {len(tr_idx)} | validation split: {len(val_idx)}")

    # ------------------------------------------------------------------
    print("\n[3/9] Ranking features by class-separation power...")
    sep_scores = feature_separation_scores(X_tr, y_tr)
    ranked = sorted(zip(FEATURE_NAMES, sep_scores), key=lambda t: -t[1])
    for name, score in ranked:
        print(f"  {name:24s} separation={score:.4f}")

    print("\n  De-duplicating: dropping features highly correlated (>0.6) with")
    print("  a stronger feature already kept (Naive Bayes double-counts these)...")
    decorr_idx = select_decorrelated_features(X_tr, y_tr, max_corr=0.6)
    decorr_names = [FEATURE_NAMES[i] for i in decorr_idx]
    dropped = [n for n in FEATURE_NAMES if n not in decorr_names]
    print(f"  kept:    {decorr_names}")
    print(f"  dropped: {dropped}")

    # Restrict everything downstream to the decorrelated subset.
    X_tr, X_val = X_tr[:, decorr_idx], X_val[:, decorr_idx]
    X_all_d, X_test_d = X_all[:, decorr_idx], X_test[:, decorr_idx]

    # ------------------------------------------------------------------
    print("\n[4/9] Tuning var_smoothing (bias/variance trade-off)...")
    print("      Reporting TRAIN vs VAL macro F1 to watch for over/underfitting.")
    best_vs, best_val_f1 = None, -1.0
    for vs in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
        model = GaussianNaiveBayesScratch(var_smoothing=vs).fit(X_tr, y_tr)
        tr_f1 = macro_f1(y_tr, model.predict(X_tr))
        val_f1 = macro_f1(y_val, model.predict(X_val))
        gap = tr_f1 - val_f1
        flag = "  <-- best" if val_f1 > best_val_f1 else ""
        print(f"  var_smoothing={vs:<8} train F1={tr_f1:.4f}  val F1={val_f1:.4f}  gap={gap:+.4f}{flag}")
        if val_f1 > best_val_f1:
            best_vs, best_val_f1 = vs, val_f1

    print(f"\n  Selected var_smoothing = {best_vs} (val macro F1 = {best_val_f1:.4f})")

    # ------------------------------------------------------------------
    # decorr_names is already ordered by separation power (strongest first),
    # so shrinking from the right is still "drop the weakest, then next-weakest".
    print("\n[5/9] Refining feature subset within the decorrelated set...")
    best_k, best_subset_f1, best_cols = len(decorr_names), -1.0, list(range(len(decorr_names)))
    for k in range(len(decorr_names), 1, -1):
        cols = list(range(k))
        model = GaussianNaiveBayesScratch(var_smoothing=best_vs).fit(X_tr[:, cols], y_tr)
        val_f1 = macro_f1(y_val, model.predict(X_val[:, cols]))
        print(f"  top-{k:2d} features: val macro F1 = {val_f1:.4f}")
        if val_f1 > best_subset_f1:
            best_k, best_subset_f1, best_cols = k, val_f1, cols

    final_feature_names = decorr_names[:best_k]
    print(f"\n  Selected top-{best_k} features (val macro F1 = {best_subset_f1:.4f}):")
    print(f"    {final_feature_names}")

    # Freeze the style-model config; used again inside the Hybrid below.
    style_cols = best_cols
    gnb_thresh_scan = lambda p_val: max(
        ((t, macro_f1(y_val, (p_val >= t).astype(int))) for t in np.arange(0.30, 0.71, 0.02)),
        key=lambda t: t[1]
    )
    style_model = GaussianNaiveBayesScratch(var_smoothing=best_vs).fit(X_tr[:, style_cols], y_tr)
    p_val_style = style_model.predict_proba(X_val[:, style_cols])[:, list(style_model.classes_).index(1)]
    style_thresh, style_f1 = gnb_thresh_scan(p_val_style)
    print(f"\n  [Style-only] best threshold={style_thresh:.2f} -> val macro F1={style_f1:.4f}")

    # ------------------------------------------------------------------
    print("\n[6/9] Building word-frequency features (bag-of-words counts)...")
    MAX_VOCAB = 5000
    train_texts_tr = train_text["text"].iloc[tr_idx].values
    vocab = build_vocabulary(train_texts_tr, MAX_VOCAB)
    print(f"  vocabulary built from TRAIN split only: {len(vocab)} words")

    t0 = time.time()
    C_tr = build_count_matrix(train_texts_tr, vocab)
    C_val = build_count_matrix(train_text["text"].iloc[val_idx].values, vocab)
    print(f"  count matrices built in {time.time()-t0:.1f}s "
          f"(train {C_tr.shape}, val {C_val.shape})")

    # ------------------------------------------------------------------
    print("\n[7/9] Sweeping vocab_size x alpha (bias/variance trade-off)...")
    print("      Reporting TRAIN vs VAL macro F1 to watch for over/underfitting.")
    best_vocab_size, best_alpha, best_mnb_f1 = None, None, -1.0
    for vocab_size in [200, 500, 1000, 2000, 5000]:
        cols = slice(0, vocab_size)
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            mnb = MultinomialNaiveBayesScratch(alpha=alpha).fit(C_tr[:, cols], y_tr)
            tr_f1 = macro_f1(y_tr, mnb.predict(C_tr[:, cols]))
            val_f1 = macro_f1(y_val, mnb.predict(C_val[:, cols]))
            gap = tr_f1 - val_f1
            flag = "  <-- best" if val_f1 > best_mnb_f1 else ""
            print(f"  vocab={vocab_size:5d} alpha={alpha:<4} train F1={tr_f1:.4f} "
                  f"val F1={val_f1:.4f} gap={gap:+.4f}{flag}")
            if val_f1 > best_mnb_f1:
                best_vocab_size, best_alpha, best_mnb_f1 = vocab_size, alpha, val_f1

    print(f"\n  Selected vocab_size={best_vocab_size}, alpha={best_alpha} "
          f"(val macro F1 = {best_mnb_f1:.4f})")

    mnb_cols = slice(0, best_vocab_size)
    mnb_model = MultinomialNaiveBayesScratch(alpha=best_alpha).fit(C_tr[:, mnb_cols], y_tr)
    p_val_mnb = mnb_model.predict_proba(C_val[:, mnb_cols])[:, list(mnb_model.classes_).index(1)]
    mnb_thresh, mnb_f1 = gnb_thresh_scan(p_val_mnb)
    print(f"  [Word-freq only] best threshold={mnb_thresh:.2f} -> val macro F1={mnb_f1:.4f}")

    machine_words, human_words = mnb_model.most_diagnostic_words(vocab, top_n=12)
    print("\n  Most MACHINE-leaning words (highest log-odds toward class 1):")
    print(f"    {[w for w, _ in machine_words]}")
    print("  Most HUMAN-leaning words (highest log-odds toward class 0):")
    print(f"    {[w for w, _ in human_words]}")

    # ------------------------------------------------------------------
    print("\n[8/9] Combining both into one Hybrid Bayesian model...")
    hybrid = HybridNaiveBayesScratch(var_smoothing=best_vs, alpha=best_alpha)
    hybrid.fit(X_tr[:, style_cols], C_tr[:, mnb_cols], y_tr)
    p_val_hybrid = hybrid.predict_proba(X_val[:, style_cols], C_val[:, mnb_cols])[
        :, list(hybrid.classes_).index(1)
    ]
    hybrid_thresh, hybrid_f1 = gnb_thresh_scan(p_val_hybrid)
    print(f"  [Hybrid: style + word-freq] best threshold={hybrid_thresh:.2f} "
          f"-> val macro F1={hybrid_f1:.4f}")

    # ------------------------------------------------------------------
    print("\n[9/9] Bake-off: choosing the best model on validation macro F1...")
    candidates = {
        "style_only": style_f1,
        "word_freq_only": mnb_f1,
        "hybrid": hybrid_f1,
    }
    for name, f1 in sorted(candidates.items(), key=lambda t: -t[1]):
        print(f"  {name:16s} val macro F1 = {f1:.4f}")
    winner = max(candidates, key=candidates.get)
    print(f"\n  Winner: {winner} (val macro F1 = {candidates[winner]:.4f})")

    print("\nRefitting winner on full 20K training set...")
    C_all = build_count_matrix(train_text["text"].values, vocab)[:, mnb_cols]
    C_test = build_count_matrix(test_text["text"].values, vocab)[:, mnb_cols]

    if winner == "style_only":
        final_model = GaussianNaiveBayesScratch(var_smoothing=best_vs).fit(X_all_d[:, style_cols], y_all)
        y_test_pred = final_model.predict(X_test_d[:, style_cols], threshold=style_thresh)
        final_thresh = style_thresh
    elif winner == "word_freq_only":
        final_model = MultinomialNaiveBayesScratch(alpha=best_alpha).fit(C_all, y_all)
        y_test_pred = final_model.predict(C_test, threshold=mnb_thresh)
        final_thresh = mnb_thresh
    else:
        final_model = HybridNaiveBayesScratch(var_smoothing=best_vs, alpha=best_alpha)
        final_model.fit(X_all_d[:, style_cols], C_all, y_all)
        y_test_pred = final_model.predict(X_test_d[:, style_cols], C_test, threshold=hybrid_thresh)
        final_thresh = hybrid_thresh

    out_path = OUTPUT_DIR / "NaiveBayes_Scratch_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": y_test_pred}).to_csv(out_path, index=False)
    print(f"✓ Saved: {out_path} ({len(y_test_pred)} rows, "
          f"machine={y_test_pred.sum()}, human={(y_test_pred==0).sum()})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Winning model: {winner}")
    print(f"Style features ({best_k}): {final_feature_names}")
    print(f"Word-freq vocab_size={best_vocab_size}, alpha={best_alpha}")
    print(f"Gaussian var_smoothing={best_vs}, decision_threshold={final_thresh:.2f}")
    print(f"Final validation macro F1: {candidates[winner]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
