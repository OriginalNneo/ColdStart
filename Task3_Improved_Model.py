"""
Task 3 ("race to the top") — Improved model search with SHIFT-AWARE validation.
================================================================================
Deliverable for SUTD 50.007 ML project (GenAI academic-abstract detection).
Classical ML only (sklearn / lightgbm). NO deep learning / LLMs.

WHY THIS SCRIPT EXISTS
----------------------
The current team-best is a word(1-2) + char_wb(3-5) TF-IDF + LinearSVC
(C=0.25, class_weight=balanced): single-holdout val macro-F1 0.8229, but it
dropped to 0.7299 on the real Kaggle public leaderboard. That ~0.09 collapse
is a TRAIN -> TEST DISTRIBUTION SHIFT (test abstracts cover different research
topics, so topic vocabulary does not transfer; only style/char patterns do).

CRITICAL METHODOLOGICAL POINT
-----------------------------
A random k-fold split of the training data CANNOT see this shift — every fold
is drawn from the same distribution. So "beat 0.8229 on CV" is necessary but
NOT sufficient. This script therefore evaluates every candidate on THREE lenses:

  (1) Stratified 5-fold CV            -> in-distribution generalization
  (2) Cluster-holdout CV (topic KMeans, hold out whole topic clusters)
                                      -> a proxy for TOPIC shift (the real risk)
  (3) Adversarial validation AUC + a "test-like" weighted holdout
                                      -> how visible the train/test shift is,
                                         and score on the most test-like rows

Lens (2)/(3) — not (1) — are the leaderboard proxy. We optimize THOSE.

APPLES-TO-APPLES
----------------
The 0.8229 headline was a single holdout. We re-run the EXACT baseline through
our own harness so the comparison is fair; the real bar is beating the baseline
under the identical harness, on the shift-aware lenses.

CANDIDATES COMPARED (all documented for the "3a: models tried" deliverable)
---------------------------------------------------------------------------
  BASE   : word(1,2) + char_wb(3,5) TF-IDF, LinearSVC C=0.25, balanced   [reproduce]
  CHAR   : char_wb(3,5)-ONLY TF-IDF, LinearSVC C tuned, balanced
           (drop topic-word n-grams entirely -> forced to rely on transferable
            style features; the documented lever)
  CHARW  : char_wb(3,5) + capped/strongly-pruned word(1,2) (min_df high),
           LinearSVC — a middle ground
  CHARLR : char_wb(3,5)-only TF-IDF, calibrated LogisticRegression
           (needed for soft-voting; LinearSVC has no predict_proba)
  ENSMB  : soft-vote (equal weight) of CHARLR (sparse char style) + LightGBM on
           DENSE, SHIFT-ROBUST stylometric features (genuinely different
           representation, so the errors are less correlated than the notebook's
           earlier 5000-feature ensembles that stalled at 0.7578). Each leg AND
           the blend are judged on the shift-aware lenses — a shift-fragile leg
           can raise CV while lowering the leaderboard, so we check.

THE STYLOMETRIC LEG WAS STRESS-TESTED FOR ITS OWN SHIFT
-------------------------------------------------------
Neither proxy above stresses drift in *stylometric* space (cluster-holdout
splits on word-unigrams; the test-like weight comes from *char* adversarial).
So we separately ran adversarial validation on the stylometric features alone:
the FULL 18-feature set had adversarial AUC 0.79 — it drifts as hard as char
n-grams (test abstracts are ~25% longer: absolute-count features were the top
shift drivers). We therefore DROP the covariate-shift-prone features (absolute
counts + length-correlated ratios), leaving 10 scale-robust density/ratio
features (adversarial AUC drops to 0.74). Critically, the blend STILL beats the
baseline on both shift proxies AFTER this pruning (cluster +0.05, test-like
+0.024), so the ensemble's edge is not merely an artifact of exploiting the
length shift. w_char is kept at 0.5 (equal weight): we lean on the char leg,
which the cluster/char-adversarial proxies directly validate.

DECISION RULE
-------------
Ship a new prediction file ONLY if a candidate beats the baseline UNDER THE SAME
HARNESS on the shift-aware lenses (cluster-holdout topic-shift proxy), by a
non-noise margin, with a train/val gap NO WORSE THAN THE BASELINE'S on that same
lens (all these bag-of-features models memorize train to ~0.99, so gap is judged
relative to the incumbent, not against an absolute constant). Otherwise the
honest, allowed outcome is "the existing 0.7299 submission stands" — we will not
fit CV harder to manufacture a win.

RESULT (this run): ENSMB is the winner and a new prediction file is written.

Run:  .venv/bin/python Task3_Improved_Model.py
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("predictions")
OUT_DIR.mkdir(exist_ok=True)
N_SPLITS = 5
BASELINE_HEADLINE = 0.8229  # prior single-holdout number, for reference only


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


# ============================================================================
# 1. DENSE STYLOMETRIC FEATURES (for the LightGBM ensemble leg)
# ----------------------------------------------------------------------------
# A genuinely DIFFERENT representation from sparse char TF-IDF: aggregate
# style signals (length regularity, punctuation/number density, lexical
# diversity) that don't depend on which words/topics appear -> designed to
# survive topic shift. Trees model their non-linear interactions.
# ============================================================================

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENT_RE = re.compile(r"[.!?]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "is", "are", "was", "were", "be", "this",
    "that", "these", "those", "we", "our", "it", "its", "from", "which",
}

STYLO_NAMES = [
    "n_chars", "n_words", "avg_word_len", "word_len_std", "type_token_ratio",
    "n_sentences", "avg_sent_len", "sent_len_std", "comma_density",
    "period_density", "semicolon_density", "paren_density", "digit_ratio",
    "upper_ratio", "stopword_ratio", "unique_bigram_ratio", "hapax_ratio",
    "exclaim_question_ratio",
]


def stylometric(text):
    text = str(text)
    n_chars = len(text)
    words = WORD_RE.findall(text)
    n_words = len(words)
    if n_words == 0:
        return np.zeros(len(STYLO_NAMES))
    wlens = np.array([len(w) for w in words], dtype=float)
    low = [w.lower() for w in words]
    counts = {}
    for w in low:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    slens = np.array([len(WORD_RE.findall(s)) for s in sents], dtype=float) \
        if sents else np.array([n_words], dtype=float)
    bigrams = [low[i] + "_" + low[i + 1] for i in range(len(low) - 1)]
    ubr = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    n_alpha = sum(c.isalpha() for c in text) or 1
    return np.array([
        n_chars,
        n_words,
        wlens.mean(),
        wlens.std(),
        len(counts) / n_words,
        len(sents),
        slens.mean(),
        slens.std(),
        text.count(",") / n_words,
        text.count(".") / n_words,
        text.count(";") / n_words,
        (text.count("(") + text.count(")")) / n_words,
        sum(c.isdigit() for c in text) / n_chars if n_chars else 0.0,
        sum(c.isupper() for c in text) / n_alpha,
        sum(w in STOPWORDS for w in low) / n_words,
        ubr,
        hapax / n_words,
        (text.count("!") + text.count("?")) / n_words,
    ], dtype=float)


def build_stylo(texts):
    return np.vstack([stylometric(t) for t in texts])


# Features dropped from the ensemble's LightGBM leg because adversarial
# validation flagged them as the top TRAIN->TEST covariate-shift drivers
# (absolute counts + length-correlated ratios). Keeping only scale-robust
# density/ratio features cuts the stylometric adversarial AUC from ~0.79 to
# ~0.74 while preserving the blend's win on both shift proxies.
DROP_STYLO = {
    "n_chars", "n_words", "n_sentences", "avg_word_len", "word_len_std",
    "type_token_ratio", "unique_bigram_ratio", "avg_sent_len",
}
ROBUST_STYLO_IDX = [i for i, n in enumerate(STYLO_NAMES) if n not in DROP_STYLO]
W_CHAR = 0.5  # ensemble blend weight on the (validated) char leg


def gbm_leg():
    """LightGBM on the dense, shift-robust stylometric features."""
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1)


# ============================================================================
# 2. VECTORIZER / ESTIMATOR FACTORIES
# ----------------------------------------------------------------------------
# Factories (not fitted objects) so each CV fold fits ONLY on its train part
# -> no leakage of vocabulary/idf from held-out rows.
# ============================================================================

def vec_word_char():
    """BASELINE representation: word 1-2 grams + char_wb 3-5 grams."""
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        sublinear_tf=True),
    ]


def vec_char_only():
    """CHAR: char_wb 3-5 grams ONLY — drop topic-word n-grams entirely."""
    return [
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        sublinear_tf=True),
    ]


def vec_char_prunedword():
    """CHARW: char_wb 3-5 + heavily pruned word 1-2 (min_df=10 keeps only
    common, more topic-agnostic words)."""
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=10,
                        max_df=0.5, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        sublinear_tf=True),
    ]


class MultiVec:
    """Fit a list of vectorizers and hstack their sparse outputs."""

    def __init__(self, factory):
        self.factory = factory

    def fit(self, texts):
        self.vecs = self.factory()
        self.mats = [v.fit(texts) for v in self.vecs]
        return self

    def transform(self, texts):
        return sparse.hstack([v.transform(texts) for v in self.vecs]).tocsr()

    def fit_transform(self, texts):
        self.vecs = self.factory()
        return sparse.hstack([v.fit_transform(texts) for v in self.vecs]).tocsr()


def est_svc(C):
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def est_char_logreg(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                              solver="liblinear", random_state=SEED)


# ============================================================================
# 3. CV HARNESSES
# ============================================================================

def stratified_folds(y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def cluster_folds(texts, y, n_clusters=10):
    """TOPIC-SHIFT proxy: cluster docs by content, then hold out WHOLE clusters.
    A fold's held-out rows are topics under-represented in its training rows,
    mimicking the real test set's unseen research areas."""
    # Cheap content vectorization for clustering only (word unigrams).
    cv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=5,
                         max_features=20000, sublinear_tf=True)
    Xc = cv.fit_transform(texts)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, n_init=5,
                         batch_size=2048)
    cl = km.fit_predict(Xc)
    # Merge clusters into N_SPLITS groups (balanced by size), hold out one group.
    order = np.argsort(-np.bincount(cl, minlength=n_clusters))
    group_of_cluster = {c: i % N_SPLITS for i, c in enumerate(order)}
    grp = np.array([group_of_cluster[c] for c in cl])
    folds = []
    for g in range(N_SPLITS):
        val = np.where(grp == g)[0]
        tr = np.where(grp != g)[0]
        if len(val) and len(np.unique(y[val])) == 2:
            folds.append((tr, val))
    return folds, cl


# ============================================================================
# 4. CANDIDATE EVALUATION (linear / SVC candidates over a MultiVec rep)
# ============================================================================

def eval_linear(texts, y, folds, factory, est_factory):
    """Return (mean_val_f1, std, mean_train_f1, gap) over the given folds."""
    val_f1s, tr_f1s = [], []
    for tr, val in folds:
        mv = MultiVec(factory)
        Xtr = mv.fit_transform(texts[tr])
        Xval = mv.transform(texts[val])
        clf = est_factory()
        clf.fit(Xtr, y[tr])
        val_f1s.append(macro_f1(y[val], clf.predict(Xval)))
        tr_f1s.append(macro_f1(y[tr], clf.predict(Xtr)))
    v, t = np.array(val_f1s), np.array(tr_f1s)
    return v.mean(), v.std(), t.mean(), t.mean() - v.mean()


def eval_ensemble(texts, y, stylo, folds):
    """Soft-vote: calibrated char-LogReg (sparse style) + LightGBM (dense,
    shift-robust stylometric). Also reports each leg alone on the same folds so
    we can see whether the blend helps or a fragile leg drags it down."""
    stylo = stylo[:, ROBUST_STYLO_IDX]  # drop covariate-shift-prone features
    blend_f1, char_f1, gbm_f1, tr_f1s = [], [], [], []
    for tr, val in folds:
        mv = MultiVec(vec_char_only)
        Xtr = mv.fit_transform(texts[tr])
        Xval = mv.transform(texts[val])
        # char leg (calibrated for probabilities)
        base = est_char_logreg()
        char_clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        char_clf.fit(Xtr, y[tr])
        p_char = char_clf.predict_proba(Xval)[:, 1]
        p_char_tr = char_clf.predict_proba(Xtr)[:, 1]
        # dense stylometric leg
        gbm = gbm_leg()
        gbm.fit(stylo[tr], y[tr])
        p_gbm = gbm.predict_proba(stylo[val])[:, 1]
        p_gbm_tr = gbm.predict_proba(stylo[tr])[:, 1]
        # blend
        p_blend = W_CHAR * p_char + (1 - W_CHAR) * p_gbm
        p_blend_tr = W_CHAR * p_char_tr + (1 - W_CHAR) * p_gbm_tr
        char_f1.append(macro_f1(y[val], (p_char >= 0.5).astype(int)))
        gbm_f1.append(macro_f1(y[val], (p_gbm >= 0.5).astype(int)))
        blend_f1.append(macro_f1(y[val], (p_blend >= 0.5).astype(int)))
        tr_f1s.append(macro_f1(y[tr], (p_blend_tr >= 0.5).astype(int)))
    return {
        "char": (np.mean(char_f1), np.std(char_f1)),
        "gbm": (np.mean(gbm_f1), np.std(gbm_f1)),
        "blend": (np.mean(blend_f1), np.std(blend_f1),
                  np.mean(tr_f1s), np.mean(tr_f1s) - np.mean(blend_f1)),
    }


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print("TASK 3 — Improved model search with SHIFT-AWARE validation")
    print("=" * 78)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    print(f"train={len(texts)}  test={len(test_texts)}  "
          f"machine={int(y.sum())} ({y.mean():.1%})  human={int((y==0).sum())}")

    # ---- Adversarial validation: is the train/test shift visible in char TF-IDF?
    print("\n" + "-" * 78)
    print("[A] ADVERSARIAL VALIDATION (train=0 vs test=1, char_wb 3-5 TF-IDF)")
    print("-" * 78)
    adv_texts = np.concatenate([texts, test_texts])
    adv_y = np.concatenate([np.zeros(len(texts)), np.ones(len(test_texts))])
    adv_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                              sublinear_tf=True)
    Xadv = adv_vec.fit_transform(adv_texts)
    adv_clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
    adv_oof = cross_val_predict(
        adv_clf, Xadv, adv_y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba", n_jobs=-1)[:, 1]
    adv_auc = roc_auc_score(adv_y, adv_oof)
    print(f"  adversarial AUC = {adv_auc:.4f}")
    print("  (0.5 => shift invisible in char features; >0.7 => strong, locatable shift)")
    # test-likeness of each TRAIN row = adversarial P(test) restricted to train rows
    p_testlike = adv_oof[:len(texts)]
    print(f"  train-row P(test-like): min={p_testlike.min():.3f} "
          f"median={np.median(p_testlike):.3f} max={p_testlike.max():.3f}")

    # ---- Adversarial validation on the STYLOMETRIC feature space too. The
    # ensemble's edge comes from this space, and NEITHER proxy above stresses
    # its drift, so we must measure it directly. High AUC here => the LightGBM
    # leg is extrapolating on the real test and its proxy win is suspect.
    print("\n  Stylometric-space adversarial AUC (honesty anchor for the GBM leg):")
    stylo = build_stylo(texts)
    stylo_test = build_stylo(test_texts)
    for tag, idx in [("full-18", list(range(len(STYLO_NAMES)))),
                     (f"robust-{len(ROBUST_STYLO_IDX)}", ROBUST_STYLO_IDX)]:
        Xs = np.vstack([stylo[:, idx], stylo_test[:, idx]])
        s_oof = cross_val_predict(
            lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                               random_state=SEED, n_jobs=-1, verbose=-1),
            Xs, adv_y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            method="predict_proba", n_jobs=-1)[:, 1]
        print(f"    {tag:10s} stylometric adversarial AUC = {roc_auc_score(adv_y, s_oof):.4f}")
    print("    => we ship the LightGBM leg on the robust subset (lower AUC = less"
          " covariate shift).")

    # ---- Build folds & shared dense features
    print("\n" + "-" * 78)
    print("[B] BUILDING VALIDATION HARNESSES")
    print("-" * 78)
    strat = stratified_folds(y)
    clus, cluster_labels = cluster_folds(texts, y)
    print(f"  stratified: {len(strat)} folds")
    print(f"  cluster-holdout: {len(clus)} folds (topics held out whole); "
          f"cluster sizes={np.bincount(cluster_labels).tolist()}")
    # test-like weighted holdout indices: top-40% most test-like rows as a val
    # set, judged after training on the rest (a directional leaderboard proxy).
    thresh = np.quantile(p_testlike, 0.60)
    testlike_val = np.where(p_testlike >= thresh)[0]
    testlike_tr = np.where(p_testlike < thresh)[0]
    print(f"  test-like holdout: train on {len(testlike_tr)} least-test-like, "
          f"validate on {len(testlike_val)} most-test-like rows")

    print(f"  stylo matrix: {stylo.shape} (ensemble leg uses the "
          f"{len(ROBUST_STYLO_IDX)} robust cols)")

    # ---- Evaluate candidates
    print("\n" + "-" * 78)
    print("[C] CANDIDATE COMPARISON (macro-F1; STRAT=in-dist, CLUS=topic-shift proxy)")
    print("-" * 78)
    print(f"  {'candidate':<10} {'STRAT val':>12} {'CLUS val':>12} "
          f"{'train':>8} {'gap':>7}")

    results = {}

    def run(name, factory, est_factory):
        sv, ss, st, sg = eval_linear(texts, y, strat, factory, est_factory)
        cv, cs, ct, cg = eval_linear(texts, y, clus, factory, est_factory)
        results[name] = dict(strat=(sv, ss), clus=(cv, cs),
                             train=st, gap=sg, clus_gap=cg)
        print(f"  {name:<10} {sv:.4f}+/-{ss:.3f} {cv:.4f}+/-{cs:.3f} "
              f"{st:>8.4f} {sg:>+7.3f}")

    run("BASE", vec_word_char, lambda: est_svc(0.25))
    run("CHAR", vec_char_only, lambda: est_svc(0.5))
    run("CHAR_C1", vec_char_only, lambda: est_svc(1.0))
    run("CHARW", vec_char_prunedword, lambda: est_svc(0.25))

    # Ensemble (reports legs + blend)
    print("\n  [ensemble] soft-vote CHARLR (sparse char) + LGBM (dense stylo):")
    for tag, folds in [("STRAT", strat), ("CLUS", clus)]:
        ens = eval_ensemble(texts, y, stylo, folds)
        print(f"    {tag}: char-leg={ens['char'][0]:.4f}  "
              f"gbm-leg={ens['gbm'][0]:.4f}  BLEND={ens['blend'][0]:.4f}"
              f"+/-{ens['blend'][1]:.3f}  gap={ens['blend'][3]:+.3f}")
        if tag == "CLUS":
            results["ENSMB"] = dict(strat=None, clus=(ens['blend'][0], ens['blend'][1]),
                                    train=ens['blend'][2], gap=None,
                                    clus_gap=ens['blend'][3])

    # ---- Test-like holdout (directional leaderboard proxy) for the finalists
    print("\n" + "-" * 78)
    print("[D] TEST-LIKE HOLDOUT (train on least-test-like, score most-test-like)")
    print("-" * 78)
    tl_folds = [(testlike_tr, testlike_val)]
    for name, factory, ef in [
        ("BASE", vec_word_char, lambda: est_svc(0.25)),
        ("CHAR", vec_char_only, lambda: est_svc(0.5)),
        ("CHARW", vec_char_prunedword, lambda: est_svc(0.25)),
    ]:
        v, s, t, g = eval_linear(texts, y, tl_folds, factory, ef)
        print(f"  {name:<10} test-like val macro-F1 = {v:.4f}  (train {t:.4f}, gap {g:+.3f})")
        results.setdefault(name, {})["testlike"] = v
    ens_tl = eval_ensemble(texts, y, stylo, tl_folds)
    print(f"  {'ENSMB':<10} test-like val macro-F1 = {ens_tl['blend'][0]:.4f}  "
          f"(char-leg={ens_tl['char'][0]:.4f} gbm-leg={ens_tl['gbm'][0]:.4f}, "
          f"gap {ens_tl['blend'][3]:+.3f})")
    results["ENSMB"]["testlike"] = ens_tl['blend'][0]

    # ---- Decision
    print("\n" + "=" * 78)
    print("[E] DECISION")
    print("=" * 78)
    base_clus = results["BASE"]["clus"][0]
    base_strat = results["BASE"]["strat"][0]
    print(f"  Baseline (BASE) under OUR harness: strat={base_strat:.4f}  "
          f"clus={base_clus:.4f}  (prior single-holdout headline was {BASELINE_HEADLINE})")

    # winner on the topic-shift proxy (the leaderboard-relevant lens)
    ranked = sorted(
        [(n, r["clus"][0]) for n, r in results.items() if r.get("clus")],
        key=lambda x: -x[1])
    print("\n  Ranking by CLUSTER-HOLDOUT (topic-shift proxy) macro-F1:")
    for n, sc in ranked:
        print(f"    {n:<10} {sc:.4f}")

    winner, winner_clus = ranked[0]
    base_gap = results["BASE"]["clus_gap"]          # baseline's own cluster gap
    wgap = results[winner].get("clus_gap")
    # Beat baseline on the topic-shift proxy by a non-noise margin ...
    beats = winner_clus > base_clus + 0.01
    # ... and generalize NO WORSE than the baseline does on that same lens.
    # (Absolute gap is uninformative here — every bag-of-features model memorizes
    # train to ~0.99; what matters is the gap RELATIVE to the incumbent.)
    gap_ok = (wgap is None) or (wgap <= base_gap + 1e-9)
    w_tl = results[winner].get("testlike")
    base_tl = results["BASE"].get("testlike")
    also_testlike = (w_tl is not None) and (base_tl is not None) and (w_tl > base_tl)
    print(f"\n  Best on shift proxy: {winner} (clus={winner_clus:.4f} vs BASE {base_clus:.4f}; "
          f"cluster gap={round(wgap,3) if wgap is not None else 'n/a'} vs BASE {base_gap:.3f})")
    print(f"  Confirmed on independent test-like holdout too: "
          f"{'YES' if also_testlike else 'NO'} "
          f"({'n/a' if w_tl is None else round(w_tl,4)} vs BASE "
          f"{'n/a' if base_tl is None else round(base_tl,4)})")

    ship = beats and gap_ok and also_testlike and winner != "BASE"
    if ship:
        print(f"\n  DECISION: {winner} beats the baseline on BOTH shift-aware proxies "
              f"with a gap no worse than the baseline's -> refit on full 20K and "
              f"write prediction file.")
        _refit_and_save(winner, texts, y, stylo, test_texts, test_ids)
    else:
        print("\n  DECISION: no candidate reliably beats the baseline on the "
              "shift-aware proxies with an acceptable train/val gap.")
        print("  => The existing submission (LinearSVC word+char, Kaggle public "
              "0.7299) SHOULD STAND. Not writing a new prediction file.")
        print("  (Refusing to ship a candidate that only wins on in-distribution "
              "CV — that is exactly the trap that caused 0.82->0.73.)")

    print(f"\n  total runtime: {time.time()-t0:.0f}s")
    print("=" * 78)


def _refit_and_save(winner, texts, y, stylo, test_texts, test_ids):
    """Refit the winning candidate on all 20K rows and write predictions."""
    if winner == "CHAR":
        mv = MultiVec(vec_char_only)
        X = mv.fit_transform(texts)
        Xt = mv.transform(test_texts)
        clf = est_svc(0.5).fit(X, y)
        pred = clf.predict(Xt)
    elif winner == "CHARW":
        mv = MultiVec(vec_char_prunedword)
        X = mv.fit_transform(texts)
        Xt = mv.transform(test_texts)
        clf = est_svc(0.25).fit(X, y)
        pred = clf.predict(Xt)
    elif winner == "CHAR_C1":
        mv = MultiVec(vec_char_only)
        X = mv.fit_transform(texts)
        Xt = mv.transform(test_texts)
        clf = est_svc(1.0).fit(X, y)
        pred = clf.predict(Xt)
    elif winner == "ENSMB":
        mv = MultiVec(vec_char_only)
        X = mv.fit_transform(texts)
        Xt = mv.transform(test_texts)
        char_clf = CalibratedClassifierCV(est_char_logreg(), method="sigmoid", cv=3)
        char_clf.fit(X, y)
        gbm = gbm_leg().fit(stylo[:, ROBUST_STYLO_IDX], y)
        stylo_t = build_stylo(test_texts)[:, ROBUST_STYLO_IDX]
        p = (W_CHAR * char_clf.predict_proba(Xt)[:, 1]
             + (1 - W_CHAR) * gbm.predict_proba(stylo_t)[:, 1])
        pred = (p >= 0.5).astype(int)
    else:
        raise ValueError(f"no refit path for {winner}")
    out = OUT_DIR / "Task3_Improved_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred.astype(int)}).to_csv(out, index=False)
    print(f"  wrote {out}  (rows={len(pred)}  machine={int(pred.sum())} "
          f"human={int((pred==0).sum())})")


if __name__ == "__main__":
    main()
