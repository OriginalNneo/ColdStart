"""
GenAI Text Detection — Iteration 2: wider search, cross-validated.
====================================================================
Builds on ML_Project_NaiveBayes_Scratch.py (still zero scikit-learn).
Iteration 1 found: style-only 0.5736, word-freq-only 0.6433, hybrid 0.6465
(single 90/10 holdout, unigrams only, mixed vocab).

This round asks four new questions, each because iteration 1 left it
unexplored:
  A. Is a single 2000-sample holdout reliable enough to rank hyperparameters
     that differ by ~0.01 F1? -> switch to k-fold cross-validation.
  B. Iteration 1's vocab only used unigrams and never separated content
     words from function words. Function-word frequency ("stylometric"
     signal — how often someone uses "the", "very", "just") is a classic
     authorship-attribution feature that is TOPIC-INDEPENDENT, which
     matters given the known train/test distribution shift. Do bigrams or
     a stopword-only/stopword-excluded vocab beat the mixed-unigram vocab?
  C. Iteration 1's fine sweep only tried vocab_size >= 200. Is the true
     optimum even smaller?
  D. The Hybrid model summed style + word-frequency evidence with implicit
     equal weight. Is 50/50 actually optimal, or does one family deserve
     more say?
"""

import time
import numpy as np
import pandas as pd

from ML_Project_NaiveBayes_Scratch import (
    SEED, DATA_DIR, OUTPUT_DIR, STOPWORDS,
    build_feature_matrix, FEATURE_NAMES,
    stratified_split, stratified_kfold, macro_f1,
    GaussianNaiveBayesScratch, MultinomialNaiveBayesScratch, HybridNaiveBayesScratch,
    feature_separation_scores, select_decorrelated_features,
    build_vocabulary, build_count_matrix,
)


def cv_score(build_and_eval_fn, folds, **kwargs):
    """Run build_and_eval_fn(tr_idx, val_idx, **kwargs) -> (train_f1, val_f1)
    over every fold, return (mean_val_f1, std_val_f1, mean_train_f1)."""
    val_f1s, tr_f1s = [], []
    for tr_idx, val_idx in folds:
        tr_f1, val_f1 = build_and_eval_fn(tr_idx, val_idx, **kwargs)
        tr_f1s.append(tr_f1)
        val_f1s.append(val_f1)
    return float(np.mean(val_f1s)), float(np.std(val_f1s)), float(np.mean(tr_f1s))


def main():
    print("=" * 70)
    print("ITERATION 2 — cross-validated search over the from-scratch Bayesian pipeline")
    print("=" * 70)

    print("\n[Load] reading data + extracting style features...")
    train_text = pd.read_csv(DATA_DIR / "train.csv")
    test_text = pd.read_csv(DATA_DIR / "test.csv")
    y_all = train_text["label"].to_numpy(dtype=int)
    texts_all = train_text["text"].values
    test_ids = test_text["id"].to_numpy()

    X_style_all = build_feature_matrix(texts_all)
    X_style_test = build_feature_matrix(test_text["text"].values)

    K = 5
    folds = stratified_kfold(y_all, k=K, seed=SEED)
    print(f"  using {K}-fold stratified CV for every comparison below")

    # ------------------------------------------------------------------
    print("\n[A] Re-tuning style model with CV (was single-split in iteration 1)...")
    # Decorrelation is a structural/unsupervised-ish choice (based on X's own
    # correlation structure); computed once on the full train set rather than
    # per-fold to keep the search tractable — the real unknowns below are B-D.
    decorr_idx = select_decorrelated_features(X_style_all, y_all, max_corr=0.6)
    decorr_names = [FEATURE_NAMES[i] for i in decorr_idx]
    X_style_all_d = X_style_all[:, decorr_idx]
    X_style_test_d = X_style_test[:, decorr_idx]
    print(f"  decorrelated feature set ({len(decorr_names)}): {decorr_names}")

    def eval_style(tr_idx, val_idx, vs):
        model = GaussianNaiveBayesScratch(var_smoothing=vs).fit(X_style_all_d[tr_idx], y_all[tr_idx])
        tr_f1 = macro_f1(y_all[tr_idx], model.predict(X_style_all_d[tr_idx]))
        val_f1 = macro_f1(y_all[val_idx], model.predict(X_style_all_d[val_idx]))
        return tr_f1, val_f1

    best_vs, best_vs_f1 = None, -1.0
    for vs in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
        mean_val, std_val, mean_tr = cv_score(eval_style, folds, vs=vs)
        flag = "  <-- best" if mean_val > best_vs_f1 else ""
        print(f"  var_smoothing={vs:<8} CV train F1={mean_tr:.4f}  "
              f"CV val F1={mean_val:.4f} +/- {std_val:.4f}{flag}")
        if mean_val > best_vs_f1:
            best_vs, best_vs_f1 = vs, mean_val
    print(f"  Selected var_smoothing={best_vs} (CV val macro F1={best_vs_f1:.4f})")

    # ------------------------------------------------------------------
    print("\n[B] Comparing vocabulary VARIANTS (ngram range x word filter)...")
    print("    Fixed at vocab_size=300, alpha=0.5 for a fair head-to-head.")

    def content_filter(w):
        return w not in STOPWORDS

    def function_filter(w):
        return w in STOPWORDS

    variants = [
        ("unigram_mixed",     (1, 1), None),
        ("unigram_content",   (1, 1), content_filter),
        ("unigram_function",  (1, 1), function_filter),
        ("uni+bigram_mixed",  (1, 2), None),
        ("uni+bigram_content",(1, 2), content_filter),
        ("uni+bigram_function",(1, 2), function_filter),
    ]

    def eval_variant(tr_idx, val_idx, ngram_range, word_filter, vocab_size, alpha):
        vocab = build_vocabulary(texts_all[tr_idx], vocab_size, ngram_range, word_filter)
        C_tr = build_count_matrix(texts_all[tr_idx], vocab, ngram_range, word_filter)
        C_val = build_count_matrix(texts_all[val_idx], vocab, ngram_range, word_filter)
        model = MultinomialNaiveBayesScratch(alpha=alpha).fit(C_tr, y_all[tr_idx])
        tr_f1 = macro_f1(y_all[tr_idx], model.predict(C_tr))
        val_f1 = macro_f1(y_all[val_idx], model.predict(C_val))
        return tr_f1, val_f1

    # 3-fold here (variant screening is coarse; fine-tune the winner with 5-fold next).
    screen_folds = stratified_kfold(y_all, k=3, seed=SEED)
    variant_results = {}
    t0 = time.time()
    for name, ngram_range, word_filter in variants:
        mean_val, std_val, mean_tr = cv_score(
            eval_variant, screen_folds,
            ngram_range=ngram_range, word_filter=word_filter, vocab_size=300, alpha=0.5
        )
        variant_results[name] = (ngram_range, word_filter, mean_val)
        print(f"  {name:22s} CV train F1={mean_tr:.4f}  CV val F1={mean_val:.4f} +/- {std_val:.4f}")
    print(f"  (variant screen took {time.time()-t0:.0f}s)")

    best_variant_name = max(variant_results, key=lambda k: variant_results[k][2])
    best_ngram_range, best_word_filter, _ = variant_results[best_variant_name]
    print(f"\n  Winning variant: {best_variant_name}")

    # ------------------------------------------------------------------
    print(f"\n[C] Fine vocab_size x alpha sweep within '{best_variant_name}' (5-fold CV)...")
    MAX_VOCAB = 1500
    best_vocab_size, best_alpha, best_mnb_f1 = None, None, -1.0
    vocab_size_grid = [30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500]
    alpha_grid = [0.1, 0.3, 0.5, 1.0, 2.0]

    # Build once per fold at MAX_VOCAB, slice for every smaller vocab_size —
    # avoids rebuilding the count matrix vocab_size_grid-many times per fold.
    per_fold_matrices = []
    for tr_idx, val_idx in folds:
        vocab = build_vocabulary(texts_all[tr_idx], MAX_VOCAB, best_ngram_range, best_word_filter)
        C_tr = build_count_matrix(texts_all[tr_idx], vocab, best_ngram_range, best_word_filter)
        C_val = build_count_matrix(texts_all[val_idx], vocab, best_ngram_range, best_word_filter)
        per_fold_matrices.append((C_tr, C_val, tr_idx, val_idx))
    print(f"  built {K} fold count-matrices at max vocab={MAX_VOCAB}")

    for vocab_size in vocab_size_grid:
        for alpha in alpha_grid:
            val_f1s, tr_f1s = [], []
            for C_tr, C_val, tr_idx, val_idx in per_fold_matrices:
                cols = slice(0, vocab_size)
                model = MultinomialNaiveBayesScratch(alpha=alpha).fit(C_tr[:, cols], y_all[tr_idx])
                tr_f1s.append(macro_f1(y_all[tr_idx], model.predict(C_tr[:, cols])))
                val_f1s.append(macro_f1(y_all[val_idx], model.predict(C_val[:, cols])))
            mean_val, mean_tr = float(np.mean(val_f1s)), float(np.mean(tr_f1s))
            gap = mean_tr - mean_val
            flag = "  <-- best" if mean_val > best_mnb_f1 else ""
            print(f"  vocab={vocab_size:5d} alpha={alpha:<4} CV train F1={mean_tr:.4f} "
                  f"CV val F1={mean_val:.4f} gap={gap:+.4f}{flag}")
            if mean_val > best_mnb_f1:
                best_vocab_size, best_alpha, best_mnb_f1 = vocab_size, alpha, mean_val

    print(f"\n  Selected vocab_size={best_vocab_size}, alpha={best_alpha} "
          f"(CV val macro F1={best_mnb_f1:.4f})")

    # ------------------------------------------------------------------
    print(f"\n[D] Tuning Hybrid style_weight (5-fold CV)...")
    weight_grid = np.arange(0.0, 1.01, 0.1)
    weight_results = {}
    for w in weight_grid:
        val_f1s = []
        for (C_tr, C_val, tr_idx, val_idx) in per_fold_matrices:
            cols = slice(0, best_vocab_size)
            hybrid = HybridNaiveBayesScratch(var_smoothing=best_vs, alpha=best_alpha, style_weight=w)
            hybrid.fit(X_style_all_d[tr_idx], C_tr[:, cols], y_all[tr_idx])
            y_pred = hybrid.predict(X_style_all_d[val_idx], C_val[:, cols])
            val_f1s.append(macro_f1(y_all[val_idx], y_pred))
        mean_val = float(np.mean(val_f1s))
        weight_results[round(w, 2)] = mean_val
        print(f"  style_weight={w:.1f}  CV val F1={mean_val:.4f}")

    best_weight = max(weight_results, key=weight_results.get)
    best_hybrid_f1 = weight_results[best_weight]
    print(f"\n  Selected style_weight={best_weight} (CV val macro F1={best_hybrid_f1:.4f})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BAKE-OFF (all CV-scored, 5-fold except variant screen which was 3-fold)")
    print("=" * 70)
    style_only_f1 = best_vs_f1
    word_freq_only_f1 = best_mnb_f1
    candidates = {
        "style_only": style_only_f1,
        f"word_freq_only ({best_variant_name})": word_freq_only_f1,
        "hybrid": best_hybrid_f1,
    }
    for name, f1 in sorted(candidates.items(), key=lambda t: -t[1]):
        print(f"  {name:38s} CV val macro F1 = {f1:.4f}")
    winner = max(candidates, key=candidates.get)
    print(f"\n  Overall winner: {winner} ({candidates[winner]:.4f})")

    print("\n  Comparison to iteration 1 (single 90/10 holdout, not CV):")
    print("    style_only:      0.5736  ->  now (CV): {:.4f}".format(style_only_f1))
    print("    word_freq_only:  0.6433  ->  now (CV): {:.4f}".format(word_freq_only_f1))
    print("    hybrid:          0.6465  ->  now (CV): {:.4f}".format(best_hybrid_f1))

    # ------------------------------------------------------------------
    print("\n[Final] Refitting winner on FULL 20K training set, tuning threshold on a held-out slice...")
    # Held-out slice purely for threshold selection (never used for any other tuning above).
    tr_idx, val_idx = stratified_split(y_all, val_frac=0.1, seed=SEED + 1)

    vocab_final = build_vocabulary(texts_all[tr_idx], best_vocab_size, best_ngram_range, best_word_filter)
    C_tr = build_count_matrix(texts_all[tr_idx], vocab_final, best_ngram_range, best_word_filter)
    C_val = build_count_matrix(texts_all[val_idx], vocab_final, best_ngram_range, best_word_filter)

    if winner == "style_only":
        model = GaussianNaiveBayesScratch(var_smoothing=best_vs).fit(X_style_all_d[tr_idx], y_all[tr_idx])
        p_val = model.predict_proba(X_style_all_d[val_idx])[:, list(model.classes_).index(1)]
    elif winner.startswith("word_freq_only"):
        model = MultinomialNaiveBayesScratch(alpha=best_alpha).fit(C_tr, y_all[tr_idx])
        p_val = model.predict_proba(C_val)[:, list(model.classes_).index(1)]
    else:
        model = HybridNaiveBayesScratch(var_smoothing=best_vs, alpha=best_alpha, style_weight=best_weight)
        model.fit(X_style_all_d[tr_idx], C_tr, y_all[tr_idx])
        p_val = model.predict_proba(X_style_all_d[val_idx], C_val)[:, list(model.classes_).index(1)]

    best_thresh, best_thresh_f1 = 0.5, -1.0
    for thresh in np.arange(0.25, 0.76, 0.01):
        f1 = macro_f1(y_all[val_idx], (p_val >= thresh).astype(int))
        if f1 > best_thresh_f1:
            best_thresh, best_thresh_f1 = thresh, f1
    print(f"  threshold={best_thresh:.2f} -> held-out macro F1={best_thresh_f1:.4f}")

    print("\n  Refitting on ALL 20K rows for the submission...")
    vocab_all = build_vocabulary(texts_all, best_vocab_size, best_ngram_range, best_word_filter)
    C_all = build_count_matrix(texts_all, vocab_all, best_ngram_range, best_word_filter)
    C_test = build_count_matrix(test_text["text"].values, vocab_all, best_ngram_range, best_word_filter)

    if winner == "style_only":
        final_model = GaussianNaiveBayesScratch(var_smoothing=best_vs).fit(X_style_all_d, y_all)
        y_test_pred = final_model.predict(X_style_test_d, threshold=best_thresh)
    elif winner.startswith("word_freq_only"):
        final_model = MultinomialNaiveBayesScratch(alpha=best_alpha).fit(C_all, y_all)
        C_test_full = build_count_matrix(test_text["text"].values, vocab_all, best_ngram_range, best_word_filter)
        y_test_pred = final_model.predict(C_test_full, threshold=best_thresh)
    else:
        final_model = HybridNaiveBayesScratch(var_smoothing=best_vs, alpha=best_alpha, style_weight=best_weight)
        final_model.fit(X_style_all_d, C_all, y_all)
        y_test_pred = final_model.predict(X_style_test_d, C_test, threshold=best_thresh)

    out_path = OUTPUT_DIR / "NaiveBayes_Scratch_v2_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": y_test_pred}).to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(y_test_pred)} rows, "
          f"machine={y_test_pred.sum()}, human={(y_test_pred==0).sum()})")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Winner: {winner}")
    print(f"Style features ({len(decorr_names)}): {decorr_names}")
    print(f"Word-freq: variant={best_variant_name}, vocab_size={best_vocab_size}, alpha={best_alpha}")
    if winner == "hybrid":
        print(f"Hybrid style_weight={best_weight}")
    print(f"var_smoothing={best_vs}, decision_threshold={best_thresh:.2f}")
    print(f"CV validation macro F1: {candidates[winner]:.4f}")
    print(f"Held-out (post-selection) macro F1: {best_thresh_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
