"""
Track 3: word/char BLOCK REWEIGHTING (classical, NO deep learning).

Idea: char block leans on STYLE (topic-robust); word block leans on TOPIC.
L2-normalization inside each TfidfVectorizer scales each block to unit rows,
so a scalar multiplier on one block genuinely shifts the SVM's tradeoff between
the two representations. Sweep the multiplier both directions.

Wraps TfidfVectorizer in a tiny sklearn-style transformer that multiplies its
transform() output by scalar alpha (fit() just fits the inner vectorizer).

Capped config (per memory budget):
  word: word(1,3), min_df=2, sublinear_tf=True
  char: char_wb(2,6), min_df=3, max_features=300000, sublinear_tf=True
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scratch_lens import load_data, get_folds, eval_rep, ANCHOR


class ScaledTfidf:
    """Wrap a TfidfVectorizer; multiply transform() output by scalar alpha."""
    def __init__(self, alpha=1.0, **tfidf_kwargs):
        self.alpha = float(alpha)
        self.vec = TfidfVectorizer(**tfidf_kwargs)

    def fit(self, X, y=None):
        self.vec.fit(X)
        return self

    def transform(self, X):
        M = self.vec.transform(X)
        if self.alpha == 1.0:
            return M
        return M.multiply(self.alpha).tocsr()


def word_vec(alpha=1.0):
    return ScaledTfidf(alpha, analyzer="word", ngram_range=(1, 3),
                       min_df=2, sublinear_tf=True)


def char_vec(alpha=1.0):
    return ScaledTfidf(alpha, analyzer="char_wb", ngram_range=(2, 6),
                       min_df=3, max_features=300000, sublinear_tf=True)


def make_factory(word_alpha, char_alpha):
    """vec_factory scaling word block by word_alpha, char block by char_alpha."""
    def factory():
        return [word_vec(word_alpha), char_vec(char_alpha)]
    return factory


def make_single(which):
    if which == "word":
        return lambda: [word_vec(1.0)]
    return lambda: [char_vec(1.0)]


ALPHAS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]


def main():
    t0 = time.time()
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    print(f"train={len(texts)} foldsA={len(foldsA)} foldsB={len(foldsB)}", flush=True)
    print(f"ANCHOR LensA={ANCHOR['A']:.4f} LensB={ANCHOR['B']:.4f}\n", flush=True)

    # --- reference points: char-only, word-only ---
    for name, fac in [("word-only", make_single("word")),
                      ("char-only", make_single("char"))]:
        a, _ = eval_rep(fac, texts, Y, foldsA)
        b, _ = eval_rep(fac, texts, Y, foldsB)
        print(f"[ref] {name:10s} LensA={a:.4f} LensB={b:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print()

    # --- alpha=1.0 anchor reproduction (capped wideB) ---
    a1, a1f = eval_rep(make_factory(1.0, 1.0), texts, Y, foldsA)
    b1, b1f = eval_rep(make_factory(1.0, 1.0), texts, Y, foldsB)
    print(f"[anchor repro alpha=1.0] LensA={a1:.4f} {a1f}\n"
          f"                         LensB={b1:.4f} {b1f}\n", flush=True)

    results = {}  # (direction, alpha) -> (A, B)

    def sweep(direction):
        print(f"=== sweep: scale {direction.upper()} block by alpha "
              f"(other block=1.0) ===", flush=True)
        print(f"{'alpha':>6} | {'LensA':>7} {'dA':>8} | {'LensB':>7} {'dB':>8} | PASS",
              flush=True)
        for al in ALPHAS:
            if direction == "char":
                fac = make_factory(1.0, al)
            else:
                fac = make_factory(al, 1.0)
            a, _ = eval_rep(fac, texts, Y, foldsA)
            b, _ = eval_rep(fac, texts, Y, foldsB)
            dA = a - ANCHOR["A"]
            dB = b - ANCHOR["B"]
            passed = (a > ANCHOR["A"]) and (b > ANCHOR["B"])
            results[(direction, al)] = (a, b)
            print(f"{al:>6.1f} | {a:>7.4f} {dA:>+8.4f} | {b:>7.4f} {dB:>+8.4f} | "
                  f"{'PASS' if passed else 'fail'}  ({time.time()-t0:.0f}s)",
                  flush=True)
        print()

    sweep("char")
    sweep("word")

    # --- pick best PASSing candidate (min of the two deltas maximized) ---
    best = None
    for (direction, al), (a, b) in results.items():
        if al == 1.0:
            continue
        if a > ANCHOR["A"] and b > ANCHOR["B"]:
            score = min(a - ANCHOR["A"], b - ANCHOR["B"])
            if best is None or score > best[0]:
                best = (score, direction, al, a, b)

    if best is None:
        print("NO alpha PASSES both lenses. Null result. No prediction written.",
              flush=True)
        return

    _, direction, al, a, b = best
    print(f"BEST PASS: scale {direction} block alpha={al}  "
          f"LensA={a:.4f} (+{a-ANCHOR['A']:.4f}) LensB={b:.4f} (+{b-ANCHOR['B']:.4f})",
          flush=True)

    # --- refit on all 20k, predict test ---
    if direction == "char":
        vecs = [word_vec(1.0), char_vec(al)]
    else:
        vecs = [word_vec(al), char_vec(1.0)]
    from sklearn.svm import LinearSVC
    Xtr = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
    Xte = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
    clf = LinearSVC(C=0.25, class_weight="balanced", random_state=42)
    clf.fit(Xtr, Y)
    preds = clf.predict(Xte)
    import pandas as pd
    pd.DataFrame({"id": test_ids, "label": preds}).to_csv(
        "scratch_agent3_pred.csv", index=False)
    print(f"wrote scratch_agent3_pred.csv  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
