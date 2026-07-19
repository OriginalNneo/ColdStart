"""
Track 4: estimator geometry swap on the FIXED capped-wideB representation.
Keep rep fixed; find an estimator whose regularization transfers better under
topic shift than LinearSVC(C=0.25, balanced). Two-lens gated.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import (LogisticRegression, SGDClassifier,
                                   PassiveAggressiveClassifier, RidgeClassifier)
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from scratch_lens import load_data, get_folds, eval_rep, macro_f1, ANCHOR

SEED = 42
t0 = time.time()


def capped_wideB_vecs():
    """Memory-capped wideB rep (per task spec)."""
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                        max_features=300000, sublinear_tf=True),
    ]


texts, Y, test_texts, test_ids = load_data()
foldsA, foldsB = get_folds()
print(f"train={len(texts)} pos={Y.mean():.4f} foldsA={len(foldsA)} foldsB={len(foldsB)} "
      f"anchor A={ANCHOR['A']} B={ANCHOR['B']} ({time.time()-t0:.0f}s)", flush=True)


def run(name, est_factory):
    a, af = eval_rep(capped_wideB_vecs, texts, Y, foldsA, est_factory=est_factory)
    b, bf = eval_rep(capped_wideB_vecs, texts, Y, foldsB, est_factory=est_factory)
    dA, dB = a - ANCHOR["A"], b - ANCHOR["B"]
    passed = (a > ANCHOR["A"]) and (b > ANCHOR["B"])
    print(f"{name:42s} A={a:.4f}({dA:+.4f}) B={b:.4f}({dB:+.4f}) "
          f"{'PASS' if passed else 'fail'}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(name=name, A=a, B=b, dA=dA, dB=dB, passed=passed)


results = []

# 0) Reproduce anchor exactly on the capped rep
results.append(run("ANCHOR LinearSVC(C=0.25,bal)",
                   lambda: LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)))

# --- LinearSVC geometry variants
for C in (0.15, 0.20, 0.35):
    results.append(run(f"LinearSVC C={C} sq_hinge dual",
                       lambda C=C: LinearSVC(C=C, class_weight="balanced", random_state=SEED)))
results.append(run("LinearSVC C=0.25 hinge dual",
                   lambda: LinearSVC(C=0.25, loss="hinge", class_weight="balanced", random_state=SEED)))

# --- LogisticRegression liblinear
for C in (0.1, 0.3, 1, 3):
    results.append(run(f"LogReg liblinear C={C}",
                       lambda C=C: LogisticRegression(C=C, solver="liblinear",
                                                      class_weight="balanced", max_iter=2000)))
# --- LogisticRegression saga
for C in (0.3, 1):
    results.append(run(f"LogReg saga C={C}",
                       lambda C=C: LogisticRegression(C=C, solver="saga",
                                                      class_weight="balanced", max_iter=3000)))

# --- SGDClassifier
for loss in ("hinge", "modified_huber", "log_loss"):
    for alpha in (1e-5, 1e-4, 1e-3):
        results.append(run(f"SGD {loss} a={alpha:g}",
                           lambda loss=loss, alpha=alpha: SGDClassifier(
                               loss=loss, alpha=alpha, class_weight="balanced",
                               max_iter=2000, tol=1e-4, random_state=SEED)))

# --- PassiveAggressive
for C in (0.01, 0.1, 1):
    results.append(run(f"PassiveAggressive C={C}",
                       lambda C=C: PassiveAggressiveClassifier(
                           C=C, class_weight="balanced", max_iter=2000,
                           tol=1e-4, random_state=SEED)))

# --- RidgeClassifier
for alpha in (1, 10, 100):
    results.append(run(f"Ridge alpha={alpha}",
                       lambda alpha=alpha: RidgeClassifier(alpha=alpha,
                                                           class_weight="balanced")))

# --- Naive Bayes (nonneg TF-IDF)
for alpha in (0.1, 0.3, 1.0):
    results.append(run(f"ComplementNB alpha={alpha}",
                       lambda alpha=alpha: ComplementNB(alpha=alpha)))
for alpha in (0.1, 0.3, 1.0):
    results.append(run(f"MultinomialNB alpha={alpha}",
                       lambda alpha=alpha: MultinomialNB(alpha=alpha)))

# --- summary
print("\n=== PASSERS (both lenses) ===", flush=True)
passers = [r for r in results if r["passed"] and not r["name"].startswith("ANCHOR")]
for r in sorted(passers, key=lambda r: -(r["dA"] + r["dB"])):
    print(f"  {r['name']:40s} A={r['A']:.4f}({r['dA']:+.4f}) B={r['B']:.4f}({r['dB']:+.4f})", flush=True)
if not passers:
    print("  (none)", flush=True)

# Best single-model passer -> refit all 20k
best = max(passers, key=lambda r: (r["dA"] + r["dB"]), default=None)
print(f"\nbest single passer: {best['name'] if best else None}", flush=True)
np.save("scratch_agent4_results.npy", np.array(results, dtype=object))
print(f"done ({time.time()-t0:.0f}s)", flush=True)
