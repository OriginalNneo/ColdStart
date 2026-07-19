"""
Task 3 — "WORK BACKWARDS" experiments: use the test set itself (its text and
the model's own confident predictions on it) to adapt the model to the test
distribution, instead of only optimizing forwards on train.

Motivation: current LB 0.72990 (LinearSVC word+char TF-IDF), 1st place is
0.78326. Known train->test topic shift (char-ngram adversarial AUC 0.81;
val 0.82 -> LB 0.73). Levers below directly attack that shift and were NOT
tried by the earlier shift-aware round (which changed the MODEL, not the
TRAINING DATA):

  VOCAB   transductive representation: fit TF-IDF vocab/idf on train+test
          text (unlabeled test text is legitimately available), classifier
          still trained on labeled train rows only.
  PSEUDO  self-training: fit on train, pseudo-label test rows where
          |decision margin| >= m, refit with those rows added at weight w,
          optionally iterate. The model "works backwards" from its own
          confident test predictions.
  IW      importance weighting: weight each training row by how test-like it
          is (out-of-fold adversarial char-ngram probabilities), so training
          focuses on the region overlapping the test distribution.

Validation: cluster-holdout topic-shift proxy from Task3_Improved_Model
(it scored the baseline 0.7383 vs real LB 0.7299 — trustworthy for BASE, but
it OVERRATED the ENSMB candidate once, so any winner here still needs a real
submission to confirm). Because fold-val labels are known, we also report
pseudo-label PRECISION — whether working backwards feeds the model truth.

Run: .venv/bin/python Task3_PseudoLabel.py
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, SEED, MultiVec, cluster_folds, est_svc, macro_f1,
    vec_word_char,
)

C_BASE = 0.25          # the proven LB-0.7299 config
PSEUDO_GRID = [        # (margin, weight, iterations)
    (1.0, 1.0, 1),
    (0.5, 0.5, 1),
    (1.0, 1.0, 2),
]
IW_CLIP = (0.25, 4.0)  # clip adversarial importance weights


def fit_predict(Xtr, ytr, Xval, sample_weight=None):
    clf = est_svc(C_BASE)
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    return clf, clf.predict(Xval)


def pseudo_label_fit(Xtr, ytr, Xun, margin, weight, iters):
    """Self-training. Returns (final clf, stats about the pseudo set)."""
    clf, _ = fit_predict(Xtr, ytr, Xun[:0] if Xun.shape[0] == 0 else Xun)
    n_pseudo, pl = 0, None
    for _ in range(iters):
        scores = clf.decision_function(Xun)
        conf = np.abs(scores) >= margin
        n_pseudo = int(conf.sum())
        if n_pseudo == 0:
            break
        pl = (scores[conf] > 0).astype(int)
        Xaug = sparse.vstack([Xtr, Xun[conf]]).tocsr()
        yaug = np.concatenate([ytr, pl])
        sw = np.concatenate([np.ones(Xtr.shape[0]),
                             np.full(n_pseudo, weight)])
        clf = est_svc(C_BASE)
        clf.fit(Xaug, yaug, sample_weight=sw)
    return clf, conf if n_pseudo else None, pl


def adversarial_train_weights(texts_tr, texts_un):
    """Out-of-fold P(test-like) for each TRAIN row via char-ngram adversarial
    validation; converted to clipped importance weights, normalized to mean 1."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                          max_features=200000, sublinear_tf=True)
    Xa = vec.fit_transform(np.concatenate([texts_tr, texts_un]))
    d = np.r_[np.zeros(len(texts_tr)), np.ones(len(texts_un))]
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    p = cross_val_predict(lr, Xa, d, cv=3, method="predict_proba")[:, 1]
    p_tr = np.clip(p[:len(texts_tr)], 1e-3, 1 - 1e-3)
    w = np.clip(p_tr / (1 - p_tr), *IW_CLIP)
    return w * (len(w) / w.sum())


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy()
    test_texts = test["text"].astype(str).to_numpy()
    print(f"train {len(texts)} | test {len(test_texts)}")

    folds, _ = cluster_folds(texts, y)
    print(f"cluster-holdout folds: {len(folds)}\n")

    names = (["BASE", "VOCAB"] +
             [f"PSEUDO(m={m},w={w},it={it})" for m, w, it in PSEUDO_GRID] +
             ["IW"])
    scores = {n: [] for n in names}
    pseudo_diag = {n: [] for n in names if n.startswith("PSEUDO")}

    for k, (tr, val) in enumerate(folds):
        ttr, tval, ytr, yval = texts[tr], texts[val], y[tr], y[val]
        print(f"fold {k+1}/{len(folds)}  (train {len(tr)}, heldout {len(val)})")

        # BASE: train-only representation and training rows
        mv = MultiVec(vec_word_char)
        Xtr = mv.fit_transform(ttr)
        _, pred = fit_predict(Xtr, ytr, mv.transform(tval))
        scores["BASE"].append(macro_f1(yval, pred))

        # transductive representation (vocab/idf sees held-out TEXT, no labels)
        mvt = MultiVec(vec_word_char)
        Xall = mvt.fit_transform(np.concatenate([ttr, tval]))
        Xtr_t, Xval_t = Xall[:len(tr)], Xall[len(tr):]

        _, pred = fit_predict(Xtr_t, ytr, Xval_t)
        scores["VOCAB"].append(macro_f1(yval, pred))

        for m, w, it in PSEUDO_GRID:
            name = f"PSEUDO(m={m},w={w},it={it})"
            clf, conf, pl = pseudo_label_fit(Xtr_t, ytr, Xval_t, m, w, it)
            scores[name].append(macro_f1(yval, clf.predict(Xval_t)))
            if conf is not None:
                prec = float((pl == yval[conf]).mean())
                cov = float(conf.mean())
                pseudo_diag[name].append((cov, prec))

        w_tr = adversarial_train_weights(ttr, tval)
        _, pred = fit_predict(Xtr_t, ytr, Xval_t, sample_weight=w_tr)
        scores["IW"].append(macro_f1(yval, pred))

        print("   " + "  ".join(f"{n}={scores[n][-1]:.4f}" for n in names))

    print("\n===== cluster-holdout means (proxy for LB) =====")
    base_mean = float(np.mean(scores["BASE"]))
    for n in names:
        s = np.array(scores[n])
        extra = ""
        if n in pseudo_diag and pseudo_diag[n]:
            cov = np.mean([d[0] for d in pseudo_diag[n]])
            prec = np.mean([d[1] for d in pseudo_diag[n]])
            extra = f"  [pseudo coverage {cov:.0%}, precision {prec:.4f}]"
        print(f"  {n:24s} {s.mean():.4f} ± {s.std():.4f} "
              f"(Δ vs BASE {s.mean()-base_mean:+.4f}){extra}")

    winner = max(names, key=lambda n: np.mean(scores[n]))
    win_mean = float(np.mean(scores[winner]))
    print(f"\nWinner: {winner}  ({win_mean:.4f}, Δ{win_mean-base_mean:+.4f})")

    # ---- refit winner recipe on FULL train + real test, save prediction ----
    print("\nRefitting winner on full train + real test text...")
    mvt = MultiVec(vec_word_char)
    Xall = mvt.fit_transform(np.concatenate([texts, test_texts]))
    Xtr_f, Xte_f = Xall[:len(texts)], Xall[len(texts):]

    if winner == "BASE":
        mv = MultiVec(vec_word_char)
        clf, test_pred = fit_predict(mv.fit_transform(texts), y,
                                     mv.transform(test_texts))
    elif winner == "VOCAB":
        clf, test_pred = fit_predict(Xtr_f, y, Xte_f)
    elif winner == "IW":
        w_tr = adversarial_train_weights(texts, test_texts)
        clf, test_pred = fit_predict(Xtr_f, y, Xte_f, sample_weight=w_tr)
    else:
        m, w, it = PSEUDO_GRID[names.index(winner) - 2]
        clf, _, _ = pseudo_label_fit(Xtr_f, y, Xte_f, m, w, it)
        test_pred = clf.predict(Xte_f)

    out = pd.DataFrame({"id": test["id"], "label": test_pred.astype(int)})
    path = OUT_DIR / "Task3_PseudoLabel_Prediction.csv"
    out.to_csv(path, index=False)
    print(f"WROTE {path} ({len(out)} rows, machine={int(out['label'].sum())}, "
          f"human={int((out['label']==0).sum())})")
    print(f"\nTotal runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
