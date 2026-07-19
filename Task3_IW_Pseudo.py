"""
Task 3 — follow-up: combine the two winning test-adaptation levers.

Task3_PseudoLabel.py found (cluster-holdout proxy, 5 folds):
  BASE 0.7383 | VOCAB +0.0061 | PSEUDO(m=1.0) +0.0061 | IW +0.0140 (all folds)

Untested: IW + PSEUDO stacked — importance-weight the train rows, then also
add the model's confident pseudo-labeled held-out rows (98% precise per the
diagnostic) and refit. This script evaluates that combo on the same folds
and compares against the recorded IW-alone per-fold scores.

Run: .venv/bin/python Task3_IW_Pseudo.py
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, MultiVec, cluster_folds, est_svc, macro_f1,
    vec_word_char,
)
from Task3_PseudoLabel import C_BASE, adversarial_train_weights

# per-fold IW-alone scores from the Task3_PseudoLabel.py run (deterministic:
# same SEED, same folds), to compare against without re-running.
IW_ALONE = [0.7399, 0.7991, 0.7289, 0.7887, 0.7049]
MARGIN, PSEUDO_W = 1.0, 1.0


def iw_pseudo_fit(Xtr, ytr, w_tr, Xun):
    """IW fit -> confident pseudo-labels -> refit with both weight sets."""
    clf = est_svc(C_BASE)
    clf.fit(Xtr, ytr, sample_weight=w_tr)
    scores = clf.decision_function(Xun)
    conf = np.abs(scores) >= MARGIN
    if conf.sum() == 0:
        return clf, 0
    pl = (scores[conf] > 0).astype(int)
    Xaug = sparse.vstack([Xtr, Xun[conf]]).tocsr()
    yaug = np.concatenate([ytr, pl])
    sw = np.concatenate([w_tr, np.full(int(conf.sum()), PSEUDO_W)])
    clf2 = est_svc(C_BASE)
    clf2.fit(Xaug, yaug, sample_weight=sw)
    return clf2, int(conf.sum())


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy()
    test_texts = test["text"].astype(str).to_numpy()

    folds, _ = cluster_folds(texts, y)
    combo = []
    for k, (tr, val) in enumerate(folds):
        ttr, tval, ytr, yval = texts[tr], texts[val], y[tr], y[val]
        mvt = MultiVec(vec_word_char)
        Xall = mvt.fit_transform(np.concatenate([ttr, tval]))
        Xtr_t, Xval_t = Xall[:len(tr)], Xall[len(tr):]
        w_tr = adversarial_train_weights(ttr, tval)
        clf, n_pl = iw_pseudo_fit(Xtr_t, ytr, w_tr, Xval_t)
        f1 = macro_f1(yval, clf.predict(Xval_t))
        combo.append(f1)
        print(f"fold {k+1}: IW+PSEUDO={f1:.4f}  (IW alone={IW_ALONE[k]:.4f}, "
              f"Δ{f1-IW_ALONE[k]:+.4f}, pseudo rows={n_pl})")

    cm, im = float(np.mean(combo)), float(np.mean(IW_ALONE))
    print(f"\nIW+PSEUDO mean={cm:.4f}  vs IW alone {im:.4f}  (Δ{cm-im:+.4f})")

    if cm > im:
        print("Combo wins -> refit on full train + test, save CSV.")
        mvt = MultiVec(vec_word_char)
        Xall = mvt.fit_transform(np.concatenate([texts, test_texts]))
        Xtr_f, Xte_f = Xall[:len(texts)], Xall[len(texts):]
        w_tr = adversarial_train_weights(texts, test_texts)
        clf, n_pl = iw_pseudo_fit(Xtr_f, y, w_tr, Xte_f)
        pred = clf.predict(Xte_f)
        out = pd.DataFrame({"id": test["id"], "label": pred.astype(int)})
        path = OUT_DIR / "Task3_IWPseudo_Prediction.csv"
        out.to_csv(path, index=False)
        print(f"WROTE {path} ({len(out)} rows, machine={int(out['label'].sum())},"
              f" pseudo rows used={n_pl})")
    else:
        print("Combo does NOT beat IW alone; keeping "
              "Task3_PseudoLabel_Prediction.csv as this track's candidate.")
    print(f"Total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
