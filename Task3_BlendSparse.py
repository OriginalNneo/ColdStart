"""
Task3_BlendSparse.py — agent E (blend track), sparse legs.

Computes cluster-fold-0/1 VAL probabilities for the two sparse legs, plus the
full-train TEST probabilities for leg A, so a blend weight can be tuned on the
cluster-holdout folds against the transformer-448 fold-val probs.

Leg A (PRIMARY): agent A's IW-tuned LinearSVC.
    word(1,3)+char_wb(2,5) TF-IDF, min_df=2, sublinear, transductive vocab,
    adversarial IW weights clip (0.1,10), C=1.0, class_weight=balanced.
    LinearSVC has no predict_proba -> Platt calibration via
    CalibratedClassifierCV(method='sigmoid', cv=3) fit with sample_weight=IW.
    Calibration is applied BYTE-IDENTICALLY on folds AND the full-train test
    refit, so the tuned weight transfers.

Leg B (ALTERNATIVE): agent B's LR-C16 [iw] — the exact pipeline that produced
    predictions/Task3_SparseScreen_probs.npy (word(1,2)+char_wb(3,5), min_df=2,
    transductive vocab, adversarial IW clip (0.25,4.0), LogisticRegression C=16
    balanced liblinear, raw predict_proba). Recomputed on folds 0/1 to tune the
    weight; the TEST probs are reused from the saved npy (no refit needed).

Run: nohup .venv/bin/python Task3_BlendSparse.py > scratch_blend_sparse.log 2>&1 &
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

from Task3_Improved_Model import DATA_DIR, cluster_folds, macro_f1, est_svc
# Leg A IW machinery (adversarial OOF p + clipped weights)
from Task3_IW_Tuned import adv_p_train, weights_from_p
# Leg B: reuse SparseScreen's exact rep builder + LR factory + its IW weights
from Task3_SparseScreen import build_rep, lr_wc

T0 = time.time()

# ---- Leg A representation: word(1,3) + char_wb(2,5), min_df=2, sublinear ----
A_CLIP = (0.1, 10.0)


def legA_rep(ttr, tval):
    """Transductive vocab on ttr+tval; returns (Xtr, Xval)."""
    mats_tr, mats_val = [], []
    for params in (dict(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
                   dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)):
        v = TfidfVectorizer(**params)
        Xall = v.fit_transform(np.concatenate([ttr, tval]))
        mats_tr.append(Xall[:len(ttr)])
        mats_val.append(Xall[len(ttr):])
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_val).tocsr()


def legA_probs(Xtr, ytr, Xval, w):
    cc = CalibratedClassifierCV(est_svc(1.0), method="sigmoid", cv=3)
    cc.fit(Xtr, ytr, sample_weight=w)
    return cc.predict_proba(Xval)[:, 1], cc


def log(m):
    print(f"[+{time.time()-T0:6.0f}s] {m}", flush=True)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    log(f"train={len(texts)} test={len(test_texts)} pos={y.mean():.3f}")

    folds, _ = cluster_folds(texts, y)
    log(f"cluster folds: {len(folds)} (val sizes {[len(v) for _, v in folds]})")

    from Task3_PseudoLabel import adversarial_train_weights  # leg B IW clip (0.25,4)

    for k in (0, 1):
        tr, va = folds[k]
        ttr, tval, ytr, yval = texts[tr], texts[va], y[tr], y[va]
        np.save(f"scratch_blend_sparse_validx_fold{k}.npy", va.astype(np.int64))

        # ---- Leg A ----
        t = time.time()
        pA = adv_p_train(ttr, tval)          # char_wb(3,5) LR C1 OOF p (test-like)
        wA = weights_from_p(pA, A_CLIP)
        XtrA, XvalA = legA_rep(ttr, tval)
        probsA, _ = legA_probs(XtrA, ytr, XvalA, wA)
        f1A = macro_f1(yval, (probsA >= 0.5).astype(int))
        np.save(f"scratch_blendA_fold{k}_probs.npy", probsA.astype(np.float64))
        log(f"fold{k} LEG A (SVC-IW-cal) F1@0.5={f1A:.4f} "
            f"probs[min/med/max]={probsA.min():.3f}/{np.median(probsA):.3f}/{probsA.max():.3f} "
            f"({time.time()-t:.0f}s)")
        del XtrA, XvalA

        # ---- Leg B (reproduce SparseScreen LR-C16 [iw]) ----
        t = time.time()
        wB = adversarial_train_weights(ttr, tval)   # clip (0.25,4.0)
        XtrB, XvalB = build_rep("wc", ttr, tval, transductive=True)
        clfB = lr_wc(16.0)
        clfB.fit(XtrB, ytr, sample_weight=wB)
        probsB = clfB.predict_proba(XvalB)[:, 1]
        f1B = macro_f1(yval, (probsB >= 0.5).astype(int))
        np.save(f"scratch_blendB_fold{k}_probs.npy", probsB.astype(np.float64))
        log(f"fold{k} LEG B (LR-C16-iw)  F1@0.5={f1B:.4f} "
            f"(SparseScreen ref: f0=0.7420 f1=0.8040) ({time.time()-t:.0f}s)")
        del XtrB, XvalB

    # ---- Leg A full-train TEST probs (same calibration, transductive train+test)
    log("full-train LEG A refit for test probs ...")
    t = time.time()
    pA_full = adv_p_train(texts, test_texts)
    wA_full = weights_from_p(pA_full, A_CLIP)
    XtrF, XteF = legA_rep(texts, test_texts)
    ccF = CalibratedClassifierCV(est_svc(1.0), method="sigmoid", cv=3)
    ccF.fit(XtrF, y, sample_weight=wA_full)
    testA = ccF.predict_proba(XteF)[:, 1]
    np.save("scratch_blendA_test_probs.npy", testA.astype(np.float64))
    log(f"LEG A test probs saved: mean={testA.mean():.4f} "
        f"[min/med/max]={testA.min():.3f}/{np.median(testA):.3f}/{testA.max():.3f} "
        f"({time.time()-t:.0f}s)")

    log(f"ALL DONE total {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
