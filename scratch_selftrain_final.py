"""
Generate the transductive SELF-TRAINING prediction on the REAL test (uncapped).
Config from tuner: base = stack (Ridge0.9 on [1.6*word(1,3)|char_wb(2,6)]),
pseudo-label frac=0.7 of test per predicted class (BALANCED), 3 rounds.
Plus free exact-match overrides (test rows whose normalized text == a train text).
"""
import re, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from scratch_lens import load_data

SEED = 42
t0 = time.time()
FRAC, ROUNDS, WS, ALPHA = 0.7, 3, 1.6, 0.9


def build(texts_tr, texts_te):
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True)
    Xw = wv.fit_transform(texts_tr).astype(np.float32); Xc = cv.fit_transform(texts_tr).astype(np.float32)
    Xwt = wv.transform(texts_te).astype(np.float32); Xct = cv.transform(texts_te).astype(np.float32)
    Xtr = sparse.hstack([Xw * WS, Xc]).tocsr()
    Xte = sparse.hstack([Xwt * WS, Xct]).tocsr()
    return Xtr, Xte


def clf_():
    return RidgeClassifier(alpha=ALPHA, class_weight="balanced")


def balanced_select(margin, frac):
    conf = np.abs(margin); pl = (margin > 0).astype(int)
    take = np.zeros(len(margin), bool)
    for c in (0, 1):
        idx = np.where(pl == c)[0]
        k = max(1, int(len(idx) * frac))
        take[idx[np.argsort(-conf[idx])[:k]]] = True
    return take, pl


def main():
    texts, Y, test_texts, test_ids = load_data()
    Xtr, Xte = build(texts, test_texts)
    print(f"built uncapped rep nfeat={Xtr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    clf = clf_().fit(Xtr, Y)
    base_pred = clf.predict(Xte).astype(int)
    print(f"round0 (base stack) machine_frac={base_pred.mean():.4f}", flush=True)

    for r in range(ROUNDS):
        m = clf.decision_function(Xte)
        take, pl = balanced_select(m, FRAC)
        Xc = sparse.vstack([Xtr, Xte[take]]).tocsr()
        yc = np.r_[Y, pl[take]]
        clf = clf_().fit(Xc, yc)
        p = clf.predict(Xte).astype(int)
        print(f"round{r+1} pseudo_used={int(take.sum())} machine_frac={p.mean():.4f} "
              f"changed_vs_base={(p!=base_pred).sum()} ({time.time()-t0:.0f}s)", flush=True)

    pred = clf.predict(Xte).astype(int)

    # ---- free exact-match overrides ----
    def norm(s): return re.sub(r"\s+", " ", str(s).strip().lower())
    tr_map = {}
    for t, l in zip(map(norm, texts), Y):
        tr_map.setdefault(t, set()).add(int(l))
    overrides = 0
    for i, t in enumerate(map(norm, test_texts)):
        if t in tr_map and len(tr_map[t]) == 1:
            lab = next(iter(tr_map[t]))
            if pred[i] != lab:
                overrides += 1
            pred[i] = lab
    print(f"exact-match overrides applied={overrides}", flush=True)

    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(
        "predictions/Task3_SelfTrain_Prediction.csv", index=False)
    print(f"wrote predictions/Task3_SelfTrain_Prediction.csv rows={len(pred)} "
          f"machine={int(pred.sum())} frac={pred.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    print(f"total {time.time()-t0:.0f}s", flush=True)
