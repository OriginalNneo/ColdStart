"""
Generate the Iter-12 winner prediction on the REAL test (uncapped rep).
=======================================================================
Config (four-lens PASS, gamma=1.0 / frac=0.5 / rounds=1):
  base = stack RidgeClassifier(0.9,bal) on [1.6*word(1,3) | char_wb(2,6)], uncapped min_df=2
  1A IW: reweight train rows by density ratio P(test-like)/(1-P) from a train-vs-TEST
         char_wb(3,5) domain classifier (gamma=1.0 = full weights)
  1D self-train: 1 round, frac=0.5 of test per predicted class, class-BALANCED, pseudo weight 1
  + free exact-match overrides (test rows whose normalized text == a single-label train text)
Writes predictions/Task3_IWSelfTrain_Prediction.csv. No Kaggle submission here.
"""
import re, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from scratch_lens import load_data

SEED = 42
t0 = time.time()
FRAC, ROUNDS, WS, ALPHA = 0.5, 1, 1.6, 0.9


def build(texts_tr, texts_te):
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True)
    Xw = wv.fit_transform(texts_tr).astype(np.float32); Xc = cv.fit_transform(texts_tr).astype(np.float32)
    Xwt = wv.transform(texts_te).astype(np.float32); Xct = cv.transform(texts_te).astype(np.float32)
    return (sparse.hstack([Xw * WS, Xc]).tocsr(),
            sparse.hstack([Xwt * WS, Xct]).tocsr())


def clf_():
    return RidgeClassifier(alpha=ALPHA, class_weight="balanced")


def iw_weights(train_texts, test_texts, clip_p=(0.05, 0.95), cap=10.0):
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    Xtr = cv.fit_transform(train_texts); Xte = cv.transform(test_texts)
    X = sparse.vstack([Xtr, Xte]).tocsr()
    z = np.r_[np.zeros(Xtr.shape[0]), np.ones(Xte.shape[0])]
    dc = LogisticRegression(max_iter=300, C=1.0, random_state=SEED).fit(X, z)
    p = np.clip(dc.predict_proba(Xtr)[:, 1], *clip_p)
    w = np.clip(p / (1 - p), 0, cap)
    return (w / w.mean()).astype(np.float64), float(p.mean())


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

    w, pmean = iw_weights(texts, test_texts)
    print(f"IW weights: test-likeness pmean={pmean:.3f} w[min/med/max]="
          f"{w.min():.2f}/{np.median(w):.2f}/{w.max():.2f} ({time.time()-t0:.0f}s)", flush=True)

    clf = clf_().fit(Xtr, Y, sample_weight=w)
    base_pred = clf.predict(Xte).astype(int)
    print(f"round0 (IW-weighted base) machine_frac={base_pred.mean():.4f}", flush=True)

    for r in range(ROUNDS):
        m = clf.decision_function(Xte)
        take, pl = balanced_select(m, FRAC)
        Xc = sparse.vstack([Xtr, Xte[take]]).tocsr()
        yc = np.r_[Y, pl[take]]
        wc = np.r_[w, np.ones(int(take.sum()))]
        clf = clf_().fit(Xc, yc, sample_weight=wc)
        p = clf.predict(Xte).astype(int)
        print(f"round{r+1} pseudo_used={int(take.sum())} machine_frac={p.mean():.4f} "
              f"changed_vs_base={(p != base_pred).sum()} ({time.time()-t0:.0f}s)", flush=True)

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

    out = "predictions/Task3_IWSelfTrain_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"wrote {out} rows={len(pred)} machine={int(pred.sum())} "
          f"frac={pred.mean():.4f} changed_vs_base={(pred != base_pred).sum()} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    print(f"total {time.time()-t0:.0f}s", flush=True)
