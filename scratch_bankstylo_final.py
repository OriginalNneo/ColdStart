"""
Generate the Iter-17 prediction on the REAL test (uncapped).
============================================================
bankstylo_iwst = base stack [1.6*word(1,3)|char_wb(2,6)] uncapped
   + StandardScaled 16-feat LLR/style bank (Iter 15) * 0.02
   + StandardScaled 227-dim stylo block (scratch_agent5_stylo) * 0.04   (Iter 17, the new leg)
   + IW covariate-shift weighting toward test (gamma=1.0)
   + 1 round frac0.5 class-balanced self-training
   + free exact-match overrides
Four-lens min +0.0101, topical +0.025..0.035; proj ~0.784-0.786. Writes
predictions/Task3_BankStyloIWSelfTrain_Prediction.csv. No Kaggle submission here.
"""
import re, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data
from scratch_stage2_features import build_bank
from scratch_lensC_combine import build_dense

SEED = 42
t0 = time.time()
FRAC, ROUNDS, WS, ALPHA, BANK_S, STYLO_S = 0.5, 1, 1.6, 0.9, 0.02, 0.04


def build_stack(texts_tr, texts_te):
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
    z = np.r_[np.zeros(Xtr.shape[0]), np.ones(Xte.shape[0])]
    dc = LogisticRegression(max_iter=300, C=1.0, random_state=SEED).fit(
        sparse.vstack([Xtr, Xte]).tocsr(), z)
    p = np.clip(dc.predict_proba(Xtr)[:, 1], *clip_p)
    w = np.clip(p / (1 - p), 0, cap)
    return (w / w.mean()).astype(np.float64)


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
    Xtr, Xte = build_stack(texts, test_texts)
    print(f"stack nfeat={Xtr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    # bank leg (16 feats, LMs fit on train)
    Btr, (Bte,) = build_bank(texts, Y, [test_texts])
    sb = StandardScaler().fit(Btr)
    Bktr = sparse.csr_matrix(sb.transform(Btr) * BANK_S)
    Bkte = sparse.csr_matrix(sb.transform(Bte) * BANK_S)
    # stylo leg (227 feats, per-doc)
    Sty_tr = build_dense(texts); Sty_te = build_dense(test_texts)
    sst = StandardScaler().fit(Sty_tr)
    Stk = sparse.csr_matrix(sst.transform(Sty_tr) * STYLO_S)
    Ste = sparse.csr_matrix(sst.transform(Sty_te) * STYLO_S)

    Ftr = sparse.hstack([Xtr, Bktr, Stk]).tocsr()
    Fte = sparse.hstack([Xte, Bkte, Ste]).tocsr()
    print(f"fused legs: bank(16)x{BANK_S} + stylo(227)x{STYLO_S} -> nfeat={Ftr.shape[1]} "
          f"({time.time()-t0:.0f}s)", flush=True)

    w = iw_weights(texts, test_texts)
    print(f"IW weights w[min/med/max]={w.min():.2f}/{np.median(w):.2f}/{w.max():.2f}", flush=True)

    clf = clf_().fit(Ftr, Y, sample_weight=w)
    base_pred = clf.predict(Fte).astype(int)
    print(f"round0 machine_frac={base_pred.mean():.4f}", flush=True)
    for r in range(ROUNDS):
        m = clf.decision_function(Fte)
        take, pl = balanced_select(m, FRAC)
        Fc = sparse.vstack([Ftr, Fte[take]]).tocsr()
        yc = np.r_[Y, pl[take]]; wc = np.r_[w, np.ones(int(take.sum()))]
        clf = clf_().fit(Fc, yc, sample_weight=wc)
        p = clf.predict(Fte).astype(int)
        print(f"round{r+1} pseudo_used={int(take.sum())} machine_frac={p.mean():.4f} "
              f"changed_vs_base={(p != base_pred).sum()} ({time.time()-t0:.0f}s)", flush=True)

    pred = clf.predict(Fte).astype(int)

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

    out = "predictions/Task3_BankStyloIWSelfTrain_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"wrote {out} rows={len(pred)} machine={int(pred.sum())} frac={pred.mean():.4f} "
          f"changed_vs_base={(pred != base_pred).sum()} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    print(f"total {time.time()-t0:.0f}s", flush=True)
