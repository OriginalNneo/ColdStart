"""
Generate the Iter-20 prediction on the REAL test (uncapped): add the function-word
syntactic-skeleton leg on top of the Iter-17 best (bankstylo_iwst).
=====================================================================
bankstylofw = base stack [1.6*word|char_wb26] uncapped
   + StandardScaled 16-feat LLR/style bank x0.02
   + StandardScaled 227-dim stylo x0.04
   + function-word skeleton TF-IDF word(1,3) x1.0   (Iter-20, new syntactic leg)
   + IW weighting toward test + 1 round frac0.5 balanced self-train + exact-match overrides
Writes predictions/Task3_BankStyloFW_Prediction.csv. No submission here.
"""
import re, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data
from scratch_stage2_features import build_bank
from scratch_lensC_combine import build_dense

SEED = 42
t0 = time.time()
FRAC, ROUNDS, WS, ALPHA, BANK_S, STYLO_S, FW_W = 0.5, 1, 1.6, 0.9, 0.02, 0.04, 1.0
_tok = re.compile(r"[A-Za-z']+|[.,;:!?()\-]")
_FW = set(w.lower() for w in ENGLISH_STOP_WORDS)


def skeleton(t):
    out = []
    for m in _tok.findall(t):
        if m.isalpha():
            out.append(m.lower() if m.lower() in _FW else "#")
        else:
            out.append(m)
    return " ".join(out)


def build_stack(tr, te):
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True)
    Xw = wv.fit_transform(tr).astype(np.float32); Xc = cv.fit_transform(tr).astype(np.float32)
    Xwt = wv.transform(te).astype(np.float32); Xct = cv.transform(te).astype(np.float32)
    return sparse.hstack([Xw * WS, Xc]).tocsr(), sparse.hstack([Xwt * WS, Xct]).tocsr()


def clf_():
    return RidgeClassifier(alpha=ALPHA, class_weight="balanced")


def iw_weights(tr, te, clip_p=(0.05, 0.95), cap=10.0):
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=30000, sublinear_tf=True)
    Xtr = cv.fit_transform(tr); Xte = cv.transform(te)
    z = np.r_[np.zeros(Xtr.shape[0]), np.ones(Xte.shape[0])]
    dc = LogisticRegression(max_iter=300, C=1.0, random_state=SEED).fit(
        sparse.vstack([Xtr, Xte]).tocsr(), z)
    p = np.clip(dc.predict_proba(Xtr)[:, 1], *clip_p)
    return (np.clip(p / (1 - p), 0, cap) / np.clip(p / (1 - p), 0, cap).mean()).astype(np.float64)


def bsel(m, frac):
    conf = np.abs(m); pl = (m > 0).astype(int); take = np.zeros(len(m), bool)
    for c in (0, 1):
        idx = np.where(pl == c)[0]; k = max(1, int(len(idx) * frac))
        take[idx[np.argsort(-conf[idx])[:k]]] = True
    return take, pl


def main():
    texts, Y, test_texts, test_ids = load_data()
    Xtr, Xte = build_stack(texts, test_texts)
    print(f"stack nfeat={Xtr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    Btr, (Bte,) = build_bank(texts, Y, [test_texts])
    sb = StandardScaler().fit(Btr)
    Styr = build_dense(texts); Stye = build_dense(test_texts)
    ss = StandardScaler().fit(Styr)
    sk_tr = [skeleton(t) for t in texts]; sk_te = [skeleton(t) for t in test_texts]
    fv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
    Fw_tr = fv.fit_transform(sk_tr).astype(np.float32); Fw_te = fv.transform(sk_te).astype(np.float32)

    Ftr = sparse.hstack([Xtr, sparse.csr_matrix(sb.transform(Btr) * BANK_S),
                         sparse.csr_matrix(ss.transform(Styr) * STYLO_S), Fw_tr * FW_W]).tocsr()
    Fte = sparse.hstack([Xte, sparse.csr_matrix(sb.transform(Bte) * BANK_S),
                         sparse.csr_matrix(ss.transform(Stye) * STYLO_S), Fw_te * FW_W]).tocsr()
    print(f"fused all legs nfeat={Ftr.shape[1]} (fw {Fw_tr.shape[1]}) ({time.time()-t0:.0f}s)", flush=True)

    w = iw_weights(texts, test_texts)
    clf = clf_().fit(Ftr, Y, sample_weight=w)
    base_pred = clf.predict(Fte).astype(int)
    print(f"round0 machine_frac={base_pred.mean():.4f}", flush=True)
    for r in range(ROUNDS):
        m = clf.decision_function(Fte); take, pl = bsel(m, FRAC)
        clf = clf_().fit(sparse.vstack([Ftr, Fte[take]]).tocsr(), np.r_[Y, pl[take]],
                         sample_weight=np.r_[w, np.ones(int(take.sum()))])
        p = clf.predict(Fte).astype(int)
        print(f"round{r+1} pseudo={int(take.sum())} machine_frac={p.mean():.4f} "
              f"changed={(p != base_pred).sum()} ({time.time()-t0:.0f}s)", flush=True)
    pred = clf.predict(Fte).astype(int)

    def norm(s): return re.sub(r"\s+", " ", str(s).strip().lower())
    tr_map = {}
    for t, l in zip(map(norm, texts), Y):
        tr_map.setdefault(t, set()).add(int(l))
    for i, t in enumerate(map(norm, test_texts)):
        if t in tr_map and len(tr_map[t]) == 1:
            pred[i] = next(iter(tr_map[t]))

    out = "predictions/Task3_BankStyloFW_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"wrote {out} rows={len(pred)} machine={int(pred.sum())} frac={pred.mean():.4f} "
          f"changed_vs_base={(pred != base_pred).sum()} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    print(f"total {time.time()-t0:.0f}s", flush=True)
