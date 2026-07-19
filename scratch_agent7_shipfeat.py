"""
Track 7: shipped 5000-dim features + adversarial shift-drop.
Classical ML only. Uses scratch_lens harness for folds/anchor.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scratch_lens import load_data, get_folds, macro_f1, ANCHOR

warnings.filterwarnings("ignore")
SEED = 42
t0 = time.time()
def log(*a): print(*a, flush=True);

FEATCOLS = [f"{i:04d}" for i in range(1, 5001)]

def load_features():
    tr = pd.read_csv("data/train_features.csv", dtype={"id": str})
    te = pd.read_csv("data/test_features.csv")
    return tr, te

def align_train(tr_feat):
    # align feature rows to data/train.csv id order (== load_data order)
    df = pd.read_csv("data/train.csv", dtype={"id": str})
    order = df["id"].to_numpy()
    tr_feat = tr_feat.set_index("id").loc[order].reset_index()
    return tr_feat, df["label"].to_numpy(dtype=int)

def to_sparse(df, cols):
    X = df[cols].to_numpy(dtype=np.float32)
    return sparse.csr_matrix(X)

def eval_fold_matrix(X, Y, folds, est_factory, drop_mask=None):
    per = []
    for tr, val in folds:
        Xtr, Xev = X[tr], X[val]
        if drop_mask is not None:
            keep = ~drop_mask
            Xtr = Xtr[:, keep]; Xev = Xev[:, keep]
        clf = est_factory()
        clf.fit(Xtr, Y[tr])
        per.append(macro_f1(Y[val], clf.predict(Xev)))
    return float(np.mean(per)), [round(v, 4) for v in per]

def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    log(f"train={len(texts)} folds A={len(foldsA)} B={len(foldsB)} anchor={ANCHOR} ({time.time()-t0:.0f}s)")

    tr_feat, te_feat = load_features()
    tr_feat, Yf = align_train(tr_feat)
    assert np.array_equal(Yf, Y), "label alignment mismatch!"
    log(f"feature align OK. train_feat rows={len(tr_feat)} test_feat rows={len(te_feat)} ({time.time()-t0:.0f}s)")

    Xship = to_sparse(tr_feat, FEATCOLS)
    Xship_test = to_sparse(te_feat, FEATCOLS)
    dens = Xship.nnz / (Xship.shape[0]*Xship.shape[1])
    log(f"Xship {Xship.shape} density={dens:.4f} max={Xship.max():.3f} ({time.time()-t0:.0f}s)")

    A_anchor, B_anchor = ANCHOR["A"], ANCHOR["B"]
    results = {}  # name -> (a,b,af,bf)

    # ---- TASK 1: shipped features alone ----
    log("\n=== TASK 1: shipped features alone ===")
    ests = {
        "SVC_C0.1":  lambda: LinearSVC(C=0.1,  class_weight="balanced", random_state=SEED),
        "SVC_C0.25": lambda: LinearSVC(C=0.25, class_weight="balanced", random_state=SEED),
        "SVC_C1":    lambda: LinearSVC(C=1.0,  class_weight="balanced", random_state=SEED),
        "LogReg":    lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    }
    for nm, ef in ests.items():
        a, af = eval_fold_matrix(Xship, Y, foldsA, ef)
        b, bf = eval_fold_matrix(Xship, Y, foldsB, ef)
        results[f"ship_{nm}"] = (a, b, af, bf)
        log(f"  {nm:10s} A={a:.4f} dA={a-A_anchor:+.4f} {af}  |  B={b:.4f} dB={b-B_anchor:+.4f} {bf}  ({time.time()-t0:.0f}s)")

    # ---- TASK 2: fusion shipped + capped raw-text wideB ----
    log("\n=== TASK 2: fusion (shipped + capped wideB raw-text) ===")
    def wideB_capped():
        return [TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6), min_df=3,
                                max_features=300000, sublinear_tf=True)]
    def eval_fusion(folds, use_ship=True, use_text=True):
        per = []
        for tr, val in folds:
            mats_tr, mats_ev = [], []
            if use_text:
                vecs = wideB_capped()
                for v in vecs:
                    v.fit(texts[tr])
                    mats_tr.append(v.transform(texts[tr]))
                    mats_ev.append(v.transform(texts[val]))
            if use_ship:
                mats_tr.append(Xship[tr]); mats_ev.append(Xship[val])
            Xtr = sparse.hstack(mats_tr).tocsr()
            Xev = sparse.hstack(mats_ev).tocsr()
            clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
            clf.fit(Xtr, Y[tr])
            per.append(macro_f1(Y[val], clf.predict(Xev)))
        return float(np.mean(per)), [round(v,4) for v in per]
    # text-only capped (sanity vs anchor) then fusion
    ta, taf = eval_fusion(foldsA, use_ship=False, use_text=True)
    tb, tbf = eval_fusion(foldsB, use_ship=False, use_text=True)
    log(f"  text_capped_only A={ta:.4f} dA={ta-A_anchor:+.4f}  |  B={tb:.4f} dB={tb-B_anchor:+.4f}  ({time.time()-t0:.0f}s)")
    results["text_capped_only"] = (ta, tb, taf, tbf)
    fa, faf = eval_fusion(foldsA, use_ship=True, use_text=True)
    fb, fbf = eval_fusion(foldsB, use_ship=True, use_text=True)
    log(f"  FUSION           A={fa:.4f} dA={fa-A_anchor:+.4f} {faf}  |  B={fb:.4f} dB={fb-B_anchor:+.4f} {fbf}  ({time.time()-t0:.0f}s)")
    results["fusion_ship+text"] = (fa, fb, faf, fbf)

    # ---- TASK 3: adversarial shift-drop ----
    log("\n=== TASK 3: adversarial shift-drop ===")
    from sklearn.model_selection import cross_val_predict
    Xadv = sparse.vstack([Xship, Xship_test]).tocsr()
    yadv = np.r_[np.zeros(Xship.shape[0]), np.ones(Xship_test.shape[0])]
    from sklearn.metrics import roc_auc_score
    advclf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    # AUC via CV on the adversarial task
    scores = cross_val_predict(advclf, Xadv, yadv, cv=5, method="decision_function", n_jobs=1)
    auc = roc_auc_score(yadv, scores)
    log(f"  adversarial AUC (train-vs-test on shipped feats) = {auc:.4f} ({time.time()-t0:.0f}s)")
    # fit on full to rank features by discriminativeness
    advclf.fit(Xadv, yadv)
    coef = np.abs(advclf.coef_.ravel())
    rank = np.argsort(-coef)  # most discriminative first
    for K in (100, 500, 1000):
        drop_mask = np.zeros(5000, dtype=bool); drop_mask[rank[:K]] = True
        ef = lambda: LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
        a, af = eval_fold_matrix(Xship, Y, foldsA, ef, drop_mask=drop_mask)
        b, bf = eval_fold_matrix(Xship, Y, foldsB, ef, drop_mask=drop_mask)
        results[f"shipdrop_K{K}"] = (a, b, af, bf)
        log(f"  drop top-{K:<4d} A={a:.4f} dA={a-A_anchor:+.4f} {af}  |  B={b:.4f} dB={b-B_anchor:+.4f} {bf}  ({time.time()-t0:.0f}s)")

    # ---- SUMMARY ----
    log("\n=== SUMMARY (vs anchor A={:.4f} B={:.4f}) ===".format(A_anchor, B_anchor))
    best = None
    for nm, (a, b, af, bf) in results.items():
        passed = (a > A_anchor) and (b > B_anchor)
        log(f"  {nm:22s} A={a:.4f} ({a-A_anchor:+.4f})  B={b:.4f} ({b-B_anchor:+.4f})  {'PASS' if passed else 'FAIL'}")
        if passed and (best is None or (a+b) > (results[best][0]+results[best][1])):
            best = nm
    log(f"\nBEST PASSING: {best}")

    # ---- refit + predict if a passing candidate exists ----
    if best is not None:
        log(f"Refitting best passing candidate '{best}' on all 20k -> scratch_agent7_pred.csv")
        if best.startswith("ship_"):
            nm = best[len("ship_"):]
            clf = ests[nm]()
            clf.fit(Xship, Y); pred = clf.predict(Xship_test)
        elif best == "fusion_ship+text":
            vecs = wideB_capped(); mats_tr=[]; mats_te=[]
            for v in vecs:
                v.fit(texts); mats_tr.append(v.transform(texts)); mats_te.append(v.transform(test_texts))
            mats_tr.append(Xship); mats_te.append(Xship_test)
            clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
            clf.fit(sparse.hstack(mats_tr).tocsr(), Y)
            pred = clf.predict(sparse.hstack(mats_te).tocsr())
        elif best == "text_capped_only":
            vecs = wideB_capped(); mats_tr=[]; mats_te=[]
            for v in vecs:
                v.fit(texts); mats_tr.append(v.transform(texts)); mats_te.append(v.transform(test_texts))
            clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
            clf.fit(sparse.hstack(mats_tr).tocsr(), Y)
            pred = clf.predict(sparse.hstack(mats_te).tocsr())
        elif best.startswith("shipdrop_"):
            K = int(best.split("K")[1])
            Xadv2 = sparse.vstack([Xship, Xship_test]).tocsr()
            yadv2 = np.r_[np.zeros(Xship.shape[0]), np.ones(Xship_test.shape[0])]
            ac = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xadv2, yadv2)
            r = np.argsort(-np.abs(ac.coef_.ravel())); dm = np.zeros(5000,bool); dm[r[:K]]=True
            keep = ~dm
            clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
            clf.fit(Xship[:,keep], Y); pred = clf.predict(Xship_test[:,keep])
        out = pd.DataFrame({"id": test_ids, "label": pred.astype(int)})
        out.to_csv("scratch_agent7_pred.csv", index=False)
        log(f"wrote scratch_agent7_pred.csv rows={len(out)}")
    else:
        log("No passing candidate -> NO prediction file written (null result).")

    log(f"\nDONE ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
