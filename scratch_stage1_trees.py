"""
STAGE 1 (trees) — do decision-tree / gradient-boosting models help, four-lens gated?
====================================================================================
User asked to try trees. They ARE eligible (classical, not DL). BUT the ledger's
single strongest lesson is that tree/ensemble CAPACITY is the failure mode here:
Iter 1 (XGBoost stack) scored 0.69924 — our worst submission ever — vs the linear
stack's 0.752. Mechanism: trees fit train topics harder (train-F1→0.99), the shifted
test punishes it. This script re-tests that prior HONESTLY on the CURRENT base + all
four shift lenses, plus the one tree use with a real shot (low-capacity shift-aware
calibration). Whatever happens is logged append-only. No deep learning.

Candidates (vs the same base stack, non-circular pool/eval protocol):
  base        RidgeClassifier(0.9,bal) on [1.6*word | char]                  (the 0.752 model)
  gbm_lsa     HistGradientBoosting on TruncatedSVD(256) of the stack features (tree-as-classifier)
  rf_lsa      RandomForest(400) on the same LSA features                     (tree-as-classifier)
  stack_gbm   average(linear stack decision, gbm_lsa margin)                 (tree as complementary leg)
  tree_calib  shallow tree (depth 3) on [stack score, log length] -> shift-aware relabel (tree-as-instrument)

GATE: min mean-Δ over {A,B,C1,C2} > 0 with non-catastrophic worst fold.
"""
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
CANDS = ["base", "gbm_lsa", "rf_lsa", "stack_gbm", "tree_calib"]


def clf_():
    return RidgeClassifier(alpha=0.9, class_weight="balanced")


def build(texts_tr, others, ws=1.6):
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                         max_features=300000, sublinear_tf=True)
    Xw = vw.fit_transform(texts_tr).astype(np.float32)
    Xc = vc.fit_transform(texts_tr).astype(np.float32)
    Xs = sparse.hstack([Xw * ws, Xc]).tocsr()
    outs = []
    for T in others:
        outs.append(sparse.hstack([vw.transform(T).astype(np.float32) * ws,
                                    vc.transform(T).astype(np.float32)]).tocsr())
    return Xs, outs


def eval_lens(name, folds, texts, Y):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        val = np.array(val); rng.shuffle(val); h = len(val) // 2
        eval_idx = val[h:]                                  # eval half (parity w/ other Stage-1 scripts)
        Xs, (Xe,) = build(texts[tr], [texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        len_tr = np.log1p(np.array([len(t) for t in texts[tr]])).reshape(-1, 1)
        len_ev = np.log1p(np.array([len(t) for t in texts[eval_idx]])).reshape(-1, 1)

        # base linear stack
        base = clf_().fit(Xs, ytr)
        d_ev = base.decision_function(Xe)
        acc["base"].append(macro_f1(yev, (d_ev > 0).astype(int)))

        # LSA for tree models (trees can't use 500k-dim sparse directly)
        svd = TruncatedSVD(n_components=256, random_state=SEED)
        Ztr = svd.fit_transform(Xs); Zev = svd.transform(Xe)

        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                             max_depth=None, random_state=SEED).fit(Ztr, ytr)
        g_ev = gbm.decision_function(Zev)
        acc["gbm_lsa"].append(macro_f1(yev, (g_ev > 0).astype(int)))

        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED,
                                    class_weight="balanced").fit(Ztr, ytr)
        acc["rf_lsa"].append(macro_f1(yev, rf.predict(Zev)))

        # trees as a COMPLEMENTARY leg: average z-scored linear + gbm margins
        zl = (d_ev - d_ev.mean()) / (d_ev.std() + 1e-9)
        zg = (g_ev - g_ev.mean()) / (g_ev.std() + 1e-9)
        acc["stack_gbm"].append(macro_f1(yev, ((zl + zg) > 0).astype(int)))

        # tree as INSTRUMENT: shallow tree on [oof stack score, log len] for shift-aware relabel
        d_tr = cross_val_predict(clf_(), Xs, ytr, cv=3, method="decision_function")
        Ftr = np.hstack([d_tr.reshape(-1, 1), len_tr])
        Fev = np.hstack([d_ev.reshape(-1, 1), len_ev])
        cal = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                     random_state=SEED).fit(Ftr, ytr)
        acc["tree_calib"].append(macro_f1(yev, cal.predict(Fev)))

        print(f"  [{name}] f{fi} base={acc['base'][-1]:.4f} gbm={acc['gbm_lsa'][-1]:.4f} "
              f"rf={acc['rf_lsa'][-1]:.4f} s+g={acc['stack_gbm'][-1]:.4f} "
              f"cal={acc['tree_calib'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    print(f"train={len(texts)} folds " + " ".join(f"{n}={len(f)}" for n, f in lenses), flush=True)
    res = {n: eval_lens(n, f, texts, Y) for n, f in lenses}

    print("\n===== Δ vs base (mean per lens; min = four-lens gate) =====", flush=True)
    print(f"{'candidate':12s}{'A':>10s}{'B':>10s}{'C1':>10s}{'C2':>10s}{'min':>10s}{'worst':>10s}  gate",
          flush=True)
    for c in CANDS:
        if c == "base":
            continue
        means = [float((res[n][c] - res[n]["base"]).mean()) for n, _ in lenses]
        worst = min(float((res[n][c] - res[n]["base"]).min()) for n, _ in lenses)
        gate = "PASS" if min(means) > 0 else "fail"
        print(f"{c:12s}" + "".join(f"{m:+10.4f}" for m in means) +
              f"{min(means):+10.4f}{worst:+10.4f}  [{gate}]", flush=True)

    print("\nabsolute per-lens means:", flush=True)
    for n, _ in lenses:
        print(f"  Lens {n}: " + ", ".join(f"{c}={res[n][c].mean():.4f}" for c in CANDS), flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
