"""
Task 3 (agent G) — TWO-LENS re-validation of the previously-promising levers.
================================================================================
Classical ML only (sklearn/numpy/scipy). NO deep learning.

WHY: every config selected by maximizing the lens-A proxy (cluster_folds in
Task3_Improved_Model.py: KMeans k=10 seed=42 on word unigrams, merged to 5
groups) scored WORSE on the real leaderboard than the untouched baseline
(proxy 0.7523 -> real 0.72853; proxy 0.7721 -> real 0.72429; baseline real
0.72990). The proxy is unbiased only for configs chosen independently of it;
selection on it inflates the winner. Fix: a SECOND, structurally different
lens whose errors decorrelate from lens A, plus the length lens as a guard.

LENSES
  A  cluster-holdout, KMeans k=10 seed=42 on WORD unigrams -> 5 groups
     (existing cluster_folds; baseline anchor 0.7383)
  B  cluster-holdout, KMeans k=16 seed=2026 on CHAR_WB(3,5) TF-IDF -> 5
     balanced groups (greedy size-balancing, same idea as lens A's merge)
  L  length lens (Task3_LengthAdapt.py): train on shortest 60% of train by
     whitespace token count, validate on longest 40% (single fold)

LEVERS (each ONE AT A TIME on top of the exact baseline
        word(1,2)+char_wb(3,5) sublinear TF-IDF min_df=2, LinearSVC C=0.25
        balanced):
  a  binary=True (no sublinear), l2 norm
  b  min_df=5
  c  transductive vocab (vectorizer fit on fold-train + fold-val text,
     classifier on labeled train rows only — Task3_PseudoLabel.py VOCAB)
  d  original adversarial IW, clip [0.25,4] — EXACT Task3_PseudoLabel.py
     machinery (transductive rep + adversarial char-ngram weights), the
     config that scored +0.014 on lens A but 0.72853 (-0.001) real
  e  binary + transductive (the one allowed combo)

ACCEPTANCE RULE (per lever): accepted only if it beats the anchor on BOTH
lens A mean and lens B mean, AND on >=4/5 folds of each, AND does not lose
more than 0.005 on lens L. A lever with lens-A delta below +0.014 that fails
lens B is decisively dead (lever d proved +0.014 on A alone means nothing).

Run: nohup .venv/bin/python Task3_TwoLens.py > scratch_twolens.log 2>&1 &
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, N_SPLITS, SEED, MultiVec, cluster_folds, est_svc,
    macro_f1, vec_word_char,
)
from Task3_PseudoLabel import adversarial_train_weights

warnings.filterwarnings("ignore")

LENSA_ANCHOR_TARGET = 0.7383   # must reproduce (tol below) before trusting runs
REPRO_TOL = 0.003
C_BASE = 0.25
SHORT_FRAC = 0.60              # lens L: shortest 60% train, longest 40% val
LENSB_K = 16
LENSB_SEED = 2026
LENS_L_MAX_LOSS = 0.005
MIN_FOLDS_WON = 4


# ---------------------------------------------------------------------------
# LENS B: char-space cluster holdout (structurally different from lens A)
# ---------------------------------------------------------------------------
def lens_b_folds(texts, y):
    """KMeans k=16 seed=2026 on char_wb(3,5) TF-IDF, merged into N_SPLITS
    balanced groups by greedy size-balancing (largest cluster first, each
    assigned to the currently-smallest group), hold out one group per fold."""
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=50000, sublinear_tf=True)
    Xc = cv.fit_transform(texts)
    km = MiniBatchKMeans(n_clusters=LENSB_K, random_state=LENSB_SEED,
                         n_init=5, batch_size=2048)
    cl = km.fit_predict(Xc)
    sizes = np.bincount(cl, minlength=LENSB_K)
    order = np.argsort(-sizes)                     # largest first
    group_tot = np.zeros(N_SPLITS, dtype=int)
    group_of_cluster = {}
    for c in order:                                 # greedy size-balancing
        g = int(np.argmin(group_tot))
        group_of_cluster[c] = g
        group_tot[g] += sizes[c]
    grp = np.array([group_of_cluster[c] for c in cl])
    folds = []
    for g in range(N_SPLITS):
        val = np.where(grp == g)[0]
        tr = np.where(grp != g)[0]
        if len(val) and len(np.unique(y[val])) == 2:
            folds.append((tr, val))
    return folds, cl, group_tot


# ---------------------------------------------------------------------------
# vectorizer factories (baseline + levers a, b)
# ---------------------------------------------------------------------------
def vec_binary_l2():
    """Lever a: binary=True, NO sublinear, l2 norm (min_df=2 as baseline)."""
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        binary=True, sublinear_tf=False, norm="l2"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        binary=True, sublinear_tf=False, norm="l2"),
    ]


def vec_mindf5():
    """Lever b: min_df=5 (everything else exactly baseline)."""
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=5,
                        sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                        sublinear_tf=True),
    ]


# ---------------------------------------------------------------------------
# unified evaluator: (factory, transductive?, iw?) over any fold list
# ---------------------------------------------------------------------------
def eval_config(texts, y, folds, factory, transductive=False, iw=False):
    """Per-fold macro-F1 for LinearSVC C=0.25 balanced over `factory` rep.
    transductive: vectorizer fit on fold-train + fold-val TEXT (no labels).
    iw: adversarial importance weights on train rows (implies transductive rep
        — exact Task3_PseudoLabel.py machinery)."""
    per_fold = []
    for tr, val in folds:
        ttr, tval = texts[tr], texts[val]
        mv = MultiVec(factory)
        if transductive or iw:
            Xall = mv.fit_transform(np.concatenate([ttr, tval]))
            Xtr, Xval = Xall[:len(tr)], Xall[len(tr):]
        else:
            Xtr = mv.fit_transform(ttr)
            Xval = mv.transform(tval)
        w = adversarial_train_weights(ttr, tval) if iw else None
        clf = est_svc(C_BASE)
        clf.fit(Xtr, y[tr], sample_weight=w)
        per_fold.append(macro_f1(y[val], clf.predict(Xval)))
    return np.array(per_fold)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("TASK 3 (agent G) — TWO-LENS re-validation (lens A + lens B + lens L)")
    print("=" * 78, flush=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    print(f"train={len(texts)}  test={len(test_texts)}", flush=True)

    # ---- build the three lenses ----
    foldsA, clA = cluster_folds(texts, y)                    # lens A (existing)
    print(f"\nLENS A: {len(foldsA)} folds  word-KMeans k=10 seed={SEED}  "
          f"cluster sizes={np.bincount(clA).tolist()}", flush=True)

    foldsB, clB, gtotB = lens_b_folds(texts, y)              # lens B (new)
    print(f"LENS B: {len(foldsB)} folds  char-KMeans k={LENSB_K} "
          f"seed={LENSB_SEED}  cluster sizes={np.bincount(clB).tolist()}  "
          f"group sizes={gtotB.tolist()}", flush=True)

    toks = np.array([len(str(t).split()) for t in texts], dtype=float)
    q = np.quantile(toks, SHORT_FRAC)
    short_idx = np.where(toks <= q)[0]
    long_idx = np.where(toks > q)[0]
    foldsL = [(short_idx, long_idx)]
    print(f"LENS L: token cutoff={q:.0f}  train(short)={len(short_idx)} "
          f"(pos={y[short_idx].mean():.3f})  val(long)={len(long_idx)} "
          f"(pos={y[long_idx].mean():.3f})", flush=True)

    # ---- lens A/B fold overlap diagnostic (are the lenses really different?)
    grpA = np.zeros(len(texts), dtype=int)
    for g, (_, val) in enumerate(foldsA):
        grpA[val] = g
    grpB = np.zeros(len(texts), dtype=int)
    for g, (_, val) in enumerate(foldsB):
        grpB[val] = g
    ct = pd.crosstab(grpA, grpB).to_numpy()
    max_overlap = ct.max() / len(texts)
    print(f"lens A x lens B group cross-tab (rows=A, cols=B):\n{ct}\n"
          f"  largest single-cell overlap = {max_overlap:.1%} of train "
          f"(uniform would be ~4%)", flush=True)

    # ================= ANCHORS: exact baseline on all three =================
    print("\n" + "-" * 78)
    print("[ANCHORS] exact baseline (word12+charwb35 sublinear min_df=2, "
          "SVC C=0.25 bal)")
    print("-" * 78, flush=True)
    anchors = {}
    for tag, folds in [("A", foldsA), ("B", foldsB), ("L", foldsL)]:
        pf = eval_config(texts, y, folds, vec_word_char)
        anchors[tag] = pf
        print(f"  lens {tag}: mean={pf.mean():.4f} +/- {pf.std():.4f}  "
              f"folds={[round(v, 4) for v in pf]}", flush=True)
    if abs(anchors["A"].mean() - LENSA_ANCHOR_TARGET) > REPRO_TOL:
        print(f"\n  *** LENS-A ANCHOR REPRODUCTION FAILED *** "
              f"{anchors['A'].mean():.4f} vs target {LENSA_ANCHOR_TARGET} "
              f"(tol {REPRO_TOL}). Stopping.", flush=True)
        print("DONE_FAIL", flush=True)
        return
    print(f"  lens-A anchor reproduced "
          f"(|diff|={abs(anchors['A'].mean()-LENSA_ANCHOR_TARGET):.4f} "
          f"<= {REPRO_TOL})", flush=True)

    # ============================ LEVERS ============================
    levers = [
        ("a_binary_l2",   vec_binary_l2,  False, False),
        ("b_mindf5",      vec_mindf5,     False, False),
        ("c_transd",      vec_word_char,  True,  False),
        ("d_advIW",       vec_word_char,  False, True),
        ("e_bin_transd",  vec_binary_l2,  True,  False),
    ]

    print("\n" + "-" * 78)
    print("[LEVERS] one at a time on top of the exact baseline; all 3 lenses")
    print("-" * 78, flush=True)

    results = {}
    for name, fac, transd, iw in levers:
        row = {}
        for tag, folds in [("A", foldsA), ("B", foldsB), ("L", foldsL)]:
            pf = eval_config(texts, y, folds, fac, transductive=transd, iw=iw)
            row[tag] = pf
            won = int((pf > anchors[tag]).sum())
            print(f"  {name:<13} lens {tag}: mean={pf.mean():.4f} "
                  f"(d={pf.mean()-anchors[tag].mean():+.4f})  "
                  f"folds-won={won}/{len(folds)}  "
                  f"folds={[round(v, 4) for v in pf]}", flush=True)
        results[name] = row

    # ============================ VERDICTS ============================
    print("\n" + "=" * 78)
    print("[VERDICTS] accept iff: beats anchor mean on A AND B, >=4/5 folds "
          "on each, lens-L loss <= 0.005")
    print("=" * 78, flush=True)
    print(f"  {'lever':<13} {'dA':>8} {'wonA':>5} {'dB':>8} {'wonB':>5} "
          f"{'dL':>8}  verdict", flush=True)

    accepted = []
    for name, row in results.items():
        dA = row["A"].mean() - anchors["A"].mean()
        dB = row["B"].mean() - anchors["B"].mean()
        dL = row["L"].mean() - anchors["L"].mean()
        wonA = int((row["A"] > anchors["A"]).sum())
        wonB = int((row["B"] > anchors["B"]).sum())
        ok = (dA > 0 and dB > 0 and wonA >= MIN_FOLDS_WON
              and wonB >= MIN_FOLDS_WON and dL >= -LENS_L_MAX_LOSS)
        verdict = "ACCEPTED" if ok else "REJECTED"
        if not ok and dA < 0.014 and dB <= 0:
            verdict += " (decisively dead: dA<+0.014 and fails lens B)"
        if ok:
            accepted.append((name, dB, dA))
        print(f"  {name:<13} {dA:>+8.4f} {wonA:>4}/5 {dB:>+8.4f} {wonB:>4}/5 "
              f"{dL:>+8.4f}  {verdict}", flush=True)

    # ============================ SHIP / NULL ============================
    if not accepted:
        print("\n  DECISION: NULL RESULT — no lever survives the two-lens "
              "protocol. This is the expected and valid outcome given the "
              "selection-inflation pathology. NOT writing a prediction file; "
              "the untouched baseline (real 0.72990) stands.", flush=True)
        print(f"\n  runtime: {time.time()-t0:.0f}s", flush=True)
        print("SHIP=none", flush=True)
        print("DONE_OK", flush=True)
        return

    # pick the accepted lever with the best lens-B mean (the decorrelated
    # lens), tie-break lens A
    accepted.sort(key=lambda x: (-x[1], -x[2]))
    winner = accepted[0][0]
    print(f"\n  DECISION: SHIP {winner} (accepted; best lens-B delta "
          f"{accepted[0][1]:+.4f}, lens-A {accepted[0][2]:+.4f}). "
          f"Refit on all 20K.", flush=True)

    fac, transd, iw = {n: (f, t, w) for n, f, t, w in levers}[winner]
    mv = MultiVec(fac)
    if transd or iw:
        Xall = mv.fit_transform(np.concatenate([texts, test_texts]))
        Xtr, Xte = Xall[:len(texts)], Xall[len(texts):]
    else:
        Xtr = mv.fit_transform(texts)
        Xte = mv.transform(test_texts)
    w = adversarial_train_weights(texts, test_texts) if iw else None
    clf = est_svc(C_BASE)
    clf.fit(Xtr, y, sample_weight=w)
    pred = clf.predict(Xte).astype(int)
    out = OUT_DIR / "Task3_TwoLens_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"  WROTE {out}  rows={len(pred)}  machine={int(pred.sum())} "
          f"human={int((pred == 0).sum())}", flush=True)

    print(f"\n  runtime: {time.time()-t0:.0f}s", flush=True)
    print(f"SHIP={winner}", flush=True)
    print("DONE_OK", flush=True)


if __name__ == "__main__":
    main()
