"""
Task 3 (agent I) — INFERENCE-TIME length adaptation. Model completely untouched.
================================================================================
Train the EXACT baseline (word(1,2)+char_wb(3,5) sublinear TF-IDF min_df=2,
LinearSVC C=0.25 balanced) on unmodified training docs; only change what is fed
to it at PREDICT time. Test abstracts are ~1.4x longer than train (repo
whitespace tokenizer: train median 187, q75 321; test median 275). All
TRAINING-side length fixes failed; this tries making test docs look like train
docs at inference.

Candidates (applied to val/test docs only, NEVER to training docs):
  a. HEAD_CROP    : truncate to first 187 whitespace tokens (train median).
  b. MULTICROP_TTA: full doc + head-187 + tail-187 + middle-187; average the
                    LinearSVC decision_function over crops; threshold at 0.
  c. SLIDING      : 187-token windows, 50% overlap (stride 93), mean decision.
  d. COND_CROP    : head-crop to 187 ONLY for docs > 321 tokens (train q75);
                    shorter docs pass through unchanged.

Two lenses:
  Lens L (PRIMARY): length-shifted holdout — train shortest 60% by token count,
                    validate longest 40%. Baseline anchor 0.8022.
  Lens A (GUARD)  : 5-fold cluster-holdout (Task3_Improved_Model.cluster_folds).
                    Baseline anchor 0.7383.
Gate: both anchors must reproduce within 0.003 or we stop.
Acceptance: lens L delta > +0.003 AND lens A mean delta >= -0.002.

Run: nohup .venv/bin/python Task3_InferAdapt.py > scratch_inferadapt.log 2>&1 &
"""
import time
import warnings

import numpy as np
import pandas as pd

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, SEED, MultiVec, cluster_folds, est_svc, macro_f1,
    vec_word_char,
)

warnings.filterwarnings("ignore")

BASE_CLUS_TARGET = 0.7383
BASE_LEN_TARGET = 0.8022
REPRO_TOL = 0.003
SHORT_FRAC = 0.60
CROP = 187        # train median whitespace tokens (repo tokenizer)
COND_THRESH = 321  # train q75 whitespace tokens
STRIDE = CROP // 2  # 50% overlap

ACCEPT_L_MARGIN = 0.003   # lens L must improve by MORE than this
ACCEPT_A_TOL = 0.002      # lens A mean must not drop by more than this


def tok_count(text):
    return len(str(text).split())


# ---------------------------------------------------------------------------
# inference-time transforms: text -> list of crop strings (scores averaged)
# ---------------------------------------------------------------------------
def t_identity(text):
    return [str(text)]


def t_head_crop(text):
    w = str(text).split()
    if len(w) <= CROP:
        return [str(text)]
    return [" ".join(w[:CROP])]


def t_multicrop(text):
    w = str(text).split()
    n = len(w)
    if n <= CROP:
        return [str(text)]
    head = " ".join(w[:CROP])
    tail = " ".join(w[-CROP:])
    ms = (n - CROP) // 2
    mid = " ".join(w[ms:ms + CROP])
    return [str(text), head, tail, mid]


def t_sliding(text):
    w = str(text).split()
    n = len(w)
    if n <= CROP:
        return [str(text)]
    starts = list(range(0, n - CROP + 1, STRIDE))
    if starts[-1] != n - CROP:
        starts.append(n - CROP)
    return [" ".join(w[s:s + CROP]) for s in starts]


def t_cond_crop(text):
    w = str(text).split()
    if len(w) <= COND_THRESH:
        return [str(text)]
    return [" ".join(w[:CROP])]


CANDIDATES = [
    ("BASE",          t_identity),
    ("HEAD_CROP",     t_head_crop),
    ("MULTICROP_TTA", t_multicrop),
    ("SLIDING",       t_sliding),
    ("COND_CROP",     t_cond_crop),
]


def n_modified(texts, transform):
    """How many docs the transform actually changes."""
    c = 0
    for t in texts:
        crops = transform(t)
        if len(crops) > 1 or crops[0] != str(t):
            c += 1
    return c


def score_transform(clf, mv, val_texts, transform):
    """Mean decision_function over the transform's crops per doc -> labels."""
    flat, idx = [], []
    for i, t in enumerate(val_texts):
        for c in transform(t):
            flat.append(c)
            idx.append(i)
    idx = np.asarray(idx)
    s = clf.decision_function(mv.transform(flat))
    counts = np.bincount(idx, minlength=len(val_texts))
    mean_s = np.bincount(idx, weights=s, minlength=len(val_texts)) / counts
    return (mean_s > 0).astype(int)


def eval_lens(texts, y, folds, tag):
    """Fit the untouched baseline once per fold; score every candidate on the
    fold's val docs. Returns {name: [per-fold f1]} and {name: n_modified}."""
    per = {n: [] for n, _ in CANDIDATES}
    modified = {n: 0 for n, _ in CANDIDATES}
    nval_total = 0
    for k, (tr, val) in enumerate(folds):
        t0 = time.time()
        mv = MultiVec(vec_word_char)
        Xtr = mv.fit_transform(texts[tr])
        clf = est_svc(0.25)
        clf.fit(Xtr, y[tr])
        nval_total += len(val)
        for name, tf in CANDIDATES:
            pred = score_transform(clf, mv, texts[val], tf)
            per[name].append(macro_f1(y[val], pred))
            if k == 0 or tag == "L":  # count modified once per lens is enough,
                modified[name] += n_modified(texts[val], tf)  # but sum across folds
        print(f"  [{tag} fold {k}] n_val={len(val)}  " +
              "  ".join(f"{n}={per[n][-1]:.4f}" for n, _ in CANDIDATES) +
              f"  ({time.time()-t0:.0f}s)", flush=True)
    return per, modified, nval_total


def main():
    t0 = time.time()
    print("=" * 78)
    print("TASK 3 (agent I) — INFERENCE-TIME adaptation (model untouched)")
    print(f"  crop target={CROP} (train median), cond threshold={COND_THRESH} "
          f"(train q75), stride={STRIDE}")
    print("=" * 78, flush=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    toks = np.array([tok_count(t) for t in texts], dtype=float)
    toks_test = np.array([tok_count(t) for t in test_texts], dtype=float)
    print(f"train={len(texts)} test={len(test_texts)}  "
          f"train tok median={np.median(toks):.0f} q75={np.quantile(toks,.75):.0f}  "
          f"test tok median={np.median(toks_test):.0f}", flush=True)

    # ---- lenses ----
    clus, cl = cluster_folds(texts, y)
    print(f"\nLens A cluster-holdout: {len(clus)} folds  "
          f"sizes={np.bincount(cl).tolist()}", flush=True)
    q = np.quantile(toks, SHORT_FRAC)
    short_idx = np.where(toks <= q)[0]
    long_idx = np.where(toks > q)[0]
    len_folds = [(short_idx, long_idx)]
    print(f"Lens L length-shifted: cutoff={q:.0f} tok  train(short)={len(short_idx)} "
          f"val(long)={len(long_idx)} (pos={y[long_idx].mean():.3f})", flush=True)

    # ---- evaluate (BASE rides along = reproduction gate) ----
    print("\n[Lens L — PRIMARY (length-shifted holdout)]", flush=True)
    perL, modL, nvalL = eval_lens(texts, y, len_folds, "L")
    print("\n[Lens A — GUARD (5-fold cluster-holdout)]", flush=True)
    perA, modA, nvalA = eval_lens(texts, y, clus, "A")

    baseL = float(np.mean(perL["BASE"]))
    baseA = float(np.mean(perA["BASE"]))

    # ---- reproduction gate ----
    print("\n" + "-" * 78)
    print("[GATE] anchor reproduction")
    print("-" * 78, flush=True)
    print(f"  lens A BASE = {baseA:.4f} (target {BASE_CLUS_TARGET}, "
          f"|diff|={abs(baseA-BASE_CLUS_TARGET):.4f})")
    print(f"  lens L BASE = {baseL:.4f} (target {BASE_LEN_TARGET}, "
          f"|diff|={abs(baseL-BASE_LEN_TARGET):.4f})", flush=True)
    if abs(baseA - BASE_CLUS_TARGET) > REPRO_TOL or \
       abs(baseL - BASE_LEN_TARGET) > REPRO_TOL:
        print("  *** REPRODUCTION FAILED — stopping, not trusting downstream. ***")
        print("DONE_FAIL", flush=True)
        return
    print("  reproduction OK on both anchors.", flush=True)

    # ---- results table ----
    print("\n" + "=" * 78)
    print("[RESULTS] (lens L = primary, lens A mean = guard)")
    print("=" * 78, flush=True)
    print(f"  {'candidate':<15} {'lensL':>8} {'dL':>8} {'lensA':>8} {'dA':>8} "
          f"{'modL':>12} {'modA':>12}")
    accepted = []
    for name, _ in CANDIDATES:
        L = float(np.mean(perL[name]))
        A = float(np.mean(perA[name]))
        dL, dA = L - baseL, A - baseA
        print(f"  {name:<15} {L:>8.4f} {dL:>+8.4f} {A:>8.4f} {dA:>+8.4f} "
              f"{modL[name]:>6}/{nvalL:<5} {modA[name]:>6}/{nvalA:<5}", flush=True)
        if name != "BASE" and dL > ACCEPT_L_MARGIN and dA >= -ACCEPT_A_TOL:
            accepted.append((name, L, dL, A, dA))
    print("\n  per-fold lens A:")
    for name, _ in CANDIDATES:
        print(f"    {name:<15} " +
              " ".join(f"{v:.4f}" for v in perA[name]), flush=True)

    # ---- decision ----
    print("\n" + "=" * 78)
    print("[DECISION]")
    print("=" * 78, flush=True)
    if not accepted:
        print("  NULL RESULT: no inference-time transform improves lens L by "
              f">{ACCEPT_L_MARGIN} while holding lens A within {ACCEPT_A_TOL}.")
        print("  Not writing a prediction file (a null result is valid).")
        print(f"\n  runtime: {time.time()-t0:.0f}s")
        print("SHIP=none")
        print("DONE_OK", flush=True)
        return

    accepted.sort(key=lambda r: -r[1])
    winner, wL, wdL, wA, wdA = accepted[0]
    wtf = dict(CANDIDATES)[winner]
    print(f"  ACCEPTED: {winner}  lensL={wL:.4f} ({wdL:+.4f})  "
          f"lensA={wA:.4f} ({wdA:+.4f}) -> refit on 20K, transform test, write.",
          flush=True)

    mv = MultiVec(vec_word_char)
    X = mv.fit_transform(texts)
    clf = est_svc(0.25)
    clf.fit(X, y)
    pred = score_transform(clf, mv, test_texts, wtf)
    n_mod_test = n_modified(test_texts, wtf)
    out = OUT_DIR / "Task3_InferAdapt_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred.astype(int)}).to_csv(out, index=False)
    print(f"  WROTE {out}  rows={len(pred)}  machine={int(pred.sum())} "
          f"human={int((pred==0).sum())}  test docs modified={n_mod_test}/6999",
          flush=True)

    print(f"\n  runtime: {time.time()-t0:.0f}s")
    print(f"SHIP={winner}")
    print("DONE_OK", flush=True)


if __name__ == "__main__":
    main()
