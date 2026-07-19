"""
Task 3 (agent C) — LENGTH-SHIFT adaptation for the SPARSE-TEXT family only.
================================================================================
Test abstracts are ~25% longer than train (this tokenizer: train median 190 vs
test 279 whitespace tokens). Two earlier submissions that leaned on length-
correlated signals deflated -0.075 from the cluster-holdout proxy to the real
leaderboard; single sparse-text models deflate only -0.008. So length is mostly
NOT the sparse family's problem -- a NULL result ("length adaptation does not
help sparse models") is a legitimate, likely outcome. We do NOT fit any lens
harder to manufacture a ship.

TWO LENSES (every experiment is scored on BOTH):
  (1) CLUSTER-HOLDOUT (5-fold, topic KMeans) -- the CALIBRATED ANCHOR.
      Baseline scores 0.7383 here vs 0.7299 real LB (deflation -0.008).
  (2) LENGTH-SHIFTED HOLDOUT -- train on shortest ~60% of train rows by token
      count, validate on longest ~40%. Directly simulates the train->test
      length shift.

CRUCIAL CAVEAT (why we never ship on the length lens alone):
  The length holdout's val partition PRESERVES the train-internal "long => machine"
  correlation (val positive rate ~0.66 vs train ~0.60). A model that EXPLOITS
  length->label therefore SCORES WELL on the length holdout -- but that is exactly
  the correlation that did NOT transfer to the real test (the -0.075 trap).
  Length-quantile importance weighting (upweight long rows) leans INTO that
  correlation and is the prime suspect to win the length lens for the WRONG
  reason. Cropping / binary+L2 / l1-norm REMOVE length reliance and may LOWER the
  length-lens score while being safer. Therefore:
    - CLUSTER-HOLDOUT is the anchor and the tie-breaker.
    - We ship ONLY if a config beats BASE on the CLUSTER lens by a non-noise
      margin (and is not worse on the length lens), with a train/val gap no worse
      than BASE. A length-lens-only win is read as the trap, not a win.

Run: nohup .venv/bin/python Task3_LengthAdapt.py > scratch_lengthadapt.log 2>&1 &
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, SEED, MultiVec, cluster_folds, est_svc, macro_f1,
)

warnings.filterwarnings("ignore")

BASE_CLUS_TARGET = 0.7383   # must reproduce before trusting anything
REPRO_TOL = 0.003
IW_CLIP = (0.25, 4.0)
SHORT_FRAC = 0.60           # length-holdout: shortest 60% train, longest 40% val
CROP_MEDIAN = 190           # ~ train-median whitespace tokens
CROP_SCALES = [130, 190, 279]  # mixture: <train-median, train-median, ~test-median


# ---------------------------------------------------------------------------
# tokenization (whitespace; consistent with cropping)
# ---------------------------------------------------------------------------
def tok_count(text):
    return len(str(text).split())


# ---------------------------------------------------------------------------
# vectorizer factories (word 1-2 + char_wb 3-5, parametrizable rep)
# ---------------------------------------------------------------------------
def make_wc_factory(binary=False, sublinear=True, norm="l2"):
    def factory():
        return [
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                            binary=binary, sublinear_tf=sublinear, norm=norm),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                            binary=binary, sublinear_tf=sublinear, norm=norm),
        ]
    return factory


BASE_FACTORY = make_wc_factory()                              # baseline rep
BINARY_L2_FACTORY = make_wc_factory(binary=True, sublinear=False, norm="l2")
L1_FACTORY = make_wc_factory(binary=False, sublinear=True, norm="l1")


# ---------------------------------------------------------------------------
# recipes: (texts_tr, y_tr, toks_tr, rng) -> (train_texts, train_y, train_w|None)
# ---------------------------------------------------------------------------
def crop_one(text, target, rng):
    """Random contiguous word crop to `target` whitespace tokens; keep raw text
    (punctuation intact) if already <= target."""
    words = str(text).split()
    if len(words) <= target:
        return str(text)
    start = int(rng.integers(0, len(words) - target + 1))
    return " ".join(words[start:start + target])


def r_base(tt, yy, tk, rng):
    return tt, yy, None


def make_crop_replace(target):
    def f(tt, yy, tk, rng):
        out = np.array([crop_one(t, target, rng) for t in tt], dtype=object)
        return out, yy, None
    return f


def make_crop_aug(targets):
    """Augment: originals + one random-scale crop per doc (doubles rows)."""
    def f(tt, yy, tk, rng):
        crops = np.array(
            [crop_one(t, targets[int(rng.integers(len(targets)))], rng)
             for t in tt], dtype=object)
        allt = np.concatenate([np.asarray(tt, dtype=object), crops])
        ally = np.concatenate([yy, yy])
        return allt, ally, None
    return f


def make_iw(w_of_tok):
    """Length-quantile importance weighting: density ratio of test vs train
    log token count, clipped, renormalized to mean 1 WITHIN the training subset
    (keeps effective C comparable to baseline)."""
    def f(tt, yy, tk, rng):
        w = w_of_tok(np.asarray(tk, dtype=float))
        w = w / w.mean()
        return tt, yy, w
    return f


def make_crop_aug_iw(targets, w_of_tok):
    """Combo: crop-augment then weight every (original+crop) row by its OWN
    length importance weight (crops get their post-crop length)."""
    base_aug = make_crop_aug(targets)

    def f(tt, yy, tk, rng):
        allt, ally, _ = base_aug(tt, yy, tk, rng)
        lens = np.array([len(str(t).split()) for t in allt], dtype=float)
        w = w_of_tok(lens)
        w = w / w.mean()
        return allt, ally, w
    return f


# ---------------------------------------------------------------------------
# unified evaluator
# ---------------------------------------------------------------------------
def eval_recipe(texts, y, toks, folds, factory, recipe, C=0.25, seed=SEED):
    vf, tf = [], []
    for i, (tr, val) in enumerate(folds):
        rng = np.random.default_rng(seed + i)
        ttr, ytr, w = recipe(texts[tr], y[tr], toks[tr], rng)
        mv = MultiVec(factory)
        Xtr = mv.fit_transform(ttr)
        Xval = mv.transform(texts[val])
        clf = est_svc(C)
        clf.fit(Xtr, ytr, sample_weight=w)
        vf.append(macro_f1(y[val], clf.predict(Xval)))
        tf.append(macro_f1(ytr, clf.predict(Xtr)))
    return float(np.mean(vf)), float(np.std(vf)), float(np.mean(tf)), \
        float(np.mean(tf) - np.mean(vf))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("TASK 3 (agent C) — LENGTH-SHIFT adaptation, sparse-text family only")
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
          f"train tok median={np.median(toks):.0f} mean={toks.mean():.0f}  "
          f"test tok median={np.median(toks_test):.0f} mean={toks_test.mean():.0f}",
          flush=True)

    # ---- length importance-weight function (test vs train, log token count) ----
    ltr = np.log1p(toks)
    lte = np.log1p(toks_test)
    kde_tr = gaussian_kde(ltr)
    kde_te = gaussian_kde(lte)

    def w_of_tok(tk):
        x = np.log1p(np.asarray(tk, dtype=float))
        r = kde_te(x) / np.maximum(kde_tr(x), 1e-9)
        return np.clip(r, *IW_CLIP)

    # ---- LENS 1: cluster-holdout (anchor) ----
    clus, cl = cluster_folds(texts, y)
    print(f"\nLENS 1 cluster-holdout: {len(clus)} folds  "
          f"cluster sizes={np.bincount(cl).tolist()}", flush=True)

    # ---- LENS 2: length-shifted holdout (shortest 60% -> longest 40%) ----
    q = np.quantile(toks, SHORT_FRAC)
    short_idx = np.where(toks <= q)[0]
    long_idx = np.where(toks > q)[0]
    length_folds = [(short_idx, long_idx)]
    print(f"LENS 2 length-shifted holdout: token cutoff={q:.0f}  "
          f"train(short)={len(short_idx)} (pos={y[short_idx].mean():.3f})  "
          f"val(long)={len(long_idx)} (pos={y[long_idx].mean():.3f})", flush=True)
    print(f"  train pos rate overall={y.mean():.3f}  "
          f"-> length lens shifts the LABEL PRIOR, not only the covariate",
          flush=True)

    # ============================ REPRODUCE GATE ============================
    print("\n" + "-" * 78)
    print("[GATE] reproduce BASE cluster-holdout = %.4f" % BASE_CLUS_TARGET)
    print("-" * 78, flush=True)
    b_cv, b_cs, b_ct, b_cg = eval_recipe(texts, y, toks, clus, BASE_FACTORY, r_base)
    print(f"  BASE cluster-holdout macro-F1 = {b_cv:.4f} +/- {b_cs:.3f} "
          f"(train {b_ct:.4f}, gap {b_cg:+.3f})", flush=True)
    if abs(b_cv - BASE_CLUS_TARGET) > REPRO_TOL:
        print(f"\n  *** REPRODUCTION FAILED *** |{b_cv:.4f}-{BASE_CLUS_TARGET}| "
              f"> {REPRO_TOL}. Stopping; not trusting downstream numbers.")
        print("DONE_FAIL", flush=True)
        return
    print(f"  reproduction OK (|diff|={abs(b_cv-BASE_CLUS_TARGET):.4f} "
          f"<= {REPRO_TOL}).", flush=True)

    # baseline on length lens (reference deflation)
    l_bv, _, l_bt, l_bg = eval_recipe(texts, y, toks, length_folds,
                                      BASE_FACTORY, r_base)
    print(f"  BASE length-holdout  macro-F1 = {l_bv:.4f} "
          f"(train {l_bt:.4f}, gap {l_bg:+.3f})  "
          f"[length-lens deflation vs cluster = {l_bv-b_cv:+.4f}]", flush=True)

    # ============================ EXPERIMENTS ============================
    print("\n" + "-" * 78)
    print("[EXPERIMENTS] each scored on BOTH lenses (C=0.25 SVC balanced)")
    print("-" * 78, flush=True)

    experiments = [
        ("BASE",              BASE_FACTORY,      r_base),
        ("CROP_med_replace",  BASE_FACTORY,      make_crop_replace(CROP_MEDIAN)),
        ("CROP_med_aug",      BASE_FACTORY,      make_crop_aug([CROP_MEDIAN])),
        ("CROP_mix_aug",      BASE_FACTORY,      make_crop_aug(CROP_SCALES)),
        ("IW_length",         BASE_FACTORY,      make_iw(w_of_tok)),
        ("REP_binary_l2",     BINARY_L2_FACTORY, r_base),
        ("REP_l1norm",        L1_FACTORY,        r_base),
        ("COMBO_cropaug_bin", BINARY_L2_FACTORY, make_crop_aug([CROP_MEDIAN])),
        ("COMBO_cropaug_iw",  BASE_FACTORY,      make_crop_aug_iw(CROP_SCALES, w_of_tok)),
    ]

    print(f"  {'config':<20} {'CLUS(anchor)':>16} {'LEN':>16} "
          f"{'clus_gap':>9} {'len_gap':>8}", flush=True)
    results = {}
    for name, fac, rec in experiments:
        cv, cs, ct, cg = eval_recipe(texts, y, toks, clus, fac, rec)
        lv, _, lt, lg = eval_recipe(texts, y, toks, length_folds, fac, rec)
        results[name] = dict(clus=cv, clus_std=cs, clus_gap=cg,
                             length=lv, length_gap=lg, factory=fac, recipe=rec)
        print(f"  {name:<20} {cv:.4f}+/-{cs:.3f} {lv:>16.4f} "
              f"{cg:>+9.3f} {lg:>+8.3f}", flush=True)

    # ============================ DECISION ============================
    print("\n" + "=" * 78)
    print("[DECISION] anchor = CLUSTER-HOLDOUT; length lens is a stability check")
    print("=" * 78, flush=True)
    base_clus = results["BASE"]["clus"]
    base_len = results["BASE"]["length"]
    base_cgap = results["BASE"]["clus_gap"]
    print(f"  BASE: cluster={base_clus:.4f} length={base_len:.4f} "
          f"clus_gap={base_cgap:+.3f}", flush=True)

    print("\n  ranked by CLUSTER-HOLDOUT (anchor):", flush=True)
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["clus"]):
        wrong = (r["length"] > base_len + 0.005) and (r["clus"] <= base_clus + 0.005)
        flag = "  <- length-lens win w/o cluster win (LIKELY THE -0.075 TRAP)" \
            if wrong and n != "BASE" else ""
        print(f"    {n:<20} clus={r['clus']:.4f} ({r['clus']-base_clus:+.4f})  "
              f"len={r['length']:.4f} ({r['length']-base_len:+.4f}){flag}",
              flush=True)

    # ship gate: beat BASE on the ANCHOR by a non-noise margin, not worse on the
    # length lens, gap no worse than BASE.
    MARGIN = 0.005
    cand = [n for n, r in results.items() if n != "BASE"
            and r["clus"] > base_clus + MARGIN
            and r["length"] >= base_len - 0.002
            and r["clus_gap"] <= base_cgap + 1e-9]
    cand.sort(key=lambda n: -results[n]["clus"])

    if cand:
        winner = cand[0]
        r = results[winner]
        print(f"\n  DECISION: {winner} beats BASE on the CLUSTER anchor "
              f"(+{r['clus']-base_clus:.4f}) AND is not worse on the length lens "
              f"(+{r['length']-base_len:.4f}) with acceptable gap -> refit + write.",
              flush=True)
        _refit_and_save(winner, r, texts, y, toks, test_texts, test_ids, w_of_tok)
        ship = winner
    else:
        print("\n  DECISION: NULL RESULT. No config beats BASE on the CLUSTER "
              "anchor by a non-noise margin while holding the length lens.",
              flush=True)
        print("  Consistent with the prior: sparse-text models are already "
              "length-robust (deflate only -0.008); length adaptation does not "
              "help this family. Not writing a prediction file.", flush=True)
        ship = None

    print(f"\n  runtime: {time.time()-t0:.0f}s", flush=True)
    print(f"SHIP={ship if ship else 'none'}", flush=True)
    print("DONE_OK", flush=True)


def _refit_and_save(winner, r, texts, y, toks, test_texts, test_ids, w_of_tok):
    rng = np.random.default_rng(SEED)
    ttr, ytr, w = r["recipe"](texts, y, toks, rng)
    mv = MultiVec(r["factory"])
    X = mv.fit_transform(ttr)
    Xt = mv.transform(test_texts)
    clf = est_svc(0.25)
    clf.fit(X, ytr, sample_weight=w)
    pred = clf.predict(Xt).astype(int)
    out = OUT_DIR / "Task3_LengthAdapt_Prediction.csv"
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
    print(f"  WROTE {out}  rows={len(pred)} machine={int(pred.sum())} "
          f"human={int((pred == 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
