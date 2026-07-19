"""
Task3_Composed.py — agent D (wave 2): do agent A's IW/sparse levers and agent C's
length levers (binary rep, crop-augmentation) STACK on top of A's winning config?

Base config = agent A's chosen config (Task3_IW_Tuned winner):
  LinearSVC C=1.0, class_weight=balanced
  word(1,3) + char_wb(2,5) TF-IDF, min_df=2, SUBLINEAR, transductive vocab
  (fit train+val), adversarial IW weights clipped [0.1, 10].
  -> agent A reported 5-fold cluster-holdout proxy = 0.7721.

Levers tested (factorial over A's config):
  a. A + binary=True   (binary=True, sublinear_tf=False, l2 norm — replaces sublinear)
  b. A + crop-aug      (each train doc + one random 190-tok contiguous crop; crop
                        gets its SOURCE ROW's adversarial IW weight)
  c. A + binary + crop-aug
  d. A + binary, C in {0.5, 2}
  e. crop-aug alone is (b) — run unconditionally.

DESIGN CHOICES (stated):
  * crop-aug: target = 190 whitespace tokens (train median), one crop per train
    doc (2x rows). Crops are seeded per fold (SEED+k). Crops reuse their source
    row's adversarial IW weight via concat(w, w) (w already mean-1 over N, so the
    2N vector is still mean-1 — no renorm needed; crops built in ttr order so the
    concat aligns).
  * transductive vocab is fit on (original train + val) ONLY, NOT on crops. Crops
    are word-subsets of train docs so add no new vocabulary; excluding them keeps
    the vocab/IDF identical to agent A's config. (Agent C measured crop-aug WITH
    crops in vocab on the OLD rep; a null here is null UNDER THIS design, not
    universal.)
  * binary rep keeps A's ngrams word(1,3)+char_wb(2,5) min_df=2 (NOT agent C's
    hardcoded word(1,2)+char(3,5)).
  * IW weights: adversarial (Task3_IW_Tuned machinery), clip (0.1,10). NOT the
    rejected length-quantile IW.

Protocol:
  STEP 1: reproduce A_base on all 5 folds; expect ~0.7721; hard-stop if >0.003 off.
  STEP 2: screen the other configs on folds 0,1 (A_base folds 0,1 reused from
          step 1 via the identical code path); full 5-fold for anything within
          0.005 of the screen leader.
  STEP 3: if any config's 5-fold mean > 0.7721, refit on all 20k (full-train
          adversarial IW, transductive vocab) and write
          predictions/Task3_Composed_Prediction.csv. Else write NO csv.

Run: nohup .venv/bin/python Task3_Composed.py > scratch_composed.log 2>&1 &
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, SEED, MultiVec, cluster_folds, est_svc, macro_f1,
)
from Task3_IW_Tuned import adv_p_train, weights_from_p
from Task3_LengthAdapt import crop_one

T0 = time.time()

# ---- A's fixed config ----
WORD_NG = (1, 3)
CHAR_NG = (2, 5)
MIN_DF = 2
CLIP = (0.1, 10.0)
CROP_TARGET = 190
A_TARGET = 0.7721
REPRO_TOL = 0.003
SCREEN_FOLDS = [0, 1]
WITHIN = 0.005   # full 5-fold for configs within this of the screen leader


def make_rep_factory(binary):
    """word(1,3)+char_wb(2,5), min_df=2. binary=True -> binary+l2 (no sublinear);
    binary=False -> A's sublinear rep."""
    sublinear = not binary
    def factory():
        return [
            TfidfVectorizer(analyzer="word", ngram_range=WORD_NG, min_df=MIN_DF,
                            binary=binary, sublinear_tf=sublinear, norm="l2"),
            TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NG, min_df=MIN_DF,
                            binary=binary, sublinear_tf=sublinear, norm="l2"),
        ]
    return factory


# ---- configs. rep in {"sub","bin"}; crop bool; C float ----
CONFIGS = [
    dict(name="A_base",        rep="sub", crop=False, C=1.0),   # reproduction / incumbent
    dict(name="A_crop",        rep="sub", crop=True,  C=1.0),   # b
    dict(name="A_bin",         rep="bin", crop=False, C=1.0),   # a
    dict(name="A_bin_crop",    rep="bin", crop=True,  C=1.0),   # c
    dict(name="A_bin_C0.5",    rep="bin", crop=False, C=0.5),   # d
    dict(name="A_bin_C2",      rep="bin", crop=False, C=2.0),   # d
]
BY_NAME = {c["name"]: c for c in CONFIGS}

# results[name][fold_k] = f1
results = {c["name"]: {} for c in CONFIGS}
adv_p_cache = {}


def make_crops(ttr, k):
    rng = np.random.default_rng(SEED + k)
    return np.array([crop_one(t, CROP_TARGET, rng) for t in ttr], dtype=object)


def eval_fold(k, folds, texts, y, config_names):
    """Evaluate the named configs on fold k. Groups by representation so each
    rep is vectorized once; caches adv_p (identical clip -> identical w)."""
    tr, val = folds[k]
    ttr, tval = texts[tr], texts[val]
    ytr, yval = y[tr], y[val]

    if k not in adv_p_cache:
        t = time.time()
        adv_p_cache[k] = adv_p_train(ttr, tval)
        print(f"  [fold {k}] adversarial OOF p computed ({time.time()-t:.0f}s)",
              flush=True)
    w = weights_from_p(adv_p_cache[k], CLIP)          # mean-1 over N train rows

    todo = [BY_NAME[n] for n in config_names if k not in results[n]]
    if not todo:
        return
    crops = None
    if any(c["crop"] for c in todo):
        crops = make_crops(ttr, k)
    ttr_aug = np.concatenate([np.asarray(ttr, dtype=object), crops]) \
        if crops is not None else None
    w_aug = np.concatenate([w, w]) if crops is not None else None

    for binflag in (False, True):
        group = [c for c in todo if (c["rep"] == "bin") == binflag]
        if not group:
            continue
        t = time.time()
        mv = MultiVec(make_rep_factory(binflag))
        mv.fit(np.concatenate([ttr, tval]))           # transductive vocab: originals+val
        Xval = mv.transform(tval)
        Xtr = mv.transform(ttr)
        Xtr_aug = mv.transform(ttr_aug) if ttr_aug is not None else None
        tvec = time.time() - t
        for c in group:
            t = time.time()
            clf = est_svc(c["C"])
            if c["crop"]:
                clf.fit(Xtr_aug, np.concatenate([ytr, ytr]), sample_weight=w_aug)
            else:
                clf.fit(Xtr, ytr, sample_weight=w)
            s = macro_f1(yval, clf.predict(Xval))
            results[c["name"]][k] = s
            print(f"  [fold {k}] {c['name']:<12} rep={'bin' if binflag else 'sub'} "
                  f"crop={c['crop']} C={c['C']} -> {s:.4f} "
                  f"(vec {tvec:.0f}s, fit {time.time()-t:.0f}s)", flush=True)
        del Xtr, Xval, Xtr_aug, mv


def mean_folds(name, ks):
    return float(np.mean([results[name][k] for k in ks]))


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    print(f"train {len(texts)} | test {len(test_texts)}", flush=True)

    folds, _ = cluster_folds(texts, y)
    n_folds = len(folds)
    print(f"cluster-holdout folds: {n_folds} "
          f"(val sizes {[len(v) for _, v in folds]})", flush=True)

    # ===== STEP 1: reproduce A_base on all folds (same code path as screen) =====
    print("\n===== STEP 1: reproduce agent A config on all 5 folds =====",
          flush=True)
    for k in range(n_folds):
        eval_fold(k, folds, texts, y, ["A_base"])
    a_mean = mean_folds("A_base", range(n_folds))
    a_perfold = [round(results["A_base"][k], 4) for k in range(n_folds)]
    print(f"REPRO A_base mean={a_mean:.4f} (target {A_TARGET})  per-fold {a_perfold}",
          flush=True)
    if abs(a_mean - A_TARGET) > REPRO_TOL:
        print(f"REPRODUCTION FAILED |{a_mean:.4f}-{A_TARGET}| > {REPRO_TOL} "
              "— stopping, no screen, no predictions.", flush=True)
        print("DONE (repro-failed)", flush=True)
        return
    print(f"reproduction OK (|diff|={abs(a_mean-A_TARGET):.4f} <= {REPRO_TOL})",
          flush=True)

    # ===== STEP 2a: screen other configs on folds 0,1 =====
    print("\n===== STEP 2a: factorial screen (folds 0,1) =====", flush=True)
    screen_names = [c["name"] for c in CONFIGS]  # A_base 0,1 already computed
    for k in SCREEN_FOLDS:
        eval_fold(k, folds, texts, y, screen_names)

    print("\n--- screen means (folds 0,1) ---", flush=True)
    ranked = sorted(screen_names, key=lambda n: -mean_folds(n, SCREEN_FOLDS))
    for n in ranked:
        print(f"  {mean_folds(n, SCREEN_FOLDS):.4f}  {n}", flush=True)
    leader = mean_folds(ranked[0], SCREEN_FOLDS)

    # ===== STEP 2b: full 5-fold for configs within 0.005 of the screen leader =====
    finalists = [n for n in screen_names
                 if mean_folds(n, SCREEN_FOLDS) >= leader - WITHIN]
    print(f"\n===== STEP 2b: full 5-fold on finalists "
          f"(within {WITHIN} of screen leader {leader:.4f}) =====", flush=True)
    print(f"finalists: {finalists}", flush=True)
    for k in range(n_folds):
        if k in SCREEN_FOLDS:
            continue
        eval_fold(k, folds, texts, y, finalists)

    print("\n--- FULL 5-FOLD RESULTS ---", flush=True)
    full = {}
    for n in finalists:
        if all(k in results[n] for k in range(n_folds)):
            full[n] = mean_folds(n, range(n_folds))
            pf = [round(results[n][k], 4) for k in range(n_folds)]
            print(f"  mean={full[n]:.4f}  per-fold {pf}  {n} "
                  f"(vs reproduced A_base {a_mean:.4f}: {full[n]-a_mean:+.4f})",
                  flush=True)

    win_name = max(full, key=full.get)
    win_mean = full[win_name]
    print(f"\nBEST full-5-fold: {win_name} = {win_mean:.4f}  "
          f"(A_target {A_TARGET}, delta {win_mean-A_TARGET:+.4f}; "
          f"vs reproduced A_base {a_mean:.4f}, delta {win_mean-a_mean:+.4f})",
          flush=True)

    # ===== STEP 3: ship only if best 5-fold mean > 0.7721 =====
    if win_mean <= A_TARGET:
        print("\nVERDICT: DID NOT STACK. No config's 5-fold mean beats "
              f"{A_TARGET}. Writing NO prediction file.", flush=True)
        print(f"PROJECTED LB of A_base (proxy-0.008): {a_mean-0.008:.4f}", flush=True)
        print(f"Total runtime {time.time()-T0:.0f}s", flush=True)
        print("ALL DONE (null)", flush=True)
        return

    win = BY_NAME[win_name]
    print(f"\nVERDICT: STACKED. {win_name} beats {A_TARGET} "
          f"(+{win_mean-A_TARGET:.4f}). Refitting on full 20k.", flush=True)
    print(f"PROJECTED LB (proxy-0.008): {win_mean-0.008:.4f}", flush=True)

    # ===== final fit on all train rows =====
    print("\n===== STEP 3: final fit on full train, predict test =====", flush=True)
    t = time.time()
    p_full = adv_p_train(texts, test_texts)
    w_full = weights_from_p(p_full, CLIP)
    print(f"  full-train adversarial weights done ({time.time()-t:.0f}s); "
          f"min={w_full.min():.3f} med={np.median(w_full):.3f} "
          f"max={w_full.max():.3f}", flush=True)

    mv = MultiVec(make_rep_factory(win["rep"] == "bin"))
    mv.fit(np.concatenate([texts, test_texts]))       # transductive vocab
    Xte = mv.transform(test_texts)
    if win["crop"]:
        rng = np.random.default_rng(SEED)
        crops = np.array([crop_one(t, CROP_TARGET, rng) for t in texts],
                         dtype=object)
        Xtr = mv.transform(np.concatenate([np.asarray(texts, dtype=object), crops]))
        ytr = np.concatenate([y, y])
        w_fit = np.concatenate([w_full, w_full])
        print(f"  crop-aug final fit: {Xtr.shape[0]} train rows "
              f"(feat dim {Xtr.shape[1]})", flush=True)
    else:
        Xtr = mv.transform(texts)
        ytr = y
        w_fit = w_full
    clf = est_svc(win["C"])
    clf.fit(Xtr, ytr, sample_weight=w_fit)
    pred = clf.predict(Xte).astype(int)

    out = pd.DataFrame({"id": test["id"], "label": pred})
    assert len(out) == 6999, f"expected 6999 rows, got {len(out)}"
    assert (out["id"].to_numpy() == test["id"].to_numpy()).all(), "id mismatch"
    path = OUT_DIR / "Task3_Composed_Prediction.csv"
    out.to_csv(path, index=False)
    print(f"WROTE {path}  rows={len(out)}  machine={int(out['label'].sum())}  "
          f"human={int((out['label']==0).sum())}", flush=True)
    print(f"Total runtime {time.time()-T0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
