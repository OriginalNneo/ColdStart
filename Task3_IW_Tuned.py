"""
Task3_IW_Tuned.py — overnight grid squeeze of the IW + sparse-TF-IDF + LinearSVC
family (agent A track "IW-tuned SVC").

Protocol (fixed):
  * Folds: the SEED=42 cluster-holdout folds from Task3_Improved_Model.cluster_folds
    (identical import, identical call -> identical folds).
  * IW machinery: identical math to Task3_PseudoLabel.adversarial_train_weights
    (char_wb(3,5) min_df=5 max_features=200k sublinear adversarial TF-IDF,
    LogisticRegression C=1 OOF via cross_val_predict cv=3, odds ratio p/(1-p),
    clip, normalize to mean 1). Clip is the only parameter varied.
  * Step 1: reproduce BASE=0.7383 and IW=0.7523 on all 5 folds; hard-stop if off.
  * Step 2: screen one-at-a-time variants around the incumbent on folds 0+1,
    then greedy-combo stage, then full 5-fold on the top finalists
    (folds 0/1 scores are reused; only folds 2-4 are added — same protocol).
  * Step 3: train the best config on all 20k rows (IW weights from full
    train-vs-test adversarial OOF) and write predictions/Task3_IW_Tuned_Prediction.csv.

min_df is applied to BOTH the word and char vectorizers (matching the baseline,
which uses min_df=2 on both).

Run: nohup .venv/bin/python Task3_IW_Tuned.py > scratch_iw_tuned.log 2>&1 &
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from Task3_Improved_Model import (
    DATA_DIR, OUT_DIR, SEED, MultiVec, cluster_folds, est_svc, macro_f1,
    vec_word_char,
)

T0 = time.time()

REPRO_BASE = 0.7383
REPRO_IW = 0.7523
REPRO_TOL = 0.0015

INCUMBENT = dict(C=0.25, min_df=2, word_ng=(1, 2), char_ng=(3, 5),
                 clip=(0.25, 4.0), trans=True)

GRID = dict(
    C=[0.1, 0.25, 0.5, 1.0],
    min_df=[2, 3, 5],
    word_ng=[(1, 2), (1, 3)],
    char_ng=[(3, 5), (2, 5), (3, 6)],
    clip=[(0.25, 4.0), (0.1, 10.0), (0.5, 2.0)],
    trans=[True, False],
)

SCREEN_FOLDS = [0, 1]
N_FINALISTS = 3


def key_of(cfg):
    return (cfg["C"], cfg["min_df"], cfg["word_ng"], cfg["char_ng"],
            cfg["clip"], cfg["trans"])


def fmt(cfg):
    return (f"C={cfg['C']} min_df={cfg['min_df']} word={cfg['word_ng']} "
            f"char={cfg['char_ng']} clip={cfg['clip']} trans={cfg['trans']}")


def make_factory(min_df, word_ng, char_ng):
    def factory():
        return [
            TfidfVectorizer(analyzer="word", ngram_range=word_ng,
                            min_df=min_df, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng,
                            min_df=min_df, sublinear_tf=True),
        ]
    return factory


# ---- IW machinery: identical to Task3_PseudoLabel.adversarial_train_weights,
# split so the expensive OOF probabilities are computed once per fold and the
# cheap clip variants reuse them.
def adv_p_train(texts_tr, texts_un):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                          max_features=200000, sublinear_tf=True)
    Xa = vec.fit_transform(np.concatenate([texts_tr, texts_un]))
    d = np.r_[np.zeros(len(texts_tr)), np.ones(len(texts_un))]
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    p = cross_val_predict(lr, Xa, d, cv=3, method="predict_proba")[:, 1]
    return np.clip(p[:len(texts_tr)], 1e-3, 1 - 1e-3)


def weights_from_p(p_tr, clip):
    w = np.clip(p_tr / (1 - p_tr), *clip)
    return w * (len(w) / w.sum())


def eval_cfgs_on_fold(cfgs, k, texts, y, folds, adv_p_cache, results):
    """Evaluate a list of configs on fold k, grouping by representation so each
    (min_df, ngrams, trans) is vectorized once. Fills results[key][k]."""
    tr, val = folds[k]
    ttr, tval, ytr, yval = texts[tr], texts[val], y[tr], y[val]
    if k not in adv_p_cache:
        t = time.time()
        adv_p_cache[k] = adv_p_train(ttr, tval)
        print(f"  [fold {k}] adversarial OOF p computed ({time.time()-t:.0f}s)",
              flush=True)
    todo = [c for c in cfgs if k not in results.setdefault(key_of(c), {})]
    by_rep = {}
    for c in todo:
        rep = (c["min_df"], c["word_ng"], c["char_ng"], c["trans"])
        by_rep.setdefault(rep, []).append(c)
    for rep, group in by_rep.items():
        min_df, wng, cng, trans = rep
        t = time.time()
        mv = MultiVec(make_factory(min_df, wng, cng))
        if trans:
            Xall = mv.fit_transform(np.concatenate([ttr, tval]))
            Xtr, Xval = Xall[:len(tr)], Xall[len(tr):]
        else:
            Xtr = mv.fit_transform(ttr)
            Xval = mv.transform(tval)
        tvec = time.time() - t
        for c in group:
            t = time.time()
            w = weights_from_p(adv_p_cache[k], c["clip"])
            clf = est_svc(c["C"])
            clf.fit(Xtr, ytr, sample_weight=w)
            s = macro_f1(yval, clf.predict(Xval))
            results[key_of(c)][k] = s
            print(f"  [fold {k}] {fmt(c)}  ->  {s:.4f}   "
                  f"(vec {tvec:.0f}s, fit {time.time()-t:.0f}s)", flush=True)
        del Xtr, Xval, mv


def screen_mean(results, key):
    return float(np.mean([results[key][k] for k in SCREEN_FOLDS]))


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    print(f"train {len(texts)} | test {len(test_texts)}", flush=True)

    folds, _ = cluster_folds(texts, y)
    print(f"cluster-holdout folds: {len(folds)} "
          f"(val sizes {[len(v) for _, v in folds]})", flush=True)

    adv_p_cache = {}

    # ================= STEP 1: REPRODUCTION =================
    print("\n===== STEP 1: reproduce BASE and IW on all folds =====", flush=True)
    base_scores, iw_scores = [], []
    for k, (tr, val) in enumerate(folds):
        ttr, tval, ytr, yval = texts[tr], texts[val], y[tr], y[val]
        # BASE: train-only vocab, no weights
        mv = MultiVec(vec_word_char)
        Xtr = mv.fit_transform(ttr)
        Xval = mv.transform(tval)
        clf = est_svc(0.25)
        clf.fit(Xtr, ytr)
        base_scores.append(macro_f1(yval, clf.predict(Xval)))
        del Xtr, Xval, mv
        # IW: transductive vocab + adversarial weights clip (0.25, 4)
        mvt = MultiVec(vec_word_char)
        Xall = mvt.fit_transform(np.concatenate([ttr, tval]))
        Xtr_t, Xval_t = Xall[:len(tr)], Xall[len(tr):]
        adv_p_cache[k] = adv_p_train(ttr, tval)
        w = weights_from_p(adv_p_cache[k], (0.25, 4.0))
        clf = est_svc(0.25)
        clf.fit(Xtr_t, ytr, sample_weight=w)
        iw_scores.append(macro_f1(yval, clf.predict(Xval_t)))
        del Xtr_t, Xval_t, Xall, mvt
        print(f"  fold {k}: BASE={base_scores[-1]:.4f}  IW={iw_scores[-1]:.4f}",
              flush=True)
    base_mean, iw_mean = float(np.mean(base_scores)), float(np.mean(iw_scores))
    print(f"REPRO BASE mean={base_mean:.4f} (target {REPRO_BASE})  "
          f"per-fold {[round(s,4) for s in base_scores]}", flush=True)
    print(f"REPRO IW   mean={iw_mean:.4f} (target {REPRO_IW})  "
          f"per-fold {[round(s,4) for s in iw_scores]}", flush=True)
    if abs(base_mean - REPRO_BASE) > REPRO_TOL or abs(iw_mean - REPRO_IW) > REPRO_TOL:
        print("REPRODUCTION FAILED — stopping without tuning or predictions.")
        print("DONE (repro-failed)", flush=True)
        return

    results = {key_of(INCUMBENT): {k: s for k, s in enumerate(iw_scores)}}

    # ================= STEP 2a: one-at-a-time screen on folds 0,1 ===========
    print("\n===== STEP 2a: one-at-a-time screen (folds 0,1) =====", flush=True)
    stage_a = [dict(INCUMBENT)]
    for dim, values in GRID.items():
        for v in values:
            if v == INCUMBENT[dim]:
                continue
            c = dict(INCUMBENT)
            c[dim] = v
            stage_a.append(c)
    seen = set()
    stage_a = [c for c in stage_a
               if key_of(c) not in seen and not seen.add(key_of(c))]
    print(f"stage A configs: {len(stage_a)}", flush=True)
    for k in SCREEN_FOLDS:
        eval_cfgs_on_fold(stage_a, k, texts, y, folds, adv_p_cache, results)

    print("\n--- stage A screen means (folds 0,1) ---", flush=True)
    for c in sorted(stage_a, key=lambda c: -screen_mean(results, key_of(c))):
        print(f"  {screen_mean(results, key_of(c)):.4f}  {fmt(c)}", flush=True)

    # ================= STEP 2b: greedy combo stage ==========================
    print("\n===== STEP 2b: combo screen (folds 0,1) =====", flush=True)
    best_val, second_c = {}, None
    for dim, values in GRID.items():
        def sc(v):
            c = dict(INCUMBENT)
            c[dim] = v
            return screen_mean(results, key_of(c))
        ranked = sorted(values, key=lambda v: -sc(v))
        best_val[dim] = ranked[0]
        if dim == "C":
            second_c = ranked[1]
        print(f"  best {dim}: {ranked[0]} "
              f"({', '.join(f'{v}:{sc(v):.4f}' for v in values)})", flush=True)

    combo = {dim: best_val[dim] for dim in GRID}
    stage_b, bseen = [], set(key_of(c) for c in stage_a)
    for cand in [combo,
                 {**combo, "C": second_c},
                 {**combo, "trans": not combo["trans"]},
                 {**combo, "clip": INCUMBENT["clip"]}]:
        if key_of(cand) not in bseen:
            bseen.add(key_of(cand))
            stage_b.append(cand)
    print(f"stage B configs: {len(stage_b)}", flush=True)
    for k in SCREEN_FOLDS:
        eval_cfgs_on_fold(stage_b, k, texts, y, folds, adv_p_cache, results)
    for c in stage_b:
        print(f"  {screen_mean(results, key_of(c)):.4f}  {fmt(c)}", flush=True)

    # ================= STEP 2c: full 5-fold on finalists ====================
    print("\n===== STEP 2c: full 5-fold on finalists =====", flush=True)
    all_cfgs = {key_of(c): c for c in stage_a + stage_b}
    ranked = sorted(all_cfgs.values(),
                    key=lambda c: -screen_mean(results, key_of(c)))
    finalists = [c for c in ranked if key_of(c) != key_of(INCUMBENT)][:N_FINALISTS]
    print("finalists (screen mean, config):", flush=True)
    for c in finalists:
        print(f"  {screen_mean(results, key_of(c)):.4f}  {fmt(c)}", flush=True)
    for k in range(len(folds)):
        if k in SCREEN_FOLDS:
            continue
        eval_cfgs_on_fold(finalists, k, texts, y, folds, adv_p_cache, results)

    print("\n--- FULL 5-FOLD RESULTS ---", flush=True)
    contenders = finalists + [INCUMBENT]
    full = {}
    for c in contenders:
        ks = sorted(results[key_of(c)])
        scores = [results[key_of(c)][k] for k in ks]
        if len(scores) == len(folds):
            full[key_of(c)] = float(np.mean(scores))
            print(f"  mean={np.mean(scores):.4f}  per-fold "
                  f"{[round(s,4) for s in scores]}  {fmt(c)}", flush=True)

    win_key = max(full, key=full.get)
    winner = next(c for c in contenders if key_of(c) == win_key)
    print(f"\nWINNER: {fmt(winner)}  full-5-fold mean={full[win_key]:.4f}  "
          f"(incumbent IW={iw_mean:.4f}, delta {full[win_key]-iw_mean:+.4f})",
          flush=True)
    print(f"PROJECTED LB (proxy-0.008): {full[win_key]-0.008:.4f}", flush=True)

    # ================= STEP 3: final fit on all train rows ==================
    print("\n===== STEP 3: final fit on full train, predict test =====",
          flush=True)
    t = time.time()
    p_full = adv_p_train(texts, test_texts)
    w_full = weights_from_p(p_full, winner["clip"])
    print(f"  full-train adversarial weights done ({time.time()-t:.0f}s); "
          f"weight stats min={w_full.min():.3f} med={np.median(w_full):.3f} "
          f"max={w_full.max():.3f}", flush=True)
    mv = MultiVec(make_factory(winner["min_df"], winner["word_ng"],
                               winner["char_ng"]))
    if winner["trans"]:
        Xall = mv.fit_transform(np.concatenate([texts, test_texts]))
        Xtr_f, Xte_f = Xall[:len(texts)], Xall[len(texts):]
    else:
        Xtr_f = mv.fit_transform(texts)
        Xte_f = mv.transform(test_texts)
    clf = est_svc(winner["C"])
    clf.fit(Xtr_f, y, sample_weight=w_full)
    pred = clf.predict(Xte_f).astype(int)

    out = pd.DataFrame({"id": test["id"], "label": pred})
    assert len(out) == 6999, f"expected 6999 rows, got {len(out)}"
    assert (out["id"].to_numpy() == test["id"].to_numpy()).all(), "id mismatch"
    path = OUT_DIR / "Task3_IW_Tuned_Prediction.csv"
    out.to_csv(path, index=False)
    print(f"WROTE {path}  rows={len(out)}  machine={int(out['label'].sum())}  "
          f"human={int((out['label']==0).sum())}", flush=True)
    print(f"Total runtime {time.time()-T0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
