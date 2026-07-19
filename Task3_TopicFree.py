"""
Task 3 (agent H) — TOPIC-FREE representations (classical, sklearn only).
========================================================================
Hypothesis: the baseline's LB deflation comes from topic vocabulary that does
not transfer. Representations that CANNOT encode content words are
shift-robust by construction. We build them and measure on TWO lenses:

  lens A = 5-fold cluster-holdout (topic-shift proxy; anchor BASE = 0.7383)
  lens L = length-shifted holdout, shortest 60% train -> longest 40% val
           (anchor BASE = 0.8022)

Candidates (LinearSVC C in {0.25, 1}, class_weight=balanced only):
  FUNC    : function-word n-grams — keep only ENGLISH_STOP_WORDS + punctuation
            tokens, every other token -> 'X'; TF-IDF word 1-3 grams.
  CHCLASS : char-class stream (letter->a, digit->9, space->_, punct kept);
            TF-IDF char 2-5 grams.
  UNION   : FUNC + CHCLASS feature union.
  UNIONB  : UNION + the BASELINE's word(1,2)+char_wb(3,5) TF-IDF (topic +
            topic-free together).

Ship rule: only UNIONB may write a prediction file, and only if it beats BASE
on BOTH lenses with >=4/5 lens-A fold-wise wins.

Run: nohup .venv/bin/python Task3_TopicFree.py > scratch_topicfree.log 2>&1 &
"""

import re
import string
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from Task3_Improved_Model import DATA_DIR, OUT_DIR, SEED, cluster_folds

warnings.filterwarnings("ignore")

BASE_CLUS_TARGET = 0.7383
BASE_LEN_TARGET = 0.8022
REPRO_TOL = 0.003
SHORT_FRAC = 0.60
C_VALUES = [0.25, 1.0]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


# ---------------------------------------------------------------------------
# Topic-free view builders (deterministic text->text maps; no fitted state,
# so precomputing on all rows leaks nothing)
# ---------------------------------------------------------------------------
STOP = frozenset(ENGLISH_STOP_WORDS)
PUNCT = frozenset(string.punctuation)
TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|[^\w\s]")


def func_stream(text):
    """Keep stopwords + punctuation, replace everything else with 'X'."""
    out = []
    for t in TOKEN_RE.findall(str(text)):
        tl = t.lower()
        if tl in STOP:
            out.append(tl)
        elif len(t) == 1 and t in PUNCT:
            out.append(t)
        else:
            out.append("X")
    return " ".join(out)


CC_TABLE = {ord(c): "a" for c in string.ascii_letters}
CC_TABLE.update({ord(c): "9" for c in string.digits})
CC_TABLE.update({ord(c): "_" for c in " \t\n\r\x0b\x0c"})


def charclass_stream(text):
    """letter->a, digit->9, whitespace->_, punctuation kept verbatim."""
    s = str(text).translate(CC_TABLE)
    if not s.isascii():  # normalize rare non-ASCII chars the same way
        s = "".join(
            "a" if ch.isalpha() else "9" if ch.isdigit()
            else "_" if ch.isspace() else ch
            for ch in s)
    return s


# ---------------------------------------------------------------------------
# Vectorizer factories per view. A candidate = list of (view_name, factory).
# ---------------------------------------------------------------------------
def v_func():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                           sublinear_tf=True, token_pattern=r"\S+",
                           lowercase=False)


def v_chclass():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, lowercase=False)


def v_base_word():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True)


def v_base_char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           sublinear_tf=True)


CANDIDATES = {
    "BASE":    [("raw", v_base_word), ("raw", v_base_char)],
    "FUNC":    [("func", v_func)],
    "CHCLASS": [("cc", v_chclass)],
    "UNION":   [("func", v_func), ("cc", v_chclass)],
    "UNIONB":  [("func", v_func), ("cc", v_chclass),
                ("raw", v_base_word), ("raw", v_base_char)],
}


def eval_candidate(views, spec, y, folds, Cs):
    """Vectorize each fold once, fit LinearSVC per C. Returns
    {C: (per_fold_val, per_fold_train)}."""
    res = {C: ([], []) for C in Cs}
    for tr, val in folds:
        Xtr_parts, Xval_parts = [], []
        for view_name, fac in spec:
            v = fac()
            Xtr_parts.append(v.fit_transform(views[view_name][tr]))
            Xval_parts.append(v.transform(views[view_name][val]))
        Xtr = sparse.hstack(Xtr_parts).tocsr()
        Xval = sparse.hstack(Xval_parts).tocsr()
        for C in Cs:
            clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED)
            clf.fit(Xtr, y[tr])
            res[C][0].append(macro_f1(y[val], clf.predict(Xval)))
            res[C][1].append(macro_f1(y[tr], clf.predict(Xtr)))
    return res


def main():
    t0 = time.time()
    print("=" * 78)
    print("TASK 3 (agent H) — TOPIC-FREE representations, two-lens protocol")
    print("=" * 78, flush=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].to_numpy()
    print(f"train={len(texts)} test={len(test_texts)}", flush=True)

    # ---- precompute views ----
    tv = time.time()
    views = {
        "raw": texts,
        "func": np.array([func_stream(t) for t in texts], dtype=object),
        "cc": np.array([charclass_stream(t) for t in texts], dtype=object),
    }
    print(f"views built in {time.time()-tv:.0f}s", flush=True)
    print(f"  func sample: {views['func'][0][:160]}")
    print(f"  cc   sample: {views['cc'][0][:160]}", flush=True)

    # ---- lenses ----
    clus, cl = cluster_folds(texts, y)
    print(f"\nlens A cluster-holdout: {len(clus)} folds  "
          f"cluster sizes={np.bincount(cl).tolist()}", flush=True)
    toks = np.array([len(str(t).split()) for t in texts], dtype=float)
    q = np.quantile(toks, SHORT_FRAC)
    length_folds = [(np.where(toks <= q)[0], np.where(toks > q)[0])]
    print(f"lens L length holdout: cutoff={q:.0f} tok  "
          f"train={len(length_folds[0][0])} val={len(length_folds[0][1])}",
          flush=True)

    # ---- evaluate all candidates on both lenses ----
    results = {}  # name -> C -> dict(lensA folds, lensL, train)
    for name, spec in CANDIDATES.items():
        Cs = [0.25] if name == "BASE" else C_VALUES
        ta = time.time()
        resA = eval_candidate(views, spec, y, clus, Cs)
        resL = eval_candidate(views, spec, y, length_folds, Cs)
        results[name] = {}
        for C in Cs:
            a_folds = resA[C][0]
            lensA = float(np.mean(a_folds))
            lensL = float(resL[C][0][0])
            results[name][C] = dict(
                a_folds=a_folds, lensA=lensA, lensL=lensL,
                a_train=float(np.mean(resA[C][1])),
                l_train=float(resL[C][1][0]))
            print(f"{name:8s} C={C:<5} lensA={lensA:.4f} "
                  f"folds={[round(f,4) for f in a_folds]}  "
                  f"lensL={lensL:.4f}  L-A gap={lensL-lensA:+.4f}  "
                  f"(trainA={np.mean(resA[C][1]):.4f})", flush=True)
        print(f"  [{name} done in {time.time()-ta:.0f}s]", flush=True)

        # gate: anchor must reproduce before we trust anything else
        if name == "BASE":
            bA, bL = results["BASE"][0.25]["lensA"], results["BASE"][0.25]["lensL"]
            okA = abs(bA - BASE_CLUS_TARGET) <= REPRO_TOL
            okL = abs(bL - BASE_LEN_TARGET) <= REPRO_TOL
            print(f"  ANCHOR CHECK: lensA {bA:.4f} vs {BASE_CLUS_TARGET} "
                  f"({'OK' if okA else 'FAIL'}); lensL {bL:.4f} vs "
                  f"{BASE_LEN_TARGET} ({'OK' if okL else 'FAIL'})", flush=True)
            if not (okA and okL):
                print("ANCHOR FAILED TO REPRODUCE — ABORTING.", flush=True)
                return

    # ---- summary table ----
    base = results["BASE"][0.25]
    print("\n" + "=" * 78)
    print("SUMMARY (deltas vs BASE anchors: lensA 0.7383-run, lensL 0.8022-run)")
    print("=" * 78)
    print(f"{'cand':8s} {'C':>5} {'lensA':>7} {'dA':>8} {'lensL':>7} "
          f"{'dL':>8} {'L-A gap':>8} {'foldwins':>8}")
    for name in CANDIDATES:
        for C, r in results[name].items():
            wins = sum(f > b for f, b in zip(r["a_folds"], base["a_folds"]))
            print(f"{name:8s} {C:>5} {r['lensA']:.4f} "
                  f"{r['lensA']-base['lensA']:+.4f} {r['lensL']:.4f} "
                  f"{r['lensL']-base['lensL']:+.4f} "
                  f"{r['lensL']-r['lensA']:+.4f} {wins}/5")

    # ---- ship decision for UNIONB ----
    print("\nSHIP DECISION (UNIONB only):", flush=True)
    qualifiers = []
    for C, r in results["UNIONB"].items():
        wins = sum(f > b for f, b in zip(r["a_folds"], base["a_folds"]))
        beats = (r["lensA"] > base["lensA"] and r["lensL"] > base["lensL"]
                 and wins >= 4)
        print(f"  C={C}: lensA {r['lensA']:.4f} vs {base['lensA']:.4f}, "
              f"lensL {r['lensL']:.4f} vs {base['lensL']:.4f}, "
              f"foldwins {wins}/5 -> {'QUALIFIES' if beats else 'no'}",
              flush=True)
        if beats:
            qualifiers.append((C, r))

    if not qualifiers:
        print("  => UNIONB does not beat BASE on both lenses with >=4/5 "
              "fold wins. NOT writing a prediction file.", flush=True)
    else:
        # prefer the C with the higher lens-L score (lens-A-max is the trap);
        # tie-break lens A
        qualifiers.sort(key=lambda cr: (cr[1]["lensL"], cr[1]["lensA"]),
                        reverse=True)
        C_ship, r = qualifiers[0]
        print(f"  => SHIPPING UNIONB C={C_ship} "
              f"(lensA {r['lensA']:.4f}, lensL {r['lensL']:.4f})", flush=True)
        tviews = {
            "raw": test_texts,
            "func": np.array([func_stream(t) for t in test_texts], dtype=object),
            "cc": np.array([charclass_stream(t) for t in test_texts],
                           dtype=object),
        }
        Xp, Xtp = [], []
        for view_name, fac in CANDIDATES["UNIONB"]:
            v = fac()
            Xp.append(v.fit_transform(views[view_name]))
            Xtp.append(v.transform(tviews[view_name]))
        X = sparse.hstack(Xp).tocsr()
        Xt = sparse.hstack(Xtp).tocsr()
        clf = LinearSVC(C=C_ship, class_weight="balanced", random_state=SEED)
        clf.fit(X, y)
        pred = clf.predict(Xt).astype(int)
        out = OUT_DIR / "Task3_TopicFree_Prediction.csv"
        pd.DataFrame({"id": test_ids, "label": pred}).to_csv(out, index=False)
        chk = pd.read_csv(out)
        assert len(chk) == 6999 and (chk["id"].to_numpy() == test_ids).all()
        print(f"  wrote {out} rows={len(chk)} ids-match=True "
              f"machine={int(pred.sum())} ({pred.mean():.1%})", flush=True)

    print(f"\ntotal runtime {time.time()-t0:.0f}s")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
