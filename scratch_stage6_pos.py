"""
STAGE 6 — heuristic pseudo-POS morpho-syntactic leg: a NEW orthogonal signal.
============================================================================
No tagger available, so approximate POS from morphology: each content word -> a
grammatical class by suffix/casing (VBG/VBD/RB/nominalization/JJ/NNP/NNS/NN),
function words -> 'FW', numbers -> 'CD', punctuation kept. N-gram the TAG SEQUENCE.
This captures the syntactic-category RHYTHM of content words — orthogonal to:
  - content char/word TF-IDF (lexical), - fw skeleton (masks all content as '#'),
  - stylo aggregate rates. A classic authorship/AI syntactic signal.

Key test: does POS add ON TOP OF Iter-17 + fw (is there orthogonal signal beyond fw)?
Candidates (four lenses, IW + frac0.5 self-train):
  iter17            [context ref]
  iter17_fw         Iter-17 + function-word skeleton x1.0   (= queued Iter-20)
  iter17_fw_pos     + pseudo-POS n-grams x{0.5,1.0}         (does it add beyond fw?)
Judge on topical A/B/C1; a real WIN = pos adds >~0.004 topical over iter17_fw. No DL.
"""
import re, time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1
from scratch_lensC_combine import lensC1_folds, lensC2_folds, build_dense
from scratch_stage2_features import build_stack, build_bank, clf_
from scratch_stage1_transduce import iw_weights, self_train

SEED = 42
t0 = time.time()
rng = np.random.RandomState(SEED)
BANK_S, STYLO_S, FW_W = 0.02, 0.04, 1.0
POS_W = [0.5, 1.0]
CANDS = ["iter17", "iter17_fw"] + [f"fw_pos{w}" for w in POS_W]
REF = "iter17_fw"

_tok = re.compile(r"[A-Za-z']+|[.,;:!?()\-]|\d+")
_FW = set(w.lower() for w in ENGLISH_STOP_WORDS)
_END = {".", "!", "?"}


def skeleton(t):
    return " ".join(m.lower() if (m[:1].isalpha() and m.lower() in _FW) else ("#" if m[:1].isalpha() else m)
                    for m in _tok.findall(t))


def pos_tag(w, sent_start):
    if w.isdigit():
        return "CD"
    if not w[0].isalpha():
        return w                                   # punctuation kept as itself
    wl = w.lower()
    if wl in _FW:
        return "FW"
    if wl.endswith("ing"): return "VBG"
    if wl.endswith("ed"): return "VBD"
    if wl.endswith("ly"): return "RB"
    if wl.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence")): return "NNZ"
    if wl.endswith(("ize", "ise", "ate", "ify")): return "VB"
    if wl.endswith(("al", "ous", "ive", "ful", "able", "ible", "ic", "ary")): return "JJ"
    if wl.endswith(("er", "or")): return "NNR"
    if w[0].isupper() and not sent_start: return "NNP"
    if wl.endswith("s"): return "NNS"
    return "NN"


def pos_seq(t):
    out, sent_start = [], True
    for m in _tok.findall(t):
        out.append(pos_tag(m, sent_start))
        sent_start = m in _END
    return " ".join(out)


def eval_lens(name, folds, texts, Y, Dsty, Skel, Pos):
    acc = {c: [] for c in CANDS}
    for fi, (tr, val) in enumerate(folds):
        tr = np.asarray(tr); val = np.asarray(val)
        rng.shuffle(val); h = len(val) // 2
        pool_idx, eval_idx = val[:h], val[h:]
        Xtr, (Xp, Xe) = build_stack(texts[tr], [texts[pool_idx], texts[eval_idx]])
        ytr, yev = Y[tr], Y[eval_idx]
        w, _ = iw_weights(texts[tr], texts[val])
        Btr, (Bp, Be) = build_bank(texts[tr], ytr, [texts[pool_idx], texts[eval_idx]])
        sb = StandardScaler().fit(Btr)
        bk = lambda D: sparse.csr_matrix(sb.transform(D) * BANK_S)
        ss = StandardScaler().fit(Dsty[tr])
        stf = lambda idx: sparse.csr_matrix(ss.transform(Dsty[idx]) * STYLO_S)
        I17_tr = sparse.hstack([Xtr, bk(Btr), stf(tr)]).tocsr()
        I17_p = sparse.hstack([Xp, bk(Bp), stf(pool_idx)]).tocsr()
        I17_e = sparse.hstack([Xe, bk(Be), stf(eval_idx)]).tocsr()

        fv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
        fw = [fv.fit_transform(Skel[tr]).astype(np.float32) * FW_W,
              fv.transform(Skel[pool_idx]).astype(np.float32) * FW_W,
              fv.transform(Skel[eval_idx]).astype(np.float32) * FW_W]
        pv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True, token_pattern=r"[^ ]+")
        P = [pv.fit_transform(Pos[tr]).astype(np.float32),
             pv.transform(Pos[pool_idx]).astype(np.float32),
             pv.transform(Pos[eval_idx]).astype(np.float32)]

        def run(extra_tr, extra_p, extra_e):
            return self_train(sparse.hstack([I17_tr] + extra_tr).tocsr(), ytr,
                              sparse.hstack([I17_p] + extra_p).tocsr(),
                              sparse.hstack([I17_e] + extra_e).tocsr(), yev,
                              frac=0.5, rounds=1, w_tr=w)

        acc["iter17"].append(run([], [], []))
        acc["iter17_fw"].append(run([fw[0]], [fw[1]], [fw[2]]))
        for pw in POS_W:
            acc[f"fw_pos{pw}"].append(run([fw[0], P[0] * pw], [fw[1], P[1] * pw], [fw[2], P[2] * pw]))
        print(f"  [{name}] f{fi} " + " ".join(f"{c}={acc[c][-1]:.4f}" for c in CANDS) +
              f" ({time.time()-t0:.0f}s)", flush=True)
    return {c: np.array(v) for c, v in acc.items()}


def main():
    texts, Y, test_texts, _ = load_data()
    Dsty = build_dense(texts)
    Skel = np.array([skeleton(t) for t in texts], dtype=object)
    Pos = np.array([pos_seq(t) for t in texts], dtype=object)
    print(f"precomputed ({time.time()-t0:.0f}s)  pos example: {pos_seq(texts[0])[:90]}", flush=True)
    foldsA, foldsB = get_folds()
    lenses = [("A", foldsA), ("B", foldsB),
              ("C1", lensC1_folds(texts, Y)),
              ("C2", lensC2_folds(texts, test_texts, Y)[0])]
    res = {n: eval_lens(n, f, texts, Y, Dsty, Skel, Pos) for n, f in lenses}

    print(f"\n===== Δ vs {REF} (topical A/B/C1 primary) =====", flush=True)
    print(f"{'candidate':12s}{'A':>9s}{'B':>9s}{'C1':>9s}{'C2':>9s}{'topical_min':>12s}", flush=True)
    for c in CANDS:
        d = [float((res[n][c] - res[n][REF]).mean()) for n, _ in lenses]
        print(f"{c:12s}" + "".join(f"{x:+9.4f}" for x in d) + f"{min(d[:3]):+12.4f}", flush=True)
    print("\nvs iter17 (does the WHOLE fw+pos stack beat Iter-17?):", flush=True)
    for c in CANDS:
        d = [float((res[n][c] - res[n]["iter17"]).mean()) for n, _ in lenses]
        print(f"  {c:12s} topical_min {min(d[:3]):+.4f}  (A{d[0]:+.4f} B{d[1]:+.4f} C1{d[2]:+.4f} C2{d[3]:+.4f})", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
