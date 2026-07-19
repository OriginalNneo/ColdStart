"""
Track 5 — TOPIC-INVARIANT STYLOMETRY (classical ML only; NO deep learning).
============================================================================
Hypothesis: function-word usage + surface style (punctuation, sentence-length
regularity, char-class mix, lexical diversity) do NOT depend on the research
topic, so they should transfer across the train->test TOPIC SHIFT that sinks
vanilla CV here. We test:

  (1) a DENSE topic-invariant feature block (extends Task3's stylometric()),
  (2) that block ALONE with LogReg(balanced) and LinearSVC(balanced), reporting
      the train->val GAP (small gap = robust, even if absolute F1 is weak),
  (3) FUSION: scale+concat the dense block with the capped wideB sparse rep,
      LinearSVC(C=0.25, balanced), sweeping the dense-block scale in {0.3,1,3}.

Judged on BOTH topic-shift lenses from scratch_lens (Lens A word-unigram KMeans,
Lens B char_wb(3,5) KMeans). PASS = beats ANCHOR on BOTH lenses.

Dense features need per-fold scaling, so we write our OWN fold loop:
StandardScaler is fit on TRAIN ROWS ONLY; val labels are never touched.

Does NOT edit any protected file. Outputs: stdout + (if PASS) scratch_agent5_pred.csv
"""
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scratch_lens import load_data, get_folds, macro_f1, ANCHOR

warnings.filterwarnings("ignore")
SEED = 42

# ---------------------------------------------------------------------------
# 1. TOPIC-INVARIANT DENSE FEATURE BLOCK  (extends Task3_Improved_Model.stylometric)
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENT_RE = re.compile(r"[.!?]+")

# ~180 English function words / high-freq closed-class tokens. These are
# topic-agnostic: authors (human vs machine) differ in HOW they glue content
# together, not in which content words the abstract's topic forces on them.
FUNCTION_WORDS = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "as", "are",
    "with", "on", "by", "this", "we", "be", "an", "which", "it", "from", "at",
    "or", "our", "not", "can", "these", "have", "has", "was", "were", "been",
    "their", "its", "such", "also", "more", "most", "than", "then", "thus",
    "however", "therefore", "moreover", "furthermore", "hence", "while",
    "whereas", "although", "though", "because", "since", "when", "where",
    "how", "what", "who", "whom", "whose", "why", "if", "but", "so", "yet",
    "both", "either", "neither", "each", "every", "any", "all", "some", "no",
    "none", "one", "two", "three", "many", "much", "few", "several", "other",
    "another", "same", "different", "new", "first", "second", "third", "last",
    "next", "previous", "here", "there", "now", "thereby", "therein",
    "thereof", "wherein", "toward", "towards", "upon", "within", "without",
    "into", "onto", "over", "under", "above", "below", "between", "among",
    "through", "throughout", "during", "before", "after", "against", "about",
    "across", "along", "around", "behind", "beyond", "despite", "per", "via",
    "given", "based", "using", "used", "shown", "show", "shows", "propose",
    "proposed", "present", "presents", "presented", "demonstrate", "achieve",
    "achieves", "achieved", "results", "result", "method", "methods",
    "approach", "model", "models", "may", "might", "could", "would", "should",
    "must", "shall", "will", "cannot", "do", "does", "did", "being", "having",
    "they", "them", "he", "she", "his", "her", "you", "your", "i", "my", "us",
    "me", "himself", "themselves", "itself", "each", "very", "well", "only",
    "even", "still", "just", "further", "particularly", "especially",
    "generally", "typically", "specifically", "respectively", "namely",
    "e.g", "i.e", "et", "al",
]
FUNCTION_WORDS = list(dict.fromkeys(FUNCTION_WORDS))  # dedupe, keep order
FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}
NFW = len(FUNCTION_WORDS)

# Hedging / epistemic-stance markers (topic-invariant author style).
HEDGES = {"may", "might", "could", "possibly", "perhaps", "suggest",
          "suggests", "appear", "appears", "seem", "seems", "likely",
          "potentially", "probably", "arguably", "presumably", "relatively",
          "somewhat", "generally", "typically", "often", "usually"}
# "be" forms + "by": crude passive-voice proxy (topic-invariant).
BE_FORMS = {"be", "been", "being", "is", "are", "was", "were", "am"}

STRUCT_NAMES = [
    "avg_word_len", "word_len_std", "type_token_ratio", "hapax_ratio",
    "dis_legomena_ratio", "unique_bigram_ratio", "avg_sent_len",
    "sent_len_std", "sent_len_cv", "n_sentences_per_100w",
    "comma_density", "period_density", "semicolon_density", "colon_density",
    "paren_density", "quote_density", "dash_density", "question_density",
    "exclaim_density", "digit_ratio", "upper_ratio", "alpha_ratio",
    "space_ratio", "punct_ratio", "short_word_ratio", "long_word_ratio",
    "hedge_ratio", "be_ratio", "by_ratio", "stopword_mass",
]
NSTRUCT = len(STRUCT_NAMES)
FEATURE_NAMES = [f"fw::{w}" for w in FUNCTION_WORDS] + STRUCT_NAMES


def _features(text):
    text = str(text)
    n_chars = len(text)
    words = WORD_RE.findall(text)
    n_words = len(words)
    out = np.zeros(NFW + NSTRUCT, dtype=np.float64)
    if n_words == 0 or n_chars == 0:
        return out
    low = [w.lower() for w in words]

    # --- function-word relative frequencies ---
    for w in low:
        j = FW_INDEX.get(w)
        if j is not None:
            out[j] += 1.0
    out[:NFW] /= n_words

    wlens = np.fromiter((len(w) for w in words), dtype=float, count=n_words)
    counts = {}
    for w in low:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    dis = sum(1 for c in counts.values() if c == 2)
    bigrams = [low[i] + "_" + low[i + 1] for i in range(n_words - 1)]
    ubr = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    sents = [s for s in SENT_RE.split(text) if s.strip()]
    slens = (np.fromiter((len(WORD_RE.findall(s)) for s in sents), dtype=float,
                         count=len(sents)) if sents
             else np.array([n_words], dtype=float))
    smean = slens.mean()

    n_alpha = sum(c.isalpha() for c in text)
    n_digit = sum(c.isdigit() for c in text)
    n_upper = sum(c.isupper() for c in text)
    n_space = sum(c.isspace() for c in text)
    n_punct = sum((not c.isalnum()) and (not c.isspace()) for c in text)
    hedge = sum(1 for w in low if w in HEDGES)
    be = sum(1 for w in low if w in BE_FORMS)
    by = sum(1 for w in low if w == "by")
    stop_mass = out[:NFW].sum()  # fraction of tokens that are function words

    s = out[NFW:]
    s[0] = wlens.mean()
    s[1] = wlens.std()
    s[2] = len(counts) / n_words                      # type-token ratio
    s[3] = hapax / n_words
    s[4] = dis / n_words                              # dis-legomena
    s[5] = ubr
    s[6] = smean
    s[7] = slens.std()
    s[8] = slens.std() / smean if smean > 0 else 0.0  # sentence-length CV
    s[9] = 100.0 * len(sents) / n_words
    s[10] = text.count(",") / n_words
    s[11] = text.count(".") / n_words
    s[12] = text.count(";") / n_words
    s[13] = text.count(":") / n_words
    s[14] = (text.count("(") + text.count(")")) / n_words
    s[15] = (text.count('"') + text.count("'")) / n_words
    s[16] = (text.count("-") + text.count("—")) / n_words
    s[17] = text.count("?") / n_words
    s[18] = text.count("!") / n_words
    s[19] = n_digit / n_chars
    s[20] = n_upper / (n_alpha or 1)
    s[21] = n_alpha / n_chars
    s[22] = n_space / n_chars
    s[23] = n_punct / n_chars
    s[24] = np.mean(wlens <= 3)                       # short-word ratio
    s[25] = np.mean(wlens >= 8)                       # long-word ratio
    s[26] = hedge / n_words
    s[27] = be / n_words
    s[28] = by / n_words
    s[29] = stop_mass
    return out


def build_dense(texts):
    D = np.vstack([_features(t) for t in texts])
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    return D


# ---------------------------------------------------------------------------
# 2. CAPPED wideB SPARSE REP (memory-bounded, per the harness contract)
# ---------------------------------------------------------------------------
def wideB_capped_vecs():
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                        max_features=300000, sublinear_tf=True),
    ]


# ---------------------------------------------------------------------------
# 3. OWN FOLD LOOPS (StandardScaler fit on TRAIN ONLY)
# ---------------------------------------------------------------------------
def eval_standalone(D, Y, folds, est_factory):
    val_f1, tr_f1 = [], []
    for tr, val in folds:
        sc = StandardScaler().fit(D[tr])
        Xtr = sc.transform(D[tr])
        Xev = sc.transform(D[val])
        clf = est_factory().fit(Xtr, Y[tr])
        val_f1.append(macro_f1(Y[val], clf.predict(Xev)))
        tr_f1.append(macro_f1(Y[tr], clf.predict(Xtr)))
    return float(np.mean(val_f1)), float(np.mean(tr_f1)), [round(v, 4) for v in val_f1]


def eval_fusion(D, texts, Y, folds, scale):
    val_f1 = []
    for tr, val in folds:
        vecs = wideB_capped_vecs()
        Xtr_sp = sparse.hstack([v.fit(texts[tr]).transform(texts[tr]) for v in vecs]).tocsr()
        Xev_sp = sparse.hstack([v.transform(texts[val]) for v in vecs]).tocsr()
        sc = StandardScaler().fit(D[tr])
        Dtr = sparse.csr_matrix(sc.transform(D[tr]) * scale)
        Dev = sparse.csr_matrix(sc.transform(D[val]) * scale)
        Xtr = sparse.hstack([Xtr_sp, Dtr]).tocsr()
        Xev = sparse.hstack([Xev_sp, Dev]).tocsr()
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
        val_f1.append(macro_f1(Y[val], clf.predict(Xev)))
    return float(np.mean(val_f1)), [round(v, 4) for v in val_f1]


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("TRACK 5 — topic-invariant stylometry (function words + surface style)")
    print("=" * 78)
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    print(f"train={len(texts)}  test={len(test_texts)}  pos={Y.mean():.4f}")
    print(f"dense block = {NFW} function words + {NSTRUCT} structural = {NFW+NSTRUCT} feats")
    print(f"ANCHOR  LensA={ANCHOR['A']:.4f}  LensB={ANCHOR['B']:.4f}")
    print(f"lensA folds={len(foldsA)}  lensB folds={len(foldsB)}  ({time.time()-t0:.0f}s)")

    print("\nbuilding dense block ...", flush=True)
    D = build_dense(texts)
    print(f"  dense shape={D.shape}  ~{D.nbytes/1e6:.0f}MB  ({time.time()-t0:.0f}s)")

    # ---- (2) standalone dense stylometry ----
    print("\n" + "-" * 78)
    print("[2] STANDALONE dense stylometry (weak absolute F1 expected; watch the GAP)")
    print("-" * 78)
    for name, ef in [("LogReg", lambda: LogisticRegression(
                          C=1.0, class_weight="balanced", max_iter=2000,
                          solver="lbfgs", random_state=SEED)),
                     ("LinSVC", lambda: LinearSVC(
                          C=1.0, class_weight="balanced", random_state=SEED))]:
        av, at, af = eval_standalone(D, Y, foldsA, ef)
        bv, bt, bf = eval_standalone(D, Y, foldsB, ef)
        print(f"  {name:7s} LensA val={av:.4f} (train={at:.4f} gap={at-av:+.3f}) {af}")
        print(f"  {name:7s} LensB val={bv:.4f} (train={bt:.4f} gap={bt-bv:+.3f}) {bf}")

    # ---- (3) fusion sweep ----
    print("\n" + "-" * 78)
    print("[3] FUSION: capped-wideB sparse (+) scaled dense, LinearSVC C=0.25 bal")
    print("-" * 78)
    print(f"  {'scale':>6} {'LensA':>8} {'dA':>8} {'LensB':>8} {'dB':>8}  PASS")
    fusion = {}
    best = None
    for scale in (0.3, 1.0, 3.0):
        a, af = eval_fusion(D, texts, Y, foldsA, scale)
        b, bf = eval_fusion(D, texts, Y, foldsB, scale)
        dA, dB = a - ANCHOR["A"], b - ANCHOR["B"]
        passed = (a > ANCHOR["A"]) and (b > ANCHOR["B"])
        fusion[scale] = (a, b, dA, dB, passed)
        print(f"  {scale:>6.1f} {a:>8.4f} {dA:>+8.4f} {b:>8.4f} {dB:>+8.4f}  "
              f"{'PASS' if passed else 'fail'}   A{af} B{bf}")
        # rank by min-lens margin over anchor (both must clear to matter)
        score = min(dA, dB)
        if best is None or score > best[1]:
            best = (scale, score, a, b, passed)
        print(f"         ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print("[DECISION]")
    print("=" * 78)
    bscale, bscore, ba, bb, bpass = best
    print(f"  best candidate: fusion scale={bscale}  LensA={ba:.4f} LensB={bb:.4f}  "
          f"min-margin={bscore:+.4f}  -> {'PASS' if bpass else 'FAIL'}")

    if bpass:
        print("\n  PASS on BOTH lenses -> refit on all 20k and write scratch_agent5_pred.csv")
        vecs = wideB_capped_vecs()
        Xsp = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
        Xtsp = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
        sc = StandardScaler().fit(D)
        Dtr = sparse.csr_matrix(sc.transform(D) * bscale)
        Dte = sparse.csr_matrix(sc.transform(build_dense(test_texts)) * bscale)
        X = sparse.hstack([Xsp, Dtr]).tocsr()
        Xt = sparse.hstack([Xtsp, Dte]).tocsr()
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(X, Y)
        pred = clf.predict(Xt).astype(int)
        pd.DataFrame({"id": test_ids, "label": pred}).to_csv("scratch_agent5_pred.csv", index=False)
        print(f"  wrote scratch_agent5_pred.csv  rows={len(pred)} "
              f"machine={int(pred.sum())} human={int((pred==0).sum())}")
    else:
        print("\n  FAIL -> no candidate clears the anchor on both lenses; nothing written.")
        print("  (Reminder: even a genuine standalone win here would DEFLATE ~0.075 as a"
              " fused/stylometric leg — so only a large two-lens fusion win could matter.)")

    print(f"\n  total runtime: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
