"""
Track 8 — external data augmentation from HuggingFace (RESEARCH PROBE).
Classical ML only: sklearn TF-IDF + LinearSVC. External data ONLY enters TRAIN.
Validation is always original-train fold-val rows, scored on original labels.

External dataset: NicolaiSivesind/ChatGPT-Research-Abstracts (license: cc)
  10000 rows, each paper -> real_abstract (human, label 0) + generated_abstract
  (ChatGPT/GPT-3.5, label 1). => 20000 paragraph-length academic abstracts, 50/50.

Rep (capped wideB, per task spec):
  word(1,3) min_df=2 sublinear_tf
  char_wb(2,6) min_df=3 max_features=300000 sublinear_tf
  LinearSVC(C=0.25, class_weight=balanced)

Fold loop is OUR OWN: train = original[tr] (+ external), validate = original[val].
ANCHOR = {A:0.7515, B:0.7467}.  PASS = beats anchor on BOTH lenses.

ELIGIBILITY CAVEAT: Kaggle competitions frequently FORBID external training data.
This is a research probe of whether external human/machine abstracts help the
topic-shift lenses — NOT a shippable win unless the competition rules permit it.
"""
import sys, time, json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scratch_lens import load_data, get_folds, macro_f1, ANCHOR, SEED

t0 = time.time()
def log(*a): print(*a, flush=True)

# ---------- external data ----------
def load_external(cap=20000):
    from datasets import load_dataset
    ds = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts", split="train")
    texts, labels = [], []
    for r in ds:
        h = (r["real_abstract"] or "").strip()
        m = (r["generated_abstract"] or "").strip()
        if len(h.split()) >= 20:
            texts.append(h); labels.append(0)
        if len(m.split()) >= 20:
            texts.append(m); labels.append(1)
    texts = np.array(texts, dtype=object); labels = np.array(labels, dtype=int)
    # dedup + cap (shuffle deterministically, keep balance-ish)
    _, uniq = np.unique(texts, return_index=True)
    uniq = np.sort(uniq); texts, labels = texts[uniq], labels[uniq]
    if len(texts) > cap:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(texts))[:cap]
        texts, labels = texts[idx], labels[idx]
    return texts, labels

def make_vecs():
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                        max_features=300000, sublinear_tf=True, dtype=np.float32),
    ]

def fit_transform(vecs, fit_texts, *xform_sets):
    for v in vecs: v.fit(fit_texts)
    outs = []
    for ts in xform_sets:
        outs.append(sparse.hstack([v.transform(ts) for v in vecs]).tocsr())
    return outs

def run_lens(name, folds, texts, Y, ext_texts, ext_labels):
    """Returns dict: anchor mean, ext_w1.0 mean, ext_w0.3 mean (+per-fold)."""
    res = {"anchor": [], "ext_w1.0": [], "ext_w0.3": []}
    for tr, val in folds:
        tr_t, tr_y = texts[tr], Y[tr]
        val_t, val_y = texts[val], Y[val]

        # --- anchor: vectorizer fit on original-train only, no external ---
        va = make_vecs()
        Xtr, Xev = fit_transform(va, tr_t, tr_t, val_t)
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, tr_y)
        res["anchor"].append(macro_f1(val_y, clf.predict(Xev)))

        # --- +external: vectorizer fit on original-train + external ---
        ve = make_vecs()
        combo_t = np.concatenate([tr_t, ext_texts])
        combo_y = np.concatenate([tr_y, ext_labels])
        Xc, Xevc = fit_transform(ve, combo_t, combo_t, val_t)
        n_orig = len(tr_t)
        for w_ext, key in [(1.0, "ext_w1.0"), (0.3, "ext_w0.3")]:
            sw = np.ones(len(combo_y), dtype=np.float32)
            sw[n_orig:] = w_ext
            clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
            clf.fit(Xc, combo_y, sample_weight=sw)
            res[key].append(macro_f1(val_y, clf.predict(Xevc)))
        del Xc, Xevc, Xtr, Xev
        log(f"  [{name}] fold done ({time.time()-t0:.0f}s)")
    return {k: (float(np.mean(v)), [round(x, 4) for x in v]) for k, v in res.items()}

def main():
    log("=== Track 8: external abstract augmentation ===")
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    log(f"train={len(texts)} pos={Y.mean():.4f}  foldsA={len(foldsA)} foldsB={len(foldsB)}")
    ext_texts, ext_labels = load_external()
    log(f"external rows={len(ext_texts)}  pos={ext_labels.mean():.4f}  ({time.time()-t0:.0f}s)")

    A = run_lens("A", foldsA, texts, Y, ext_texts, ext_labels)
    log("Lens A done"); log(json.dumps({k: v[0] for k, v in A.items()}))
    B = run_lens("B", foldsB, texts, Y, ext_texts, ext_labels)
    log("Lens B done"); log(json.dumps({k: v[0] for k, v in B.items()}))

    log("\n================ RESULTS ================")
    log(f"ANCHOR (task) A={ANCHOR['A']:.4f}  B={ANCHOR['B']:.4f}")
    for cond in ["anchor", "ext_w1.0", "ext_w0.3"]:
        a, af = A[cond]; b, bf = B[cond]
        dA = a - ANCHOR["A"]; dB = b - ANCHOR["B"]
        passed = (a > ANCHOR["A"]) and (b > ANCHOR["B"])
        log(f"{cond:10s}  A={a:.4f} (Δ{dA:+.4f}) {af}")
        log(f"{'':10s}  B={b:.4f} (Δ{dB:+.4f}) {bf}  PASS={passed}")

    # pick best external condition by min-lens margin over anchor
    best = None
    for cond in ["ext_w1.0", "ext_w0.3"]:
        a = A[cond][0]; b = B[cond][0]
        passed = (a > ANCHOR["A"]) and (b > ANCHOR["B"])
        margin = min(a - ANCHOR["A"], b - ANCHOR["B"])
        if best is None or margin > best[1]:
            best = (cond, margin, passed, a, b)
    log(f"\nBEST external cond={best[0]} min_margin={best[1]:+.4f} PASS={best[2]}")

    results = {"anchor_A": ANCHOR["A"], "anchor_B": ANCHOR["B"],
               "A": {k: v[0] for k, v in A.items()},
               "B": {k: v[0] for k, v in B.items()},
               "best": {"cond": best[0], "margin": best[1], "pass": best[2]}}
    with open("scratch/agent8_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("wrote scratch/agent8_results.json")

    if best[2]:  # PASS on both lenses -> refit full + external, write pred
        log("\nBEST PASSES both lenses -> refit full 20k + external, write pred")
        w_ext = float(best[0].split("w")[1])
        ve = make_vecs()
        combo_t = np.concatenate([texts, ext_texts])
        combo_y = np.concatenate([Y, ext_labels])
        Xall, Xte = fit_transform(ve, combo_t, combo_t, test_texts)
        sw = np.ones(len(combo_y), dtype=np.float32); sw[len(texts):] = w_ext
        clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
        clf.fit(Xall, combo_y, sample_weight=sw)
        pred = clf.predict(Xte)
        import pandas as pd
        pd.DataFrame({"id": test_ids, "label": pred}).to_csv("scratch_agent8_pred.csv", index=False)
        log(f"wrote scratch_agent8_pred.csv  pred_pos={pred.mean():.4f}")
    else:
        log("\nBest does NOT pass both lenses -> NO prediction file written (honest).")
    log(f"done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
