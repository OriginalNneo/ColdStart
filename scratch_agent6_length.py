"""
Track 6 — length / truncation robustness (classical ML only).
Hypothesis: long docs carry MORE topic content; limiting length may improve
topic-shift robustness. Test token/char truncation + head+tail on wideB rep,
judged on BOTH lenses vs the cached anchor.

Does NOT edit any protected file. Writes only scratch_agent6_* outputs.
"""
import time
import numpy as np
from scratch_lens import load_data, get_folds, eval_rep, wideB_vecs, ANCHOR

t0 = time.time()


def trunc_tokens(texts, n):
    out = np.empty(len(texts), dtype=object)
    for i, t in enumerate(texts):
        out[i] = " ".join(t.split()[:n])
    return out


def trunc_chars(texts, n):
    out = np.empty(len(texts), dtype=object)
    for i, t in enumerate(texts):
        out[i] = t[:n]
    return out


def head_tail(texts, h, tl):
    out = np.empty(len(texts), dtype=object)
    for i, t in enumerate(texts):
        toks = t.split()
        if len(toks) <= h + tl:
            out[i] = " ".join(toks)
        else:
            out[i] = " ".join(toks[:h] + toks[-tl:])
    return out


def run(name, arr, texts_orig, Y, foldsA, foldsB, results):
    a, af = eval_rep(wideB_vecs, arr, Y, foldsA)
    b, bf = eval_rep(wideB_vecs, arr, Y, foldsB)
    dA, dB = a - ANCHOR["A"], b - ANCHOR["B"]
    pas = a > ANCHOR["A"] and b > ANCHOR["B"]
    results.append((name, a, b, dA, dB, pas))
    print(f"{name:22s} LensA={a:.4f} ({dA:+.4f})  LensB={b:.4f} ({dB:+.4f})  "
          f"{'PASS' if pas else 'fail'}  [{time.time()-t0:.0f}s]", flush=True)
    return a, b


def main():
    texts, Y, test_texts, test_ids = load_data()
    foldsA, foldsB = get_folds()
    lens = np.array([len(t.split()) for t in texts])
    print(f"train={len(texts)} pos={Y.mean():.4f}  token_len: "
          f"min={lens.min()} med={int(np.median(lens))} max={lens.max()}  "
          f"foldsA={len(foldsA)} foldsB={len(foldsB)}", flush=True)

    results = []
    # 1) FULL text anchor reproduction
    run("FULL(anchor)", texts, texts, Y, foldsA, foldsB, results)
    print(f"  cached ANCHOR A={ANCHOR['A']} B={ANCHOR['B']}", flush=True)

    # 2) token truncation
    for n in (75, 150, 300, 600):
        run(f"tok{n}", trunc_tokens(texts, n), texts, Y, foldsA, foldsB, results)

    # 3) char truncation
    for n in (500, 1000, 2000):
        run(f"char{n}", trunc_chars(texts, n), texts, Y, foldsA, foldsB, results)

    # 4) head+tail
    run("head300+tail100", head_tail(texts, 300, 100), texts, Y, foldsA, foldsB, results)

    # summary
    print("\n==== SUMMARY (delta vs anchor) ====", flush=True)
    passers = [r for r in results if r[5] and r[0] != "FULL(anchor)"]
    for name, a, b, dA, dB, pas in results:
        print(f"{name:22s} A={a:.4f}({dA:+.4f}) B={b:.4f}({dB:+.4f}) "
              f"{'PASS' if pas else 'fail'}", flush=True)

    if passers:
        # best passer by min-delta (robust across both lenses)
        best = max(passers, key=lambda r: min(r[3], r[4]))
        print(f"\nBEST PASSER: {best[0]}  (minDelta={min(best[3],best[4]):+.4f})", flush=True)
        _refit_and_predict(best[0], texts, Y, test_texts, test_ids)
    else:
        print("\nNo truncation PASSES both lenses. Null result — full text retained.", flush=True)


def _apply(name, texts):
    if name.startswith("tok"):
        return trunc_tokens(texts, int(name[3:]))
    if name.startswith("char"):
        return trunc_chars(texts, int(name[4:]))
    if name == "head300+tail100":
        return head_tail(texts, 300, 100)
    return texts


def _refit_and_predict(name, texts, Y, test_texts, test_ids):
    from scipy import sparse
    from sklearn.svm import LinearSVC
    import pandas as pd
    tr = _apply(name, texts)
    te = _apply(name, test_texts)
    vecs = wideB_vecs()
    Xtr = sparse.hstack([v.fit(tr).transform(tr) for v in vecs]).tocsr()
    Xte = sparse.hstack([v.transform(te) for v in vecs]).tocsr()
    clf = LinearSVC(C=0.25, class_weight="balanced", random_state=42)
    clf.fit(Xtr, Y)
    pred = clf.predict(Xte)
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(
        "scratch_agent6_pred.csv", index=False)
    print(f"wrote scratch_agent6_pred.csv (trunc={name}, {len(pred)} rows, "
          f"pos={pred.mean():.4f})", flush=True)


if __name__ == "__main__":
    main()
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
