"""
Task3_BlendCombine.py — agent E (blend track), final combine + test write.

Loads the cluster-fold-0/1 VAL probabilities produced by:
  * Task3_BlendFolds.py  -> transformer-448 (scratch_blend_fold{k}_probs.npy)
  * Task3_BlendSparse.py -> leg A (SVC-IW-cal) and leg B (LR-C16-iw)

Tunes the blend weight w in {0.0..1.0} for p = w*transformer + (1-w)*sparse on
the cluster-holdout folds (macro-F1 @0.5), for each sparse leg; also evaluates
rank-averaging. Picks the config by 2-fold mean (fold 0 called out separately),
then applies it to the TEST probabilities and writes the prediction CSV.

Run: .venv/bin/python Task3_BlendCombine.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from Task3_Improved_Model import DATA_DIR, OUT_DIR, cluster_folds


def mf1(yt, yp):
    return f1_score(yt, yp, average="macro")


TRANS_ALONE = {0: 0.7763, 1: 0.9005}  # reported by Task3_Transformer.py 448 run


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    y = train["label"].to_numpy(dtype=int)
    folds, _ = cluster_folds(train["text"].astype(str).to_numpy(), y)

    # ---- load + align fold-val probs ----
    T, A, B, YV = {}, {}, {}, {}
    for k in (0, 1):
        vt = np.load(f"scratch_blend_fold{k}_validx.npy")
        vs = np.load(f"scratch_blend_sparse_validx_fold{k}.npy")
        vc = folds[k][1]
        assert np.array_equal(vt, vs), f"fold{k}: transformer/sparse validx mismatch"
        assert np.array_equal(vt, vc), f"fold{k}: validx != cluster_folds val"
        T[k] = np.load(f"scratch_blend_fold{k}_probs.npy")
        A[k] = np.load(f"scratch_blendA_fold{k}_probs.npy")
        B[k] = np.load(f"scratch_blendB_fold{k}_probs.npy")
        YV[k] = y[vt]
        assert len(T[k]) == len(A[k]) == len(B[k]) == len(YV[k])
    print("alignment OK (transformer/sparse/cluster_folds val indices identical)\n")

    # ---- validity gate: transformer fold-val probs reproduce the 448 run ----
    print("VALIDITY GATE (transformer fold-val @0.5 vs reported 448 run):")
    for k in (0, 1):
        got = mf1(YV[k], (T[k] >= 0.5).astype(int))
        d = got - TRANS_ALONE[k]
        flag = "" if abs(d) <= 0.02 else "  <-- WARN >0.02 drift"
        print(f"  fold{k}: recomputed={got:.4f}  reported={TRANS_ALONE[k]:.4f}  "
              f"delta={d:+.4f}{flag}")
    print()

    legs = {"A_SVC-IW-cal": A, "B_LR-C16-iw": B}
    ws = [round(x, 1) for x in np.arange(0.0, 1.0001, 0.1)]

    # ---- grid over weight per leg ----
    rows = []  # (leg, method, w, f0, f1, mean)
    for legname, P in legs.items():
        for w in ws:
            f = {}
            for k in (0, 1):
                p = w * T[k] + (1 - w) * P[k]
                f[k] = mf1(YV[k], (p >= 0.5).astype(int))
            rows.append((legname, "prob", w, f[0], f[1], (f[0] + f[1]) / 2))
        # rank-average (equal weight of rank-normalized probs)
        f = {}
        for k in (0, 1):
            rt = rankdata(T[k]) / len(T[k])
            rs = rankdata(P[k]) / len(P[k])
            pr = 0.5 * rt + 0.5 * rs
            f[k] = mf1(YV[k], (pr >= 0.5).astype(int))
        rows.append((legname, "rankavg", None, f[0], f[1], (f[0] + f[1]) / 2))

    # ---- print full table ----
    print("FULL FOLD TABLE  (w = weight on TRANSFORMER; sparse gets 1-w)")
    print(f"  {'leg':14s} {'method':7s} {'w':>4s}  {'fold0':>7s} {'fold1':>7s} {'mean2':>7s}")
    for legname in legs:
        for r in [x for x in rows if x[0] == legname]:
            wl = f"{r[2]:.1f}" if r[2] is not None else "rank"
            print(f"  {r[0]:14s} {r[1]:7s} {wl:>4s}  "
                  f"{r[3]:7.4f} {r[4]:7.4f} {r[5]:7.4f}")
        print()

    # transformer-alone (w=1) and sparse-alone (w=0) references
    t_f0 = mf1(YV[0], (T[0] >= 0.5).astype(int))
    t_f1 = mf1(YV[1], (T[1] >= 0.5).astype(int))
    t_mean = (t_f0 + t_f1) / 2
    print(f"TRANSFORMER-ALONE (w=1.0): f0={t_f0:.4f} f1={t_f1:.4f} mean={t_mean:.4f}\n")

    # ---- pick best prob-blend config by 2-fold mean (exclude pure endpoints so
    #      "blend" means an actual mix; but still report if endpoint is best) ----
    prob_rows = [r for r in rows if r[1] == "prob"]
    best = max(prob_rows, key=lambda r: (r[5], r[3]))  # tie-break by fold0
    print(f"BEST prob config by mean2: leg={best[0]} w={best[2]:.1f}  "
          f"f0={best[3]:.4f} f1={best[4]:.4f} mean={best[5]:.4f}")

    # honesty: does the chosen blend beat transformer-alone on BOTH folds?
    beats_both = (best[3] > t_f0 + 1e-9) and (best[4] > t_f1 + 1e-9)
    beats_mean = best[5] > t_mean + 1e-9
    print(f"  vs transformer-alone: beats BOTH folds={beats_both}  "
          f"beats mean={beats_mean}")
    print(f"  fold0 (hard): blend {best[3]:.4f} vs transformer {t_f0:.4f} "
          f"({best[3]-t_f0:+.4f})")
    print(f"  fold1        : blend {best[4]:.4f} vs transformer {t_f1:.4f} "
          f"({best[4]-t_f1:+.4f})\n")

    # ---- apply chosen config to TEST ----
    trans_test = np.load("predictions/Task3_Transformer448_Prediction_probs.npy")
    if best[0].startswith("A"):
        sparse_test = np.load("scratch_blendA_test_probs.npy")
    else:
        sparse_test = np.load("predictions/Task3_SparseScreen_probs.npy")
    assert len(trans_test) == len(sparse_test) == len(test) == 6999
    w = best[2]
    p_test = w * trans_test + (1 - w) * sparse_test
    pred = (p_test >= 0.5).astype(int)

    np.save(OUT_DIR / "Task3_Blend_probs.npy", p_test.astype(np.float64))
    out = pd.DataFrame({"id": test["id"], "label": pred})
    assert len(out) == 6999
    assert (out["id"].to_numpy() == test["id"].to_numpy()).all(), "id mismatch"
    assert set(out["label"].unique()) <= {0, 1}
    path = OUT_DIR / "Task3_Blend_Prediction.csv"
    out.to_csv(path, index=False)
    print(f"WROTE {path}  rows={len(out)}  machine={int(pred.sum())} "
          f"({pred.mean():.1%})  human={int((pred == 0).sum())}")
    print(f"  chosen: leg={best[0]}  w(transformer)={w:.1f}  "
          f"sparse_test=({'legA calibrated SVC-IW' if best[0].startswith('A') else 'legB LR-C16-iw saved npy'})")
    print(f"  test prob mean={p_test.mean():.4f}")


if __name__ == "__main__":
    main()
