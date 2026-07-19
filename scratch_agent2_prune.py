"""
Track 2 — topic-word pruning for shift-robustness.
Strip topic-specific signal so the model leans on style, not topic.
Classical ML only. Capped wideB anchor per orchestrator spec.

Caps (identical for anchor + all candidates):
  word:  word(1,3), min_df=2, sublinear_tf=True
  char:  char_wb(2,6), min_df=3, max_features=300000, sublinear_tf=True
"""
import sys, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from scipy import sparse

from scratch_lens import load_data, get_folds, eval_rep, macro_f1, ANCHOR, SEED

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

texts, Y, test_texts, test_ids = load_data()
foldsA, foldsB = get_folds()
log(f"train={len(texts)} test={len(test_texts)} foldsA={len(foldsA)} foldsB={len(foldsB)}")

# ---------- capped vectorizer factories ----------
def word_vec(max_df=1.0, stop=None):
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                           max_df=max_df, sublinear_tf=True,
                           stop_words=(list(stop) if stop else None))

def char_vec(max_df=1.0):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                           max_features=300000, max_df=max_df, sublinear_tf=True)

def make_factory(word_max_df=1.0, char_max_df=1.0, stop=None):
    return lambda: [word_vec(word_max_df, stop), char_vec(char_max_df)]

def evaluate(name, factory):
    a, af = eval_rep(factory, texts, Y, foldsA)
    b, bf = eval_rep(factory, texts, Y, foldsB)
    da, db = a - ANC_A, b - ANC_B
    pas = (a > ANC_A) and (b > ANC_B)
    log(f"{name:28s} A={a:.4f}({da:+.4f}) B={b:.4f}({db:+.4f}) "
        f"{'PASS' if pas else 'fail'}  {af} {bf}")
    return dict(name=name, A=a, B=b, dA=da, dB=db, PASS=pas, factory=factory)

# ---------- 1) capped anchor (internal baseline) ----------
log("=== 1) capped anchor ===")
anchor_fac = make_factory()
ca, caf = eval_rep(anchor_fac, texts, Y, foldsA)
cb, cbf = eval_rep(anchor_fac, texts, Y, foldsB)
ANC_A, ANC_B = ca, cb
log(f"capped-anchor  A={ca:.4f} {caf}")
log(f"               B={cb:.4f} {cbf}")
log(f"ledger ANCHOR cache A={ANCHOR['A']} B={ANCHOR['B']} (spec baseline)")

results = []

# ---------- 2) max_df sweep ----------
log("=== 2) max_df sweep (both vectorizers) ===")
for mdf in [0.9, 0.7, 0.5, 0.3]:
    results.append(evaluate(f"maxdf_both={mdf}", make_factory(mdf, mdf)))
log("--- max_df on WORD block only ---")
for mdf in [0.9, 0.7, 0.5, 0.3]:
    results.append(evaluate(f"maxdf_word={mdf}", make_factory(mdf, 1.0)))

# ---------- 3) supervised topic-word removal ----------
log("=== 3) supervised topic-word removal (chi2 vs KMeans topic clusters) ===")
# replicate lens-A style topic clustering on full train
cv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=5,
                     max_features=20000, sublinear_tf=True)
Xc = cv.fit_transform(texts)
cl = MiniBatchKMeans(10, random_state=SEED, n_init=5, batch_size=2048).fit_predict(Xc)
# chi2: which unigrams most associate with topic cluster labels
chi_scores, _ = chi2(Xc, cl)
vocab = np.array(cv.get_feature_names_out())
order = np.argsort(-np.nan_to_num(chi_scores))
for K in [50, 200, 500, 1000]:
    stop = set(vocab[order[:K]].tolist())
    results.append(evaluate(f"topic_chi2_K={K}", make_factory(stop=stop)))
log(f"sample top topic words: {list(vocab[order[:20]])}")

# ---------- 4) adversarial-drop (covariate-shift drivers) ----------
log("=== 4) adversarial validation drop ===")
# label train=0 test=1 on word unigrams; features driving separation = shift drivers
adv_texts = np.concatenate([texts, test_texts])
adv_y = np.concatenate([np.zeros(len(texts)), np.ones(len(test_texts))])
av = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=5,
                     max_features=40000, sublinear_tf=True)
Xadv = av.fit_transform(adv_texts)
adv_clf = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)
adv_clf.fit(Xadv, adv_y)
avocab = np.array(av.get_feature_names_out())
coef = adv_clf.coef_.ravel()
# |coef| large = most train/test discriminative (either direction)
aorder = np.argsort(-np.abs(coef))
# quick AV signal quality: 3-fold cv-ish accuracy proxy
from sklearn.model_selection import cross_val_score
try:
    avacc = cross_val_score(LinearSVC(C=0.5, class_weight="balanced", random_state=SEED),
                            Xadv, adv_y, cv=3, scoring="roc_auc").mean()
    log(f"adversarial train/test AUC={avacc:.4f} (0.5=indistinguishable)")
except Exception as e:
    log(f"AV auc skipped: {e}")
for K in [50, 200, 500, 1000]:
    stop = set(avocab[aorder[:K]].tolist())
    results.append(evaluate(f"adv_drop_K={K}", make_factory(stop=stop)))
log(f"sample top shift words: {list(avocab[aorder[:20]])}")

# ---------- summary + refit best if PASS ----------
log("=== SUMMARY ===")
log(f"capped-anchor A={ANC_A:.4f} B={ANC_B:.4f}")
passing = [r for r in results if r["PASS"]]
for r in sorted(results, key=lambda r: -(r["dA"] + r["dB"])):
    log(f"  {r['name']:28s} A={r['A']:.4f}({r['dA']:+.4f}) "
        f"B={r['B']:.4f}({r['dB']:+.4f}) {'PASS' if r['PASS'] else ''}")

if passing:
    best = max(passing, key=lambda r: min(r["dA"], r["dB"]))
    log(f"BEST PASS: {best['name']}  A={best['A']:.4f} B={best['B']:.4f}")
    # refit on all 20k, predict test
    vecs = best["factory"]()
    Xtr = sparse.hstack([v.fit(texts).transform(texts) for v in vecs]).tocsr()
    Xte = sparse.hstack([v.transform(test_texts) for v in vecs]).tocsr()
    clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, Y)
    pred = clf.predict(Xte)
    import csv
    with open("scratch_agent2_pred.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for i, p in zip(test_ids, pred):
            w.writerow([i, int(p)])
    log(f"wrote scratch_agent2_pred.csv ({len(pred)} rows, pos={pred.mean():.4f})")
else:
    log("NO candidate passes BOTH lenses -> null result, no prediction written.")
log("DONE")
