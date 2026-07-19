"""
Task 3 — Markov / n-gram PERPLEXITY detector (fully classical, no deep learning).
================================================================================
The original "N-grams + Markov model" idea, repurposed for DETECTION rather than
generation. Machine-generated text is typically more *predictable* under an n-gram
language model than human text. So:

  1. Fit TWO class-conditional character n-gram language models (stupid-backoff,
     hand-rolled): one on HUMAN training docs, one on MACHINE training docs.
  2. Score each document under both -> mean log-prob (a perplexity proxy).
  3. Features per doc:
        llp_human, llp_machine           (mean log-prob under each class LM)
        LLR = llp_machine - llp_human     (the discriminative signal)
        ppl_human, ppl_machine            (perplexities)
        surprisal_std                     (burstiness of per-char surprisal)
  4. Classify on those few dense features; also test APPENDING them to the
     baseline TF-IDF LinearSVC to see if they ADD signal.

WHY THIS MIGHT DODGE THE TOPIC-SHIFT TAX (the honest hypothesis): raw perplexity
is topic-confounded (rare-topic docs look "surprising" regardless of author), but
the LLR is a *difference* of two LMs, so the topic component partially cancels —
it may transfer better than topic-laden TF-IDF. We test that on the cluster proxy.

All LMs are fit on fold-TRAIN only (leakage-safe). Judged on vanilla 5-fold AND
the realistic cluster-holdout, with the train/val gap shown.

Run:  .venv/bin/python Task3_MarkovPerplexity.py
"""
import math
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from Task3_Improved_Model import cluster_folds, macro_f1

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = Path("data")
ORDER = 4          # character n-gram order
ALPHA = 0.4        # stupid-backoff discount
np.random.seed(SEED)


class CharLM:
    """Hand-rolled character n-gram LM with stupid-backoff scoring.
    Not a normalized probability (stupid backoff), but a valid predictability
    score for feature extraction. Add-1 at the unigram floor so nothing is 0."""

    def __init__(self, order=ORDER, alpha=ALPHA):
        self.order = order
        self.alpha = alpha
        self.ng = [defaultdict(int) for _ in range(order + 1)]   # ng[n]: n-gram -> count
        self.ctx = [defaultdict(int) for _ in range(order + 1)]  # ctx[n]: (n-1)-gram context total
        self.V = 0
        self.uni_total = 0

    def fit(self, texts):
        vocab = set()
        for t in texts:
            s = "\x02" + t + "\x03"          # sentence-boundary sentinels
            L = len(s)
            for i in range(L):
                c = s[i]
                vocab.add(c)
                self.ng[1][c] += 1
                self.uni_total += 1
                for n in range(2, self.order + 1):
                    if i - (n - 1) < 0:
                        break
                    gram = s[i - (n - 1): i + 1]
                    self.ng[n][gram] += 1
                    self.ctx[n][gram[:-1]] += 1
        self.V = max(len(vocab), 1)
        return self

    def _score_char(self, hist, c):
        """Stupid backoff S(c|hist)."""
        for n in range(self.order, 1, -1):
            ctx = hist[-(n - 1):]
            gram = ctx + c
            cnt = self.ng[n].get(gram, 0)
            if cnt > 0:
                return (self.alpha ** (self.order - n)) * cnt / self.ctx[n][ctx]
        # unigram floor (add-1)
        return (self.alpha ** (self.order - 1)) * (self.ng[1].get(c, 0) + 1) / (self.uni_total + self.V)

    def doc_features(self, t):
        """Return (mean_log_prob, perplexity, surprisal_std) for one doc."""
        s = "\x02" + t + "\x03"
        logs = []
        for i in range(1, len(s)):
            hist = s[max(0, i - (self.order - 1)): i]
            p = self._score_char(hist, s[i])
            logs.append(math.log(p) if p > 0 else -20.0)
        logs = np.asarray(logs, dtype=np.float64)
        mlp = float(logs.mean()) if len(logs) else -20.0
        ppl = float(math.exp(-mlp))
        sstd = float(logs.std()) if len(logs) else 0.0
        return mlp, ppl, sstd


def markov_features(train_texts, train_y, eval_texts):
    """Fit HUMAN and MACHINE char-LMs on train, extract 5 features for eval docs.
    Columns: [llp_human, llp_machine, LLR, ppl_human, surprisal_std_machine]."""
    lm_h = CharLM().fit(train_texts[train_y == 0])
    lm_m = CharLM().fit(train_texts[train_y == 1])
    F = np.zeros((len(eval_texts), 5), dtype=np.float32)
    for i, t in enumerate(eval_texts):
        h_mlp, h_ppl, _ = lm_h.doc_features(t)
        m_mlp, m_ppl, m_sstd = lm_m.doc_features(t)
        F[i] = (h_mlp, m_mlp, m_mlp - h_mlp, h_ppl, m_sstd)
    return F


def vec_word_char_mats(texts, tr_idx, ev_idx):
    vecs = [TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)]
    tr = sparse.hstack([v.fit(texts[tr_idx]).transform(texts[tr_idx]) for v in vecs]).tocsr()
    ev = sparse.hstack([v.transform(texts[ev_idx]) for v in vecs]).tocsr()
    return tr, ev


def eval_folds(name, folds, texts, Y, mode):
    """mode: 'markov_lr' | 'markov_svc' | 'baseline+markov'. Returns (f1, train_f1)."""
    pred = np.full(len(Y), -1, dtype=int); mask = np.zeros(len(Y), bool); trf = []
    for tr, val in folds:
        Ftr = markov_features(texts[tr], Y[tr], texts[tr])
        Fev = markov_features(texts[tr], Y[tr], texts[val])
        sc = StandardScaler().fit(Ftr)
        Ztr, Zev = sc.transform(Ftr), sc.transform(Fev)
        if mode == "markov_lr":
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                     random_state=SEED).fit(Ztr, Y[tr])
            pv, ptr = clf.predict(Zev), clf.predict(Ztr)
        elif mode == "markov_svc":
            clf = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED).fit(Ztr, Y[tr])
            pv, ptr = clf.predict(Zev), clf.predict(Ztr)
        else:  # baseline TF-IDF + markov feats appended
            Xtr, Xev = vec_word_char_mats(texts, tr, val)
            Xtr = sparse.hstack([Xtr, sparse.csr_matrix(Ztr)]).tocsr()
            Xev = sparse.hstack([Xev, sparse.csr_matrix(Zev)]).tocsr()
            clf = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED).fit(Xtr, Y[tr])
            pv, ptr = clf.predict(Xev), clf.predict(Xtr)
        pred[val] = pv; mask[val] = True; trf.append(macro_f1(Y[tr], ptr))
    return macro_f1(Y[mask], pred[mask]), float(np.mean(trf))


def main():
    t0 = time.time()
    print("=" * 84, flush=True)
    print("MARKOV / n-gram PERPLEXITY DETECTOR (classical, no DL)", flush=True)
    print("=" * 84, flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    texts = train["text"].astype(str).to_numpy()
    Y = train["label"].to_numpy(dtype=int)
    print(f"train={len(texts)}  char n-gram order={ORDER}  stupid-backoff alpha={ALPHA}", flush=True)

    van = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(Y)), Y))
    clus, _ = cluster_folds(texts, Y)

    print("\n  baseline word+char TF-IDF LinearSVC: vanilla 0.8189 / cluster 0.7404 (bar to beat)\n", flush=True)
    for mode, label in [("markov_svc", "Markov feats only (SVC)"),
                        ("markov_lr", "Markov feats only (LogReg)"),
                        ("baseline+markov", "baseline TF-IDF + Markov feats")]:
        vf, vtr = eval_folds(label, van, texts, Y, mode)
        cf, ctr = eval_folds(label, clus, texts, Y, mode)
        print(f"  {label:<34} vanilla={vf:.4f}  cluster={cf:.4f}  "
              f"gap={ctr-cf:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 84, flush=True)
    print("READ: does the Markov perplexity/LLR feature beat 0.7404 cluster, or ADD to", flush=True)
    print("the baseline? A small gap + a cluster gain that survives would be the real edge.", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
