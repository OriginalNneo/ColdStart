"""
FORENSIC EDA — hunt for an exploitable classical EDGE (leak / duplicates / source /
killer feature). A ~0.045 jump over a tuned linear model almost always = structure,
not a better estimator. Check the cheap high-EV things first.
"""
import re, hashlib, time
import numpy as np, pandas as pd
from collections import Counter

t0 = time.time()
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
print(f"train={len(tr)} test={len(te)} cols_tr={list(tr.columns)} cols_te={list(te.columns)}", flush=True)
print(f"train label balance: {tr['label'].value_counts().to_dict()}  pos_frac={tr['label'].mean():.4f}", flush=True)

# ---- ID structure ----
print("\n== ID structure ==", flush=True)
print(f"train id: min={tr['id'].min()} max={tr['id'].max()} n={tr['id'].nunique()} "
      f"monotonic={tr['id'].is_monotonic_increasing}", flush=True)
print(f"test  id: min={te['id'].min()} max={te['id'].max()} n={te['id'].nunique()} "
      f"monotonic={te['id'].is_monotonic_increasing}", flush=True)
overlap_ids = set(tr['id']) & set(te['id'])
print(f"id overlap train∩test: {len(overlap_ids)}", flush=True)
# is label correlated with id? (ordering leak)
if tr['id'].is_monotonic_increasing:
    for q in range(5):
        chunk = tr.iloc[q*len(tr)//5:(q+1)*len(tr)//5]
        print(f"  id-quintile {q}: pos_frac={chunk['label'].mean():.4f} idrange=[{chunk['id'].min()},{chunk['id'].max()}]", flush=True)

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

# ---- exact duplicate texts within train ----
print("\n== duplicate texts ==", flush=True)
trn = tr['text'].map(norm)
ten = te['text'].map(norm)
dup_tr = trn.duplicated(keep=False)
print(f"train exact-dup rows (normalized): {dup_tr.sum()}", flush=True)
# label consistency among duplicate groups
if dup_tr.sum():
    g = tr.assign(n=trn).groupby('n')['label']
    inconsistent = (g.nunique() > 1).sum()
    print(f"  dup groups: {g.ngroups} with dups={ (g.size()>1).sum() }, label-inconsistent groups={inconsistent}", flush=True)

# ---- train<->test exact overlap (THE big one: direct label lookup) ----
tr_map = {}
for t, l in zip(trn, tr['label']):
    tr_map.setdefault(t, set()).add(l)
hits = ten.map(lambda x: x in tr_map)
print(f"\n== TRAIN∩TEST exact text overlap: {hits.sum()} / {len(te)} test rows ({hits.mean()*100:.2f}%) ==", flush=True)
if hits.sum():
    consistent = sum(1 for x in ten[hits] if len(tr_map[x]) == 1)
    print(f"  of those, {consistent} map to a SINGLE train label (usable as direct answers)", flush=True)

# ---- near-duplicate via normalized-prefix hash (cheap shingle) ----
def prefix_hash(s, n=120):
    return hashlib.md5(norm(s)[:n].encode()).hexdigest()
trp = set(tr['text'].map(prefix_hash))
tep = te['text'].map(prefix_hash)
near = tep.map(lambda h: h in trp)
print(f"\n== near-dup (first-120-char hash) test rows matching some train: {near.sum()} ({near.mean()*100:.2f}%) ==", flush=True)

# ---- killer single features? separability of cheap surface signals ----
print("\n== cheap surface signals: mean by label (machine=1 vs human=0) ==", flush=True)
def feats(s):
    s = str(s)
    w = re.findall(r"[A-Za-z0-9']+", s)
    nw = max(len(w), 1)
    return {
        "len_chars": len(s),
        "len_words": len(w),
        "avg_wlen": np.mean([len(x) for x in w]) if w else 0,
        "n_commas": s.count(","),
        "n_semicolon": s.count(";"),
        "n_emdash": s.count("—") + s.count("–"),
        "n_hyphen": s.count("-"),
        "n_paren": s.count("("),
        "uniq_ratio": len(set(x.lower() for x in w))/nw,
        "digit_frac": sum(c.isdigit() for c in s)/max(len(s),1),
        "upper_frac": sum(c.isupper() for c in s)/max(len(s),1),
        "newline": s.count("\n"),
        "double_space": s.count("  "),
        "trailing_space": int(s != s.rstrip()),
        "nonascii": sum(ord(c) > 127 for c in s),
    }
F = pd.DataFrame([feats(x) for x in tr['text']])
F['label'] = tr['label'].values
for c in F.columns[:-1]:
    m1 = F[F.label==1][c].mean(); m0 = F[F.label==0][c].mean()
    sd = F[c].std() + 1e-9
    cohens_d = (m1 - m0) / sd
    flag = "  <== STRONG" if abs(cohens_d) > 0.5 else ""
    print(f"  {c:15s} human={m0:10.4f} machine={m1:10.4f}  d={cohens_d:+.3f}{flag}", flush=True)

# ---- specific machine-tell tokens ----
print("\n== token tells (freq per class) ==", flush=True)
tells = ["delve","intricate","moreover","furthermore","realm","showcase","underscore",
         "leverage","pivotal","nuanced","comprehensive","notably","tapestry","boasts"]
low_tr = tr['text'].map(lambda s: " "+str(s).lower()+" ")
for w in tells:
    f1 = low_tr[tr.label==1].str.contains(r"\b"+w+r"\b", regex=True).mean()
    f0 = low_tr[tr.label==0].str.contains(r"\b"+w+r"\b", regex=True).mean()
    flag = "  <==" if abs(f1-f0) > 0.05 else ""
    print(f"  {w:14s} human={f0:.3f} machine={f1:.3f} Δ={f1-f0:+.3f}{flag}", flush=True)

print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
