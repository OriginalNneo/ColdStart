"""
Task3_BlendFolds.py — agent E (blend track).

Fine-tune DistilBERT-448 on cluster folds 0 and 1 ONLY (SEED=42 folds from
Task3_Improved_Model.cluster_folds, identical import -> identical folds) and
save the VALIDATION probabilities P(label=1) for each fold's held-out rows.
These fold-val probs are what the transformer run (Task3_Transformer.py) did
NOT persist, and they're needed to tune a blend weight on the cluster-holdout
folds rather than guessing it.

Training paths are reused verbatim from Task3_Transformer.py (train_one,
tokenizer, TextDataset), so the model, seed, max_len(448)/bs(8)/lr/epochs and
load_best_model_at_end model selection are byte-for-byte the same as the run
that produced fold scores 0.7763 (fold 0) and 0.9005 (fold 1).

Run:
  MAX_LEN=448 BATCH=8 nohup .venv/bin/python Task3_BlendFolds.py \
      > scratch_blend.log 2>&1 &
"""
import os
# Match the 448 run BEFORE importing Task3_Transformer (it reads these at import).
os.environ.setdefault("MAX_LEN", "448")
os.environ.setdefault("BATCH", "8")

import time
import numpy as np
import pandas as pd

from Task3_Transformer import (
    make_tokenizer, tokenize, TextDataset, train_one,
    MAX_LENGTH, BATCH_SIZE, DEVICE, EPOCHS,
)
from Task3_Improved_Model import cluster_folds, macro_f1

FOLDS_TO_RUN = [0, 1]


def main():
    t0 = time.time()
    print("=" * 78, flush=True)
    print("Task3_BlendFolds — DistilBERT fold-val probs (folds 0,1)", flush=True)
    print(f"device={DEVICE}  max_len={MAX_LENGTH}  bs={BATCH_SIZE}  epochs={EPOCHS}",
          flush=True)
    print("=" * 78, flush=True)

    train = pd.read_csv("data/train.csv")
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    print(f"train={len(texts)}  machine={int(y.sum())} ({y.mean():.1%})", flush=True)

    folds, _cl = cluster_folds(texts, y)
    tokenizer = make_tokenizer()

    for k in FOLDS_TO_RUN:
        tr, va = folds[k]
        print(f"\n---- FOLD {k}: train {len(tr)}, val {len(va)} ----", flush=True)
        t = time.time()
        vf1, trainer = train_one(
            tokenizer, texts[tr], y[tr], texts[va], y[va], f"blend{k}"
        )
        # Predict val probabilities with the (best) selected model.
        val_ds = TextDataset(tokenize(tokenizer, texts[va]), y[va])
        out = trainer.predict(val_ds)
        logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
        logits = np.asarray(logits, dtype=np.float64)
        # numerically-stable softmax, P(label=1)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p1 = (e[:, 1] / e.sum(axis=1))
        # sanity: argmax-F1 should match the reported eval macro-F1
        chk = macro_f1(y[va], (p1 >= 0.5).astype(int))
        np.save(f"scratch_blend_fold{k}_probs.npy", p1.astype(np.float64))
        np.save(f"scratch_blend_fold{k}_validx.npy", va.astype(np.int64))
        print(f"  fold {k}: eval_macro_f1={vf1:.4f}  recheck@0.5={chk:.4f}  "
              f"saved scratch_blend_fold{k}_probs.npy ({len(p1)} rows)  "
              f"[{time.time()-t:.0f}s]", flush=True)
        del trainer

    print(f"\nALL DONE  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
