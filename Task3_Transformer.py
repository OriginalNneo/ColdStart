"""
Task 3 — Fine-tune a small pretrained transformer (DistilBERT) for
GenAI academic-abstract detection (SUTD 50.007 ML project).
Metric = Macro F1. Target: beat the classical best (Kaggle public LB 0.7299).

WHY THIS SCRIPT
---------------
The classical best (word+char TF-IDF + LinearSVC) hit LB 0.7299 despite a
0.8229 vanilla-holdout score: a train->test TOPIC distribution shift deflates
in-distribution validation by ~0.09. A random/stratified holdout therefore
OVERSTATES the leaderboard. The only number that tracked the LB was the
cluster-holdout proxy in Task3_Improved_Model.py (baseline proxy 0.7383 vs
real LB 0.7299). So we judge the transformer on BOTH lenses:

  (1) 90/10 stratified holdout  -> config selection / in-distribution ceiling
  (2) cluster-holdout folds     -> topic-shift proxy = the leaderboard-relevant
                                    number (imported, identical KMeans folds)

MODEL / HYPERPARAMETERS
-----------------------
  distilbert-base-uncased, max_length=256, batch_size=16, lr=2e-5,
  linear warmup (0.06), weight_decay=0.01, epochs=2, macro-F1 early model
  selection (load_best_model_at_end). Device = MPS if available else CPU.
  fp16 disabled (unsupported on MPS).

RUN
---
  .venv/bin/python Task3_Transformer.py
Env knobs (for smoke-testing only):
  SMOKE=1        -> tiny subset, 1 epoch (sanity check the plumbing)
  N_CLUSTER_FOLDS=2  -> how many cluster-holdout folds to fine-tune (default 2)
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# Shift-aware topic-cluster folds + macro_f1, reused from the classical harness
# so the fold definitions are IDENTICAL (same SEED=42 KMeans clustering).
from Task3_Improved_Model import cluster_folds, macro_f1

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = int(os.environ.get("MAX_LEN", "256"))
BATCH_SIZE = int(os.environ.get("BATCH", "16"))
LR = 2e-5
EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_CLUSTER_FOLDS = int(os.environ.get("N_CLUSTER_FOLDS", "2"))

DATA_DIR = "data"
OUT_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(
    OUT_DIR, os.environ.get("OUT_NAME", "Task3_Transformer_Prediction.csv"))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TextDataset(torch.utils.data.Dataset):
    """Pre-tokenized dataset. labels optional (None for test-time inference)."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro")}


def make_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(tokenizer, texts):
    return tokenizer(
        list(texts), truncation=True, max_length=MAX_LENGTH, padding=False
    )


def train_one(tokenizer, tr_texts, tr_y, va_texts, va_y, run_name, epochs=EPOCHS):
    """Fine-tune from a fresh pretrained model. Returns (best_macro_f1, trainer).

    Early model selection on the provided validation set (macro-F1, per epoch).
    """
    set_seed(SEED)
    train_ds = TextDataset(tokenize(tokenizer, tr_texts), tr_y)
    val_ds = TextDataset(tokenize(tokenizer, va_texts), va_y)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join("/tmp", "hf_" + run_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_macro_f1"], trainer


def train_gap(trainer, tr_texts, tr_y, tokenizer):
    """Macro-F1 on the training split (to report the train/val gap)."""
    tr_ds = TextDataset(tokenize(tokenizer, tr_texts), tr_y)
    out = trainer.predict(tr_ds)
    logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    preds = np.argmax(logits, axis=-1)
    return macro_f1(tr_y, preds)


def main():
    t0 = time.time()
    print("=" * 78)
    print("TASK 3 — DistilBERT fine-tune (shift-aware validation)")
    print(f"device={DEVICE}  SMOKE={SMOKE}  cluster_folds={N_CLUSTER_FOLDS}")
    print("=" * 78)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), dtype={"id": str})
    texts = train["text"].astype(str).to_numpy()
    y = train["label"].to_numpy(dtype=int)
    test_texts = test["text"].astype(str).to_numpy()
    test_ids = test["id"].astype(str).to_numpy()
    print(f"train={len(texts)}  test={len(test_texts)}  "
          f"machine={int(y.sum())} ({y.mean():.1%})")

    if SMOKE:
        idx = np.random.RandomState(SEED).choice(len(texts), 400, replace=False)
        texts, y = texts[idx], y[idx]
        test_texts, test_ids = test_texts[:50], test_ids[:50]

    tokenizer = make_tokenizer()
    epochs = 1 if SMOKE else EPOCHS

    # ---- (1) 90/10 stratified holdout: config selection / in-dist ceiling ----
    print("\n" + "-" * 78)
    print("[1] STRATIFIED 90/10 HOLDOUT (config selection, in-distribution)")
    print("-" * 78)
    tr_i, va_i = train_test_split(
        np.arange(len(texts)), test_size=0.10, stratify=y, random_state=SEED
    )
    strat_val_f1, strat_trainer = train_one(
        tokenizer, texts[tr_i], y[tr_i], texts[va_i], y[va_i], "strat", epochs
    )
    strat_train_f1 = train_gap(strat_trainer, texts[tr_i], y[tr_i], tokenizer)
    strat_gap = strat_train_f1 - strat_val_f1
    print(f"  stratified holdout macro-F1 = {strat_val_f1:.4f}  "
          f"(train {strat_train_f1:.4f}, gap {strat_gap:+.4f})")
    del strat_trainer

    # ---- (2) cluster-holdout folds: topic-shift proxy (the LB-relevant number) ----
    print("\n" + "-" * 78)
    print("[2] CLUSTER-HOLDOUT FOLDS (topic-shift proxy; LB-relevant)")
    print("-" * 78)
    folds, _cl = cluster_folds(texts, y)
    n_eval = min(N_CLUSTER_FOLDS, len(folds))
    clus_scores = []
    for fi in range(n_eval):
        tr, va = folds[fi]
        vf1, tr_obj = train_one(
            tokenizer, texts[tr], y[tr], texts[va], y[va], f"clus{fi}", epochs
        )
        clus_scores.append(vf1)
        print(f"  cluster fold {fi}: val macro-F1 = {vf1:.4f}  "
              f"(train {len(tr)}, val {len(va)})")
        del tr_obj
    clus_mean = float(np.mean(clus_scores)) if clus_scores else float("nan")
    print(f"  cluster-holdout mean macro-F1 = {clus_mean:.4f}  "
          f"over {n_eval} folds: {[round(s,4) for s in clus_scores]}")

    # ---- (3) final: retrain on ALL rows, predict test ----
    print("\n" + "-" * 78)
    print("[3] FINAL: retrain on ALL training rows, predict test")
    print("-" * 78)
    set_seed(SEED)
    full_ds = TextDataset(tokenize(tokenizer, texts), y)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir="/tmp/hf_full",
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        seed=SEED,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=full_ds,
        processing_class=tokenizer, data_collator=collator,
    )
    trainer.train()

    test_ds = TextDataset(tokenize(tokenizer, test_texts))
    out = trainer.predict(test_ds)
    logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    preds = np.argmax(logits, axis=-1).astype(int)

    # Softmax probabilities saved alongside the labels so downstream blends
    # (e.g. transformer x LinearSVC) don't need to re-run inference.
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    np.save(OUT_PATH.replace(".csv", "_probs.npy"), probs)

    pd.DataFrame({"id": test_ids, "label": preds}).to_csv(OUT_PATH, index=False)
    print(f"  wrote {OUT_PATH}  rows={len(preds)}  "
          f"machine={int(preds.sum())} ({preds.mean():.1%})  "
          f"human={int((preds==0).sum())}")

    # ---- summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  model                : {MODEL_NAME}  (max_len {MAX_LENGTH}, bs "
          f"{BATCH_SIZE}, lr {LR}, {epochs} epochs)")
    print(f"  stratified holdout F1: {strat_val_f1:.4f} (gap {strat_gap:+.4f})")
    print(f"  cluster-holdout F1   : {clus_mean:.4f}  {[round(s,4) for s in clus_scores]}")
    print(f"  test label dist      : machine {preds.mean():.1%}")
    print(f"  runtime              : {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
