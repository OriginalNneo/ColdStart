# Refinement Campaign — 8 parallel tracks (classical only, two-lens gated)

Anchor to beat: **wideB** = LinearSVC(C=0.25, balanced) on word(1,3)+char_wb(2,6), real Kaggle **0.74477**, Lens A 0.7439 / Lens B 0.7640.
Goal (user): push toward 0.88. Honest frontier: LB leader 0.795 (DL). Realistic classical transferable ceiling ~0.75–0.76. We hunt for any *transferable* edge; two-lens gate rejects winner's-curse mirages.

Already dead (ledger): stacking/XGB meta, richer word n-grams on base, Markov perplexity, clustered experts, threshold tuning, C tuning, wider char(2,7), transductive vocab, adversarial IW, min_df=5, binary/l2.

## Tracks (each = one subagent, imports scratch_lens.py, writes scratch_agentN_*.py, NO edits to shared/ledger/predictions)
1. **NBSVM** — NB log-count-ratio reweighting of TF-IDF before the linear SVM (Wang & Manning). Untried; often +1–2 F1 on text, stays single linear model.
2. **Topic-word pruning** — sweep max_df / drop most topic-discriminative terms to strip topic markers → shift robustness on the wideB rep.
3. **Block reweighting** — scale char block vs word block (char = more style/topic-robust). Sweep α on the single model.
4. **Estimator geometry** — same wideB rep, swap LinearSVC → LogReg / SGD(modified_huber) / PassiveAggressive / ComplementNB, light tune. Different regularization may transfer.
5. **Topic-invariant stylometry** — function-word + punctuation-rhythm surface features (topic-independent authorship signal), standalone and lightly fused.
6. **Length/truncation robustness** — truncate to first N tokens/chars, per-length normalization; abstracts span 27–17436 chars.
7. **Provided 5000-feat TF-IDF + adversarial feature dropping** — the shipped features alone and as a concat leg; drop top covariate-shift features.
8. **External data augmentation (HuggingFace)** — pull human/machine abstract corpora to add training signal; flag competition-rules caveat; judge on the lenses.

## Acceptance
Report Lens A & Lens B macro-F1 (candidate vs its own wideB anchor under identical caps). PASS = beats anchor on BOTH lenses. Only passers get re-validated by me at full resolution + considered for a Kaggle slot.
