# ColdStart — Project Instructions

SUTD 50.007 Machine Learning project: detect machine-generated academic-abstract text.
Kaggle competition `50-007-machine-learning-may-2026`, metric **Macro F1**, team **Cold Start**.

## HARD RULES
- **No deep learning** in any submitted/presented model (no transformers, neural nets, or
  fine-tuned pretrained models). Classical ML only. Transformer/RoBERTa/blend artifacts on
  Kaggle are NOT eligible as the course model.
- **Never overwrite results.** Every experiment/submission is APPEND-ONLY. When a new idea is
  tried, ADD a new dated iteration entry stating the new try and its result — do not edit or
  replace an earlier iteration's numbers. This applies to `TESTING_REPORT.md` (the iteration
  ledger), memory, and any results table. Prior results stay verbatim; new tries are appended.
- **Kaggle submissions:** ask/confirm per submission; 5/day quota, resets midnight UTC. Token
  is at `~/.kaggle/access_token` (KGAT_ format) via `KAGGLE_API_TOKEN` env var — rotate/delete
  post-project (it has been exposed in chat + on disk).

## VALIDATION DISCIPLINE (the thing that actually works here)
- The train→test distribution has a **topic shift**. Vanilla CV overstates the real leaderboard
  by ~0.09. Judge models on the **cluster-holdout proxy**, not vanilla CV.
- **Two-lens rule (anti-winner's-curse):** accept a refinement only if it beats the current best
  on TWO independent topic-shift lenses (Lens A = word-unigram KMeans cluster_folds; Lens B =
  char_wb(3,5) KMeans k=16 seed 2026). Single-lens "wins" have deflated repeatedly.
- **Deflation calibration:** same-family single sparse-text model ≈ proxy −0.008; anything with
  ensembles / stacks / tree or stylometric legs ≈ proxy −0.075..0.08.
- What works: MINIMAL, in-family changes that add shift-robustness (not capacity), two-lens gated.
  What fails: stacking, richer word n-grams on the base config, Markov perplexity, clustered experts.

## CURRENT BEST (eligible classical) — see TESTING_REPORT.md ledger for full history
- **STACK = RidgeClassifier(alpha=0.9, balanced) on [1.6·word(1,3) | char_wb(2,6)] uncapped TF-IDF
  → real Kaggle 0.75210** (`predictions/Task3_StackRidgeWord16_Prediction.csv`, Iter 9). Two minimal
  in-family levers stacked (estimator geometry + word-block reweight), 4-lens gated incl. a shift-probe.
  First eligible classical model above our own ineligible transformer (0.75186).
- Prior bests (superseded, kept for history): wideB LinearSVC C=0.25 on word(1,3)+char_wb(2,6) 0.74477
  (`predictions/Task3_Refined_Prediction.csv`); wideA 0.73370; baseline 0.72990.
