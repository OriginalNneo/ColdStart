# Overnight Task 3 Run — 2026-07-15 (started ~01:00 SGT)

Goal: produce ranked, verified submission candidates for the 5 Kaggle slots that
open at 8:00 SGT. Standing best 0.72990 (LinearSVC word+char TF-IDF). Target 0.78326.

## Calibration rule (fixed, paid for with 2 failed submissions)
- Single sparse-text model: proxy − ~0.008 = projected real LB
- Anything with stylometric/dense-tree legs or ensembles thereof: proxy − ~0.075
- Transformer family: unknown deflation, but small train/val gap and beat baseline on both folds tested

## Live tracks
| Track | Runner | Status | Output |
|---|---|---|---|
| DistilBERT max_len=448 (restart; caffeinate) | background job | DONE + VERIFIED (03:00) — holdout 0.8718 (256-ver: 0.8498); cluster folds 0.7763/0.9005 mean 0.8384 (256-ver: 0.8160; SVC baseline same folds: 0.7621); test machine-rate 72.7%; runtime 7282s | predictions/Task3_Transformer448_Prediction.csv + _probs.npy |
| Transformer448 × sparse blend (agent E, opus) | subagent | DONE + VERIFIED (03:57) — leg A calibrated SVC-IW, w=0.5; fold0 0.7866 / fold1 0.9085 (mean 0.8475) beats transformer-alone 0.8378 on BOTH folds | predictions/Task3_Blend_Prediction.csv + _probs.npy |
| IW LinearSVC grid (agent A, fable) | subagent | DONE + VERIFIED (re-run reproduced 0.7721 exactly, per-fold identical) — LEAD CANDIDATE | predictions/Task3_IW_Tuned_Prediction.csv |
| Compose A×C levers (agent D, opus) | subagent | DONE — levers did not stack (0.7736 within noise of 0.7721); crop-aug hurts on tuned config | predictions/Task3_Composed_Prediction.csv (marginal) |
| Diverse sparse screen incl. NBSVM (agent B, fable) | subagent | DONE + VERIFIED — LR C=16 + IW 0.7537 (ties incumbent); calibrated probs saved for blending | predictions/Task3_SparseScreen_Prediction.csv |
| Length-shift adaptation (agent C, opus) | subagent | DONE + VERIFIED — binary rep + crop-aug 0.7504; length-IW rejected as trap; length shift NOT the sparse family's problem | predictions/Task3_LengthAdapt_Prediction.csv |
| Transformer seed-ensemble (agent F, opus) | subagent | DONE + VERIFIED (05:18) — seeds 42/7/2024 full-train 448 (SKIP_EVAL); machine-rates 72.7/75.1/73.0% (all within 5pt of seed42 — none suspect); prob-mean ensemble machine 73.7%; re-blend 0.5·ens+0.5·sparseA machine 73.7% | predictions/Task3_Transformer448_ens_Prediction.csv (+_probs.npy), predictions/Task3_BlendEns_Prediction.csv (+_probs.npy) |

## Results ledger (agents append below; every number must state its evaluation protocol)
| Time | Track | Config | Proxy F1 (5-fold cluster-holdout unless stated) | Projected LB | CSV |
|---|---|---|---|---|---|
| baseline | LinearSVC word+char | C=0.25, balanced | 0.7383 | 0.7299 (actual) | Task3_Final_Prediction.csv |
| prior | + importance weighting | clip [0.25,4] | 0.7523 | ~0.744 | Task3_PseudoLabel_Prediction.csv |
| prior | DistilBERT-256 | 2 folds only: 0.8160 (folds run high; baseline=0.7621 on same folds) | — | ~0.75+ | Task3_Transformer_Prediction.csv |
| 03:00 | DistilBERT-448 (fixes 66% test truncation) | max_len 448, bs 8, lr 2e-5, 2 ep; holdout 0.8718 gap +0.041 | 2 folds: 0.8384 (0.7763/0.9005) — beats 256-ver on both folds | ~0.76+ (same 2-fold caveat as 256-ver) | Task3_Transformer448_Prediction.csv (+ _probs.npy) |
| 01:35 | length-adapt (agent C) | binary=True TF-IDF (word 1-2 + char_wb 3-5, no sublinear, l2) + crop-augment to train-median 190 tok, LinearSVC C=0.25 balanced | cluster 0.7504 (BASE repro 0.7383); length-shifted holdout (short 60%→long 40%) 0.8076 (BASE 0.8022) — beat BASE on both lenses; IW-by-length REJECTED (won length lens only = the −0.075 trap signature) | ~0.742 (single sparse −0.008 rule) | Task3_LengthAdapt_Prediction.csv |
| ~02:00 | IW-tuned SVC | LinearSVC C=1.0, word(1,3)+char_wb(2,5) TF-IDF min_df=2 sublinear, transductive vocab, IW clip [0.1,10] | 0.7721 (beats IW 0.7523 on all 5 folds; repro of BASE/IW exact) | 0.7641 | Task3_IW_Tuned_Prediction.csv |
| 01:21 | sparse screen (agent B) | LogisticRegression C=16 liblinear balanced, word(1-2)+char_wb(3-5) TF-IDF, IW machinery (transductive vocab + adversarial weights); repro BASE=0.7383/IW=0.7523 exact; beat NBSVM/CNB/SGD/char-only | 0.7537 (+0.0014 vs IW incumbent, within noise) | 0.7457 | Task3_SparseScreen_Prediction.csv (+ _probs.npy P(1)) |
| 02:07 | composed (agent D) | A-config (LinearSVC word(1,3)+char_wb(2,5) min_df=2, transductive vocab, IW clip [0.1,10]) + agent-C levers stacked. Factorial: binary rep replacing sublinear + crop-aug(190 tok, crops keep source IW wt; vocab excludes crops) + C∈{0.5,1,2}. Winner A_bin_C0.5 (binary+l2, C=0.5, NO crop). Repro of A_base = 0.7721 EXACT. | 0.7736 (+0.0015 vs A_base 0.7721 — WITHIN fold-noise; per-fold [.7549,.8274,.7672,.798,.7204], binary helps big folds but hurts hard fold4 −0.006; crop HURTS uniformly −0.002, C=2 hurts, A_bin@C1 wash −0.0004). Levers effectively DID NOT stack. | 0.7656 | Task3_Composed_Prediction.csv (marginal; A_bin_C0.5 binary+l2 C=0.5) |
| 03:57 | blend (agent E) | prob-blend p=w·transformer448 + (1−w)·sparse, weight TUNED on cluster folds 0/1 (not guessed). Legs: A=calibrated LinearSVC-IW (word(1,3)+char_wb(2,5) min_df2, transductive vocab, adversarial IW clip[0.1,10], C=1 balanced, Platt CalibratedClassifierCV sigmoid cv=3 w/ sample_weight=IW — same calib on folds+full-train test); B=LR-C16-iw (reused SparseScreen probs, folds reproduced 0.7420/0.8040 EXACT). Transformer fold-val probs reproduced 448 run: 0.7754/0.9003 (reported 0.7763/0.9005). CHOSEN: leg A, w=0.5. | 0.8475 mean2 (fold0 0.7866, fold1 0.9085) — BEATS transformer-alone (0.8378; f0 0.7754, f1 0.9003) on BOTH folds: +0.0112 fold0, +0.0082 fold1. Leg B w=0.5 close 2nd (0.8469). Rank-avg worse (0.8269, 0.5-thresh scale penalty). | n/a (transformer deflation unknown; blend is strictly ≥ its legs on both folds) | Task3_Blend_Prediction.csv (+ _probs.npy; 6999 rows, ids verified, machine 73.0%) |
| 05:18 | seed-ens (agent F) | Task3_Transformer_Seed.py (copy of Task3_Transformer.py; SEED from env, SKIP_EVAL=1 skips stratified+cluster stages → straight to full-train). 3 full-train DistilBERT-448 fits (bs8, 2ep) SEED∈{42,7,2024}. Ensemble = mean of P(1) probs, argmax@0.5. Re-blend reuses agent E's EXACT recipe: p=0.5·ens+0.5·legA(scratch_blendA_test_probs.npy, calibrated SVC-IW full-train test probs). Per-seed machine-rate 72.7/75.1/73.0% (seed7 +2.3pt vs seed42's 72.7% — within ±5pt, NOT suspect). Pairwise label agreement 42v7 93.24%, 42v2024 94.11%, 7v2024 92.98%. Runtimes 2039s/2041s. | HONEST: NOT fold-validated — seed-averaging is variance reduction only; the fold-validated numbers (blend 0.8475 mean2) belong to the single-seed models. Ensemble machine 73.7% (5155); differs from seed42 by 220, seed7 283, seed2024 236 labels. Blend-ens machine 73.7% (5158); differs from agent-E single-seed blend by 198 labels. | n/a (variance-reduction candidate; no new fold proxy) | Task3_Transformer448_ens_Prediction.csv (+_probs.npy), Task3_BlendEns_Prediction.csv (+_probs.npy; 6999 rows, ids verified) |

## Morning submission slate — FINAL (assembled 05:30 SGT; nothing auto-submitted)

All CSVs verified: 6,999 rows, header `id,label`, ids match data/test.csv order.
Recommended submission order for the 5 slots (quota resets 08:00 SGT):

| # | File | What it is | Why this rank |
|---|---|---|---|
| 1 | predictions/Task3_BlendEns_Prediction.csv | 3-seed DistilBERT-448 prob-ensemble × calibrated SVC-IW, w=0.5 | Highest expected score. The single-seed blend beat transformer-alone on BOTH proxy folds (0.7866/0.9085 vs 0.7754/0.9003, w-plateau robust); seed-averaging adds variance reduction on top. |
| 2 | predictions/Task3_Transformer448_Prediction.csv | DistilBERT-448 alone, seed 42 | Fold-validated (2-fold 0.8384, beats 256-ver on both folds). Measures the transformer family's real LB deflation and isolates the blend's contribution. |
| 3 | predictions/Task3_IW_Tuned_Prediction.csv | LinearSVC C=1, word(1,3)+char_wb(2,5), transductive vocab, IW clip [0.1,10] | The safe bet: 5-fold proxy 0.7721 (verified by independent re-run, wins all 5 folds), most-trusted projection 0.7641 via the −0.008 rule. |
| 4 | predictions/Task3_Blend_Prediction.csv | Single-seed (42) blend, w=0.5 | The exact artifact validated on folds; differs from #1 by only 198 labels — submit if slots remain to measure the seed-ens delta. |
| 5 | predictions/Task3_Transformer_Prediction.csv | DistilBERT-256 (yesterday's run) | Completes the 256-vs-448 truncation comparison. Alternatively HOLD this slot. |

Decision guide once scores land:
- Every new score vs 0.72990 is kept automatically by Kaggle (keeps team best) — probes cost nothing but a slot.
- #2 vs #1 isolates the blend gain; #2 vs 256-ver (0.72990-era models) calibrates transformer deflation for the report.
- If #3 lands near its 0.7641 projection, the calibration rule is confirmed a third time — strong report material either way.
- NOT slated: Task3_SparseScreen (ties #3's family, weaker), Task3_Composed (within noise of #3), Task3_LengthAdapt (0.7504 proxy, dominated), Task3_TreeModels (projected ~0.73 by the −0.075 rule).
