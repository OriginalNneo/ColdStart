# Task 3 — Model Refinement & Testing Report (Iteration Ledger)

**Project:** SUTD 50.007 Machine Learning — GenAI academic-abstract detection
**Kaggle:** `50-007-machine-learning-may-2026` · metric **Macro F1** · team **Cold Start**
**Constraint:** Classical ML only — **no deep learning**.

> **Recording rule (per project CLAUDE.md):** this file is an **append-only ledger**.
> Each iteration/try is a dated block with its real result. Earlier iterations are
> never overwritten — new tries are appended so the full progression stays auditable.

---

## Executive summary

| # | Iteration / try | Model | Real Kaggle | Result |
|---|---|---|---|---|
| 0 | Baseline (proven) | LinearSVC · word(1,2)+char_wb(3,5) | **0.72990** | reference |
| 1 | XGBoost stack | LogReg+SVM+RF → XGBoost meta | 0.69924 | ✗ below baseline |
| 2 | Representation sweep | 3/4-grams, char-only, wide-char, sentence | offline only | ✗ no transfer edge |
| 3 | Meta hill-climb | coordinate descent on XGB meta | offline 0.7847 | ✗ noise, projects 0.707 |
| 4 | Markov perplexity | char n-gram LM LLR features | offline only | ✗ robust but too weak |
| 5 | **Refinement r1 — wideA** | LinearSVC · word(1,2)+char_wb(2,6) | **0.73370** | ✓ **beat baseline +0.004** |
| 6 | **Refinement r1 — wideB** | LinearSVC · word(1,3)+char_wb(2,6) | **0.74477** | ✓✓ **best eligible, +0.015** |
| 7 | Refinement r2 (C=0.50) | LinearSVC · word(1,3)+char_wb(2,6) C=0.50 | 0.74363 | ✗ passed 2-lens but did not transfer |
| 8 | Refinement r3 (threshold) | wideB + tuned Macro-F1 decision threshold | not submitted | ✗ failed 2-lens transfer test |

**Current best eligible classical model: wideB = 0.74477** (`predictions/Task3_Refined_Prediction.csv`) — plateau reached; Iter 7's offline win did not transfer.

---

## The governing fact: train→test topic shift

Train and test cover different research-topic mixes. This makes ordinary cross-validation
misleading (it overstates the real leaderboard by ~0.09) and it is the reason nearly every
"improvement" failed. Two validation lenses are used throughout:

- **Lens A** — `cluster_folds`: word-unigram KMeans, hold out whole topic clusters (proven; calibrated ~−0.008 to the real LB for same-family single models).
- **Lens B** — char_wb(3,5) KMeans k=16, seed 2026 (built to be independent of Lens A).

**Two-lens rule:** accept a refinement only if it beats the current best on **both** lenses.
This defeats the winner's curse that sank every single-lens "win."

---

# ITERATION LEDGER (append-only)

## Iter 0 — Baseline (reference) · real Kaggle 0.72990
`LinearSVC(C=0.25, class_weight="balanced")` on word(1,2)+char_wb(3,5) TF-IDF. The standing
1st-place *eligible* model. Vanilla CV ~0.82, Lens A 0.7404 — real 0.72990.

## Iter 1 — XGBoost stacking ensemble · real Kaggle 0.69924 · ✗
`Task3_XGBStack.py`. LogReg + LinearSVM + RandomForest base learners → out-of-fold scores →
XGBoost meta-judge (Gaussian noise injected as regularizer).

| Model | vanilla | Lens A |
|---|---|---|
| LogReg | 0.8058 | 0.7251 |
| LinearSVM | 0.8189 | 0.7404 |
| RandomForest | 0.8162 | 0.7465 |
| XGB-STACK | 0.8490 | 0.7813 |

Beat every base offline, so it was submitted → **0.69924, below baseline.** The proxy over-predicted by ~0.08 (the ensemble deflation). Confirms: stacking adds capacity that overfits the train topics.

## Iter 2 — Representation sweep · offline · ✗ (no transfer edge)
`Task3_RepCompare.py`, single LinearSVC, Lens A (gap = train−cluster):

| Representation | vanilla | Lens A | gap |
|---|---|---|---|
| word(1,2)+char(3,5) base | 0.8189 | 0.7404 | +0.252 |
| word(1,3)+char(3,5) | 0.8222 | 0.7397 | +0.255 |
| word(1,4)+char(3,5) [4-gram] | 0.8211 | 0.7375 | +0.257 |
| char(3,5) only | 0.8017 | 0.7232 | +0.215 |
| word(1,2)+char(2,6) wide | 0.8223 | 0.7450 | +0.248 |
| base + sentence/stylo | 0.8197 | 0.7424 | +0.250 |

Richer word n-grams raised vanilla but lowered Lens A and widened the gap. char-only was most
robust (smallest gap) but weakest. **wide-char(2,6) was the one representation above base** — the seed for Iter 5/6.

## Iter 3 — XGBoost meta hill-climb · offline 0.7847 · ✗
`Task3_XGBStack_Search.py`. Two coordinate-descent passes over all XGB meta knobs + noise +
leg-subset. Best moved 0.7813 → **0.7847 (+0.003 = noise)**; projects ~0.707 real. Meta-tuning
is a non-lever; the offline surface is flat and every point projects below baseline.

## Iter 4 — Markov / n-gram perplexity detector · offline · ✗
`Task3_MarkovPerplexity.py`. Hand-rolled class-conditional char 4-gram LMs (stupid-backoff, no
NN); features = per-doc log-prob / perplexity / LLR(machine−human).

| Config | vanilla | Lens A | gap |
|---|---|---|---|
| Markov feats only (SVC) | 0.7618 | 0.7084 | +0.199 |
| Markov feats only (LogReg) | 0.7612 | 0.7084 | +0.199 |
| baseline TF-IDF + Markov feats | 0.7982 | 0.7287 | +0.265 |

The LLR **cleanly separates classes** (human −0.40 vs machine +0.29) and is the **most topic-robust
signal found** (smallest standalone gap, +0.199) — but 5 features can't match a full vocabulary,
and appending them naively distorted the linear SVM (0.7404 → 0.7287). Right shape, insufficient
magnitude.

## Iter 5 — Refinement round 1: wideA · real Kaggle 0.73370 · ✓ (first to beat baseline)
`Task3_Refined.py`. Same baseline architecture, only the char range widened **(3,5) → (2,6)**;
word(1,2) unchanged. Two-lens validated:

| Candidate | Lens A | Lens B |
|---|---|---|
| base word(1,2)+char(3,5) | 0.7404 | 0.7563 |
| **wideA word(1,2)+char(2,6)** | **0.7450** ✓ | **0.7593** ✓ |

Submitted wideA → **0.73370** (+0.0038 over baseline). Lens A 0.7450 − 0.011 ≈ 0.734, matching the
same-family calibration — it transferred because it stayed in-family (no ensemble penalty) and
cleared both lenses. `predictions/Task3_Refined_wideA_Prediction.csv`.

## Iter 6 — Refinement round 1: wideB · real Kaggle 0.74477 · ✓✓ (new best eligible)
`Task3_Refined.py`, second two-lens passer: word range also widened **(1,2) → (1,3)**.

| Candidate | Lens A | Lens B |
|---|---|---|
| **wideB word(1,3)+char(2,6)** | 0.7439 ✓ | **0.7640** ✓ |

Submitted wideB → **0.74477** (+0.0149 over baseline, +0.011 over wideA). It **over-performed** its
Lens-A projection (≈ zero deflation, landing between the two lenses). Word trigrams — which *hurt*
on the base char(3,5) config (Iter 2) — *help* once paired with wide char(2,6): the interaction
matters. This classical model now sits within ~0.007 of the #2 team (0.75607) and the project's own
ineligible transformer (0.75186). `predictions/Task3_Refined_Prediction.csv` (new best eligible).

## Iter 7 — Refinement round 2 (in-family push around wideB) · real Kaggle 0.74363 · ✗ (did not transfer)
`Task3_Refined2.py`. Pre-registered in-family variants around wideB, accepted only if they beat
wideB on **both** lenses.

| Candidate | Lens A | Lens B | Beats wideB both? |
|---|---|---|---|
| wideB ref  w(1,3) c(2,6) C0.25 | 0.7439 | 0.7640 | — |
| R2a  w(1,4) c(2,6) C0.25 | 0.7412 | 0.7635 | ✗ (word 4-grams hurt) |
| R2b  w(1,3) c(2,7) C0.25 | 0.7419 | 0.7634 | ✗ (wider char hurts) |
| **R2c  w(1,3) c(2,6) C0.50** | **0.7507** | **0.7662** | ✓✓ passed guard |
| R2d  w(1,4) c(2,7) C0.25 | 0.7409 | 0.7619 | ✗ (both-wider, worst) |

R2c (only the regularization loosened, C 0.25→0.50) was the sole two-lens passer, so it was
submitted → **0.74363 — marginally BELOW wideB's 0.74477 (−0.0011).**

**Honest finding (nuanced):** R2c landed almost exactly on its same-family projection (Lens A 0.7507
− 0.008 ≈ 0.743; actual 0.74363). It didn't "fail" — it delivered what the proxy predicted. What
actually happened is that **wideB *over*-delivered** its own projection (projected ~0.736, real
0.74477), so the two cluster together at ~0.744 and wideB is nominally on top by 0.0011 (inside the
noise band). The takeaway is not "C=0.50 is bad" but "**we are on a flat plateau at ~0.744; the
in-family levers (n-gram range, C) are exhausted and no longer produce real, separable gains.**"
wideB (C=0.25) stays the designated best. Further improvement, if any exists, will not come from
LinearSVC hyperparameters.

---

## Iter 8 — Refinement round 3: decision-threshold tuning for Macro-F1 · NOT submitted · ✗
`Task3_Refined3_threshold.py`. wideB uses the default decision boundary (0); since the metric is
Macro-F1 on imbalanced classes, the optimal threshold under the shifted test may differ. Tuned the
threshold on wideB OOF decision scores, two-lens, with a transfer test.

| Threshold | Lens A | Lens B |
|---|---|---|
| default t = 0 | 0.7439 | 0.7640 |
| Lens-A-optimal t = −0.140 | **0.7541** (+0.010) | 0.7563 (−0.008) |
| Lens-B-optimal t = −0.020 | 0.7463 | 0.7644 |
| robust single t* (max of min) = −0.140 | 0.7541 | 0.7563 |

**No single threshold beats t = 0 on *both* lenses** — the shift that helps Lens A (+0.010) hurts
Lens B. This is a lens-specific mirage: single-lens tuning would have "found" +0.010 here and it
would have deflated on submission (exactly the Iter-7 pattern). The two-lens guard refused it, so the
last Kaggle slot was **not** spent. **Threshold tuning does not transfer; plateau re-confirmed.**

## Iter 9 — 8-track refinement campaign + lever stack · real Kaggle 0.75210 · ✓✓✓ (new best eligible, beats our transformer)
`scratch_campaign_plan.md`, harness `scratch_lens.py` (two-lens anchor A 0.7515 / B 0.7467 on
identical cached fold indices). Ran 8 parallel classical tracks, each two-lens gated. Seven were
null (consistent with the plateau): NBSVM reweighting (best −0.007), topic-word pruning /
max_df / chi² / adversarial-drop (best single-lens +0.0004), length/truncation (full text best),
shipped-5000-feat + adv-drop (none pass), external HF augmentation (none pass). **Two tracks broke
the plateau**, both minimal in-family single-model changes (lowest-deflation family):

- **Track 4 — estimator geometry.** Swapping LinearSVC → `RidgeClassifier(alpha≈0.85–1.0, balanced)`
  on the same rep passed both lenses; re-validated at **full uncapped resolution** (α=0.85 → A +0.0055,
  B +0.0059; smooth PASS plateau over α∈[0.5,2.0], not a spike).
- **Track 3 — block reweighting.** Scaling the **word** block ×1.6 (char ×1.0) passed both lenses
  (A +0.0060, B +0.0060). Note: contradicted the track's own "char is more topic-robust" hypothesis —
  empirically the *word* block wants more relative weight. (Char-block upweight failed entirely.)

**Decisive follow-up (`scratch_lensC_combine.py`):** tested whether the two levers stack, on FOUR
lenses — A, B, plus two new independent ones: C1 = word(2,3) KMeans holdout (independent topic basis),
C2 = adversarial test-similarity quintile holdout (mimics the real train→test covariate shift).

| candidate | A Δ | B Δ | C1 Δ | C2 (shift) Δ | min |
|---|---|---|---|---|---|
| Ridge α0.9 | +0.0055 | +0.0050 | +0.0053 | +0.0041 | +0.0041 |
| word×1.6 SVC | +0.0062 | +0.0064 | +0.0062 | +0.0060 | +0.0060 |
| **Ridge0.9 + word×1.6 (STACK)** | +0.0063 | **+0.0094** | **+0.0091** | +0.0074 | **+0.0063** |
| stylo_fusion (dense leg) | +0.0184 | +0.0215 | +0.0156 | +0.0194 | +0.0156 |

The two levers **stack additively** (min-margin +0.0063, 4/4 lenses incl. the shift-probe). Submitted
the stack — `RidgeClassifier(alpha=0.9, balanced)` on `[1.6·word(1,3) | char_wb(2,6)]` uncapped TF-IDF
→ **real Kaggle 0.75210** (`predictions/Task3_StackRidgeWord16_Prediction.csv`, submission 54839212).
**+0.00733 over wideB; over-delivered the ~0.750 projection** (like wideB did), and is the **first
eligible classical model above our own ineligible transformer (0.75186).** New designated best.

**Open thread — stylo_fusion (NOT submitted):** `scratch_agent5_stylo.py` fused a 227-dim
topic-invariant dense block (function-word rates, punctuation/sentence-length stats, hedging, passive
proxy; StandardScaler fit train-only, no leakage) with wideB → proxy +0.018 on ALL four lenses,
including C2 (+0.0194). The expected stylometric collapse on the shift-probe **did not reproduce** —
so the only remaining argument against it is the ledger's prior real-submission calibration (dense/
stylometric legs deflated −0.075..0.08; cf. the 0.71995 stylometric submission). Outcome is genuinely
bimodal (~0.69 if that deflation holds, ~0.76 if it doesn't). Deferred as the highest-value gamble for
a future slot; C2 (a within-train proxy) can neither confirm nor rule out the documented deflation.

## Iter 10 — Reframe to shift-recovery: forensics + transductive self-training · offline · candidate QUEUED (not yet submitted)
**Trigger:** user intel (direct contact with the #1 team) that the 0.795 leaders are **classical, not
deep learning** — they have an *edge*, not a bigger model. This overturns the prior "classical ceiling
~0.76" assumption and reframes the search from "refine the estimator by thousandths" to "find the
shift-robustness edge." (5/5 submissions already used this UTC day on the Iter-9 stack 0.75210; the
below is offline prep for the next reset.)

**Headroom quantified (`scratch_selftrain.py`).** Vanilla random-5fold CV of the stack = **0.8330**.
So the topic-shift tax = 0.8330 − 0.75210 real ≈ **+0.081 — this is the prize.** The classical
ceiling *if the shift were neutralized* is ~0.833, **above the 0.795 leaders.** The gap is entirely
distribution shift, and it is recoverable classically.

**Forensics (`scratch_forensic.py`) — ruled out cheap leaks.** No ID/ordering leak (UUID ids, zero
train∩test id overlap, non-monotonic). Shipped `train_features.csv`/`test_features.csv` = a plain
5000-dim TF-IDF (weaker than our 1.2M-dim word+char rep; Track-7 null explained). No single killer
feature (strongest surface signals: `n_paren` d=−0.30 humans use more parens, `avg_wlen` d=+0.37,
`newline`/`nonascii` moderate). Token "AI-tells" weak (furthermore +0.048, comprehensive +0.036).
FREE answers found: **19 test rows exactly match a train text** (single-label), **224 (3.2%)** match
on first-120-char hash — folded in as post-processing overrides (0 conflicts with the self-train pred).

**Transductive self-training (`scratch_selftrain.py`, `_tune.py`, `_final.py`) — the new edge.**
Fit stack on labeled train → pseudo-label the most confident test rows → refit; repeat. Attacks the
shift directly by adapting the boundary to the test topic/style. Validated on a clean non-circular
protocol (held-out cluster split into an unlabeled pool for self-training + a disjoint eval never
pseudo-labeled). Tuned over confidence-fraction × rounds × class-balanced selection:

| config | mean Δ (A+B) | worst fold | A | B |
|---|---|---|---|---|
| frac0.5 bal r1 | +0.0070 | −0.0054 | +0.0074 | +0.0066 |
| frac0.7 bal r1 | +0.0122 | −0.0046 | +0.0125 | +0.0119 |
| **frac0.7 bal r3** | **+0.0124** | **−0.0007** | +0.0120 | +0.0127 |
| frac0.9 bal r1 | +0.0108 | −0.0020 | +0.0106 | +0.0110 |

Best = **frac 0.7, class-balanced pseudo-labels, 3 rounds → +0.0124 on BOTH lenses, worst fold ≈ 0**
— the largest and most stable lever in the whole ledger. Class-balancing matters (prevents majority
drift); frac 0.7 is the sweet spot. Generated the real-test prediction uncapped
(`predictions/Task3_SelfTrain_Prediction.csv`): 427 test rows flipped vs the base stack, machine-frac
converged 0.620→0.596. **Projects ~0.764 if the +0.0124 transfers — and the real test gives a far
larger unlabeled pool (6999 rows vs ~500/cluster in the proxy), so the real gain may exceed the proxy.**
**QUEUED as the #1 submission for the next reset; empirical LB confirmation pending.**

**Round-2 stack refinement (`scratch_round2.py`), 4-lens gated.** Stack micro-tuning is flat: best
(α=0.8, word×1.5) min-margin +0.0071 vs current (α=0.9, word×1.6) +0.0063 — within noise, no change.
**stack+stylo (dense leg, scale 0.5) = +0.0218 min across all 4 lenses (highest proxy of anything);
stylo adds *on top of* the stack** (complementary, not redundant). Still the unresolved high-deflation
class (dense/stylometric leg, ledger −0.075..0.08) — bimodal, deferred as a gamble slot.

## Why the winners won and the losers lost (mechanism)

- **Losers (Iters 1–4)** all *added capacity or specialization* — stacking, richer word n-grams,
  Markov features. On a task whose difficulty is generalizing to unseen topics, extra capacity
  fits train topics harder (train-F1 → 0.99) and the shifted test punishes it. Plus the winner's
  curse: any config selected on one proxy is optimistically biased for that config.
- **Winners (Iters 5–6)** did the *opposite*: a **minimal, in-family** change (just a wider char
  n-gram range) that adds **shift-robustness, not capacity** — char(2,6) captures sub-morpheme
  2-grams and short-word-spanning 6-grams that lean on author *style* over *topic* — and it was
  only trusted after clearing **two independent lenses**. Staying in the baseline's single-model
  family kept the deflation at ~−0.008 instead of the ~−0.08 that sinks ensembles.

## Leaderboard context

Public LB (2026-07-19): son 0.79513 · Ghost 0.75607 · **Cold Start (us) 0.75186 (our own transformer, INELIGIBLE)**.
Our *eligible* classical best is now **0.74477 (wideB)** — the DL premium above us has nearly closed.
The teams ahead are almost certainly using deep learning; matching them *classically* past ~0.75 is
an open question the round-2+ ledger will keep testing.

## Appendix — scripts & artifacts
`Task3_XGBStack.py` · `Task3_XGBStack_Search.py` · `Task3_RepCompare.py` · `Task3_MarkovPerplexity.py` ·
`Task3_Refined.py` · `Task3_Refined2.py` · logs `scratch_*.log` · predictions in `predictions/`.
Submissions: XGB-stack 54820453 (0.69924) · wideA 54822048 (0.73370) · wideB 54822156 (0.74477).
