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

---

## Iter 11 (2026-07-20) — Stage 1 transductive shift-recovery, FOUR-LENS gated (`scratch_stage1_transduce.py`)

**Goal.** Push past self-training toward recovering more of the +0.081 topic-shift tax, and — for the
first time — judge transductive levers on ALL FOUR lenses (A/B/C1/C2), not just A/B. Note: the queued
self-training (Iter 10) was only ever validated on Lens A/B; this closes that gap.

**Levers (all vs the same base stack = RidgeClassifier(0.9,bal) on [1.6·word(1,3)|char_wb(2,6)]),
non-circular pool/eval protocol:**
- `selftrain` — validated frac0.7 / class-balanced / 3-round (reference)
- `iw` — covariate-shift importance weighting: reweight train rows by density ratio P(test-like)/(1−P)
  from a train-vs-target char_wb(3,5) domain classifier (target features only, no labels)
- `iw_selftrain` — the two combined

**Result (Δ macro-F1 vs base, mean over 5 folds/lens):**

| lever | Lens A | Lens B | Lens C1 | Lens C2 (shift-probe) | min | four-lens gate |
|---|---|---|---|---|---|---|
| selftrain | +0.0120 | +0.0127 | +0.0028 | **−0.0033** | −0.0033 | **FAIL** |
| iw | +0.0078 | +0.0021 | +0.0090 | **−0.0006** | −0.0006 | **FAIL** |
| **iw_selftrain** | +0.0142 | +0.0197 | +0.0121 | **−0.0005** | −0.0005 | **FAIL** |

**Verdict: no lever clears the strict four-lens gate — all three are marginally negative on C2.**
Honest, and it re-informs the queued Iter-10 bet:

1. **Self-training is weaker than the A/B picture implied.** On the topic lens C1 it is only +0.0028
   (one bad fold C1-f4 = −0.0337 nearly erases it), and on the shift-probe C2 it is **−0.0033**. The
   ~0.764 projection from the 2-lens +0.0124 is therefore less certain than it looked.
2. **IW is the shift-ROBUST lever, self-training the high-mean/high-variance one.** IW stays positive
   on exactly the hard folds where self-training craters (C1-f4: st −0.0337 vs iw +0.0037; C2-f0:
   st −0.0083 vs iw +0.0004). But IW alone is weak/fragile (negative on Lens B: −0.0095, −0.0111 folds).
3. **The combination Pareto-dominates the queued champion on all four lenses** (A .0142>.0120,
   B .0197>.0127, C1 .0121>.0028, C2 −.0005>−.0033): IW pulls self-training's C2 deflation from
   −0.0033 up to ≈0. So `iw_selftrain` is strictly the better transductive candidate than plain
   self-training — but it *still* just misses C2 (−0.0005, within fold noise).

**Mechanism read.** These levers recover +0.012..0.020 under whole-TOPIC-cluster holdout (A/B/C1) —
real signal against topical shift — but give ≈0 on the within-train test-likeness probe (C2). C2's
held-out "most test-like" train quintiles are still barely test-like (median P≈0.13), so the base is
already well-calibrated there and there is little for transduction to add. Which lens better predicts
the REAL test (topically shifted AND stylistically different) is the open question; the strict gate
treats C2 as a veto, so by discipline this is a FAIL, not a ship.

**Next.** Four-lens tuning sweep of `iw_selftrain` (IW clip/cap strength × self-train frac/rounds) to
convert the −0.0005 C2 miss into ≥0 while holding the A/B/C1 gains — the miss is sub-noise, so a
gentler config plausibly clears all four. If nothing clears C2, the transductive family is capped at
"topical-only" and Stage 1 pivots to co-training (1C) / label spreading (1B). No Kaggle slot spent.

---

## Iter 12 (2026-07-20) — Stage 1 tuning: `iw_selftrain` CLEARS the four-lens gate (`scratch_stage1_tune.py`)

**Goal.** Convert Iter-11's sub-noise C2 miss (−0.0005) into a clean four-lens pass by softening the
adaptation. Swept softened-IW gamma {0.0, 0.5, 1.0} × self-train frac {0.5, 0.7} × rounds {1, 3}, same
base stack / non-circular pool-eval protocol / all four lenses.

**Result — the fix is `frac=0.5` (trust FEWER, higher-confidence pseudo-labels):**

| gamma·frac·rounds | A | B | C1 | C2 | min | worst | gate |
|---|---|---|---|---|---|---|---|
| g0.5 f0.5 r1 | +0.0085 | +0.0094 | +0.0101 | +0.0021 | **+0.0021** | −0.0088 | PASS |
| g1.0 f0.5 r3 | +0.0120 | +0.0166 | +0.0099 | +0.0019 | +0.0019 | −0.0087 | PASS |
| **g1.0 f0.5 r1 (robust pick)** | +0.0113 | +0.0133 | +0.0130 | +0.0015 | +0.0015 | **−0.0078** | PASS |
| g0.5 f0.7 r1 | +0.0122 | +0.0149 | +0.0108 | +0.0009 | +0.0009 | −0.0053 | PASS |
| g1.0 f0.7 r3 (=Iter-11 iw_selftrain) | +0.0142 | +0.0197 | +0.0121 | −0.0004 | −0.0004 | −0.0190 | fail |
| g0.0 f0.7 r3 (=queued plain self-train) | +0.0120 | +0.0127 | +0.0028 | −0.0033 | −0.0033 | −0.0337 | fail |

**Verdict: 7 of 12 configs clear the strict four-lens gate; all passing configs use frac=0.5.** The
lever is real — first four-lens PASS since Iter 9. Mechanism: over-labeling (frac 0.7) injects
shift-wrong pseudo-labels that ding the C2 shift-probe; frac 0.5 keeps only the confident core, so C2
turns positive (+0.0015..+0.0021) while the topical gains hold (A/B/C1 +0.009..+0.017).

**Chosen config: `gamma=1.0, frac=0.5, rounds=1`** — picked for ROBUSTNESS not max-min (anti-winner's-
curse): best worst-fold among strong configs (−0.0078), strong topical gains (A+0.0113 B+0.0133
C1+0.0130), positive on the shift-probe (C2+0.0015), and a single round (least compounding / smallest
selection surface). This is full covariate-shift importance weighting + one conservative self-training
round. Strictly dominates the queued plain self-training (Iter 10) on all four lenses.

**Honest magnitude.** The four-lens min (+0.0015..+0.0021) is thinner than Iter-9's +0.0063, and the C2
gain is small — so the projected REAL increment is modest (order +0.003..+0.008 on top of 0.752 IF it
transfers; the big +0.010..+0.017 sits on the topical lenses whose real-test fidelity is the open
question). A genuine gated step, not a leap. Getting to 0.80 still needs additional axes stacked on top.

**Next.** (1) Generate the real-test prediction for g1.0/f0.5/r1 (ready for a future slot; no submission
now). (2) Continue Stage 1 offline: co-training (1C) and the user-requested tree/GBM test — both
four-lens gated, logged append-only. No Kaggle slot spent this iteration.

---

## Iter 13 (2026-07-20) — Stage 1 (1C) co-training word↔char — NULL (`scratch_stage1_cotrain.py`)

**Idea.** Use the two natural views (word(1,3), char_wb(2,6)) to cross-check pseudo-labels so only
higher-precision pool rows get labeled — hoping cleaner labels survive the C2 shift-probe where
single-view self-training deflated. Two variants: `agree_st` (label a pool row only if BOTH views
agree AND both are top-frac confident; refit stack) and `cotrain` (classic: each view teaches the
other for rounds; final stack on the union). Same base / protocol / four lenses.

**Result (Δ vs base):**

| cand | A | B | C1 | C2 | min | worst | gate |
|---|---|---|---|---|---|---|---|
| agree_st | −0.0008 | +0.0042 | −0.0031 | −0.0003 | −0.0031 | −0.0266 | **fail** |
| cotrain | +0.0011 | +0.0014 | −0.0045 | −0.0031 | −0.0045 | −0.0323 | **fail** |

**Verdict: NULL — co-training is strictly worse than `iw_selftrain` (Iter 12) on every lens** and has
worse worst-folds (−0.027, −0.032), cratering on exactly the hard folds A4/C1-f4 where the base is
weakest. The agreement gate labeled fewer rows but did NOT add robustness — without IW's covariate-
shift reweighting there is nothing stabilizing the hard folds. **1C dead-ended; the working Stage 1
lever remains IW-reweighting + conservative self-training (Iter 12, g1.0/f0.5/r1).**

---

## Iter 14 (2026-07-20) — Decision-tree / GBM legs (user request) — DEFLATE, halted early (`scratch_stage1_trees.py`)

**Why run it.** User asked to try decision trees. They ARE eligible (classical, not DL), so the honest
move is to re-test the ledger's tree-deflation prior on the CURRENT strong base + four shift lenses,
rather than refuse. Candidates: `gbm_lsa` (HistGradientBoosting on TruncatedSVD-256 of the stack),
`rf_lsa` (RandomForest-400 on the same), `stack_gbm` (average linear-stack + gbm margins — trees as a
complementary leg), `tree_calib` (shallow depth-3 tree on [oof stack score, log length] — low-capacity
shift-aware relabel). Non-circular protocol, eval half of each held-out cluster.

**Result — complete Lens A (5 folds), then HALTED (uniform, decisive, saves ~30 min compute):**

| candidate | Lens A Δ vs base |
|---|---|
| gbm_lsa | **−0.0599** |
| rf_lsa | **−0.0948** |
| stack_gbm (linear+gbm avg leg) | −0.0206 |
| tree_calib (shallow, [score,len]) | −0.0238 |

Per-fold the loss was uniform and often catastrophic on the EASY folds (A4: base 0.779 → gbm 0.611
(−0.168), rf 0.580 (−0.199)); trees only occasionally helped a hard fold (A3 gbm +0.021) but nowhere
near enough to net positive. **All four tree variants deflate hard — fails the gate before Lens B is
even needed. Halted after a full, unanimous Lens A.**

**Interpretation (confirms the governing mechanism live).** Trees on this task fit train topics and the
shifted eval punishes them — the exact failure that made Iter-1's XGBoost stack our worst submission
(0.69924). Notably the `stack_gbm` leg *drags the strong linear model DOWN* (−0.0206), i.e. the tree
leg is not complementary — it is strictly worse and correlated-wrong. Even the low-capacity `tree_calib`
loses, because a depth-3 step function on the decision score is coarser than the linear threshold it
replaces. **Decision-tree / GBM approaches are ruled OUT for this problem, on evidence, on the current
base. The linear stack + transductive shift-recovery (Iter 12) remains the path.**

---

## Iter 15 (2026-07-20) — Stage 2: topic-invariant LLR/style bank CLEARS four-lens gate (`scratch_stage2_features.py`)

**Idea.** Add NEW signal that survives topic shift (not more capacity, not transduction): a 16-dim
topic-invariant feature bank fused as a small dense leg on the base stack —
class-conditional char(3,4) & word(1,2) n-gram mean log-prob under human-LM vs machine-LM + their LLR
(12 feats), gzip compression ratio, type/token ratio, repeated-bigram rate, hapax ratio. Revives the
Iter-4 "perplexity/LLR is the most topic-robust signal but 5 feats can't match a vocabulary" thread —
fixing BOTH magnitude (16 feats) and fusion (StandardScaled, small scale).

**Key mechanism discovered — the OOV-backstop.** In-distribution the bank HURTS (LLR is linearly
redundant with the TF-IDF n-gram span, so it only adds scale-mismatch noise: random split −0.013..−0.039
across scales). But UNDER TOPIC SHIFT the redundancy breaks — test docs are full of OOV n-grams whose
TF-IDF is ~zero, yet the LLR/style statistics still compute a meaningful score. So the bank helps
precisely where the sparse stack goes quiet. Scale is critical: it must be TINY (StandardScaled × 0.02)
or it distorts the ridge.

**Result (Δ vs base, four lenses, FULL-cluster holdout = Iter-9 anchor protocol):**

| scale | A | B | C1 | C2 | min | worst fold | gate |
|---|---|---|---|---|---|---|---|
| **s=0.02** | +0.0112 | +0.0106 | +0.0106 | +0.0050 | **+0.0050** | **−0.0009** | **PASS** |
| s=0.03 | +0.0106 | +0.0079 | +0.0098 | +0.0020 | +0.0020 | −0.0079 | PASS |
| s=0.04 | +0.0059 | +0.0045 | +0.0063 | −0.0010 | −0.0010 | fail |

**`llr_s0.02` is the best-quality lever of the campaign since Iter 9:** min-margin +0.0050 (vs Iter-9
stack +0.0063, iw_selftrain +0.0015) with worst fold only −0.0009 — essentially non-negative on all 20
folds. Crucially POSITIVE on the C2 shift-probe (+0.0050), unlike the whole transductive family.

**Complementary to transduction.** The bank is +0.0164 on C1-f4 and +0.0087 on B-f4 — the exact hard
folds where iw_selftrain (Iter 12) and co-training (Iter 13) CRATERED (−0.019, −0.032). Different signal,
opposite failure modes → they should STACK. Next experiment: bank(s0.02) + iw_selftrain combined.

**Caveat (honest).** It is still a dense leg — technically the class the ledger flags for real-submission
deflation (−0.075..0.08 for the 227-dim stylo block). BUT this is 16 feats at scale 0.02 — a minor
perturbation of the single linear model, orders of magnitude smaller capacity than the stylo block or an
ensemble — so its deflation risk is far lower, closer to the ~−0.008 same-family class. C2 passing is
strong evidence but not proof (C2 is a within-train proxy); the LB is the final arbiter. No slot spent.

---

## Iter 16 (2026-07-20) — STACK: topic-invariant bank + iw_selftrain — best gated candidate ever (`scratch_stage2_combine.py`)

**Idea.** Iter-12 iw_selftrain (transductive) and Iter-15 llr_s0.02 (topic-invariant bank) have OPPOSITE
failure modes, so test whether they ADD. Four lenses, non-circular pool/eval protocol, base stack +
StandardScaled 16-feat bank ×0.02 + IW weighting + 1 round frac0.5 self-train.

**Result (Δ vs base):**

| candidate | A | B | C1 | C2 | min | worst fold | gate |
|---|---|---|---|---|---|---|---|
| iwst (Iter 12) | +0.0113 | +0.0133 | +0.0130 | +0.0015 | +0.0015 | −0.0078 | PASS |
| bank (Iter 15) | +0.0116 | +0.0087 | +0.0123 | +0.0037 | +0.0037 | −0.0015 | PASS |
| **bank_iwst (STACK)** | **+0.0237** | **+0.0259** | **+0.0196** | **+0.0046** | **+0.0046** | **+0.0022** | **PASS** |

**bank_iwst is the strongest, most robust candidate in the whole ledger: min +0.0046 AND worst fold
+0.0022 — POSITIVE on all 20 folds** (no lever has ever been non-negative across all four lenses).
Additivity: A super-additive (+0.0008), B super-additive (+0.0039), C1 sub-additive (−0.0057), C2
sub-additive (−0.0007). Net: the two levers ADD cleanly on the TOPICAL lenses (A/B/C1 → +0.020..0.026);
on the C2 shift-probe the bank dominates (transduction ≈0 there, as established). Complementarity is
real — on the hard folds where iwst goes negative (A-f3, B-f4) the bank rescues the combination.

**Honest projection.** The gate metric is the C2 min (+0.0046, shift-faithful, bank-driven) → a
conservative real ≈ +0.004..0.005 over 0.752 (~0.756). The big topical gains (~+0.02) would push higher
(~0.76..0.765) IF the real test's topical shift resembles the cluster lenses — the open question the LB
resolves. Deflation class: it is now stack + tiny dense leg + transduction; the dense leg is 16 feats at
scale 0.02 (minor perturbation, not the 227-dim gamble), so expected deflation is closer to the ~−0.008
same-family band than the −0.08 ensemble band — but this is the first candidate combining three
mechanisms, so treat the projection as a range, not a point.

**Next.** Generate the combined real-test prediction (uncapped stack + bank leg + IW + self-train) →
`predictions/Task3_BankIWSelfTrain_Prediction.csv`. Recommend it as the next Kaggle submission — it is
the best-validated candidate and only the LB can calibrate the three-mechanism deflation. No slot spent yet.

### Iter 16 — REAL KAGGLE RESULT (submission 54845349, 2026-07-20): **0.77913** ✅ NEW BEST

`Task3_BankIWSelfTrain_Prediction.csv` → **public 0.77913**, up **+0.02703** from the prior best 0.75210.
This BLEW PAST the conservative projection (~0.756) and even the optimistic range (~0.765). Decisive findings:

1. **The topical lenses (A/B/C1) are the faithful real-test proxy, NOT C2.** Real gain +0.0270 matches the
   topical-lens proxy (+0.020..0.026), while the C2 shift-probe (+0.0046) badly UNDER-predicted. The real
   test is genuinely topic-shifted; the cluster-holdout lenses model it well, the within-train
   test-likeness probe does not. → For transductive + topic-invariant levers, trust A/B/C1; C2 is a floor.
2. **The dense-leg deflation fear did NOT materialize** — real gain (+0.027) EXCEEDED the topical proxy,
   i.e. ~zero (even negative) deflation, not the −0.075..0.08 feared for dense legs. Because the bank is
   TINY (16 feats × scale 0.02) it behaves like a same-family change, not an ensemble. This retires the
   deflation objection for small topic-invariant fusion — and materially de-risks the 227-dim stylo gamble.
3. **First eligible classical model to clear the field convincingly:** 0.77913 > Ghost 0.75607, > our own
   ineligible transformer 0.75186; now only ~0.016 below the leader son (0.79513) and ~0.021 below 0.80.

**Recalibration:** proxy→real deflation for the transductive + tiny-topic-invariant-bank family ≈ **0.000
(topical-lens basis)**. The +0.081 topic-shift tax is being recovered exactly as the headroom analysis
predicted. Path to 0.80 is now concrete: more topic-invariant signal (2A stylo now low-risk per finding 2)
+ Stage 4 shift-aware calibration, each 4-lens gated on the TOPICAL lenses primarily.

---

## Iter 17 (2026-07-20) — Stage 2b: add 227-dim stylo leg on top of bank_iwst (`scratch_stage2b_expand.py`)

**Idea.** Iter-16's LB (0.77913) retired the dense-leg deflation fear, so stack the 227-dim stylo block
(`scratch_agent5_stylo`, function-word/punctuation/sentence-structure) as a second topic-invariant leg on
top of the current best. Sweep stylo scale {0.01,0.02,0.04}, four lenses, gate on TOPICAL lenses (A/B/C1)
per the Iter-16 finding that they — not C2 — predict real.

**Result (Δ vs base; combined bank×0.02 + stylo×s + IW + frac0.5 self-train):**

| candidate | A | B | C1 | C2 | 4-lens min | topical min |
|---|---|---|---|---|---|---|
| bank_iwst (Iter 16 = 0.77913) | +0.0237 | +0.0259 | +0.0196 | +0.0046 | +0.0046 | +0.0196 |
| bankstylo_s0.01 | +0.0266 | +0.0277 | +0.0201 | +0.0053 | +0.0053 | +0.0201 |
| bankstylo_s0.02 | +0.0263 | +0.0302 | +0.0219 | +0.0060 | +0.0060 | +0.0219 |
| **bankstylo_s0.04** | **+0.0279** | **+0.0348** | **+0.0254** | **+0.0101** | **+0.0101** | **+0.0254** |

**Incremental stylo gain over bank_iwst (s0.04) is POSITIVE on all four lenses:** A +0.0042, B +0.0089,
C1 +0.0058, C2 +0.0055. Monotone in scale (0.04 > 0.02 > 0.01); larger scale helps most on the HARD/shift
folds (B-f4 +0.0199, C1-f1 +0.0160). Stylo captures signal complementary to both the LLR bank and the
transductive lever. Four-lens min doubled (+0.0046 → +0.0101).

**Projection.** Using the Iter-16-calibrated relation (real ≈ topical proxy, deflation ≈0): topical mean
≈ +0.029 → real ≈ **0.784..0.786**. Config to ship: `bank(×0.02) + stylo(×0.04) + IW + frac0.5 1-round
self-train`, uncapped rep → `predictions/Task3_BankStyloIWSelfTrain_Prediction.csv`. Submitting (slots left
today: was 4). Note s0.04 is the top of the swept range and still rising — a higher-scale probe is the
obvious follow-up.

### Iter 17 — REAL KAGGLE RESULT (submission 54846095, 2026-07-20): **0.79080** ✅ NEW BEST

`Task3_BankStyloIWSelfTrain_Prediction.csv` → public **0.79080**, up **+0.01167** from Iter-16 (0.77913).
Again EXCEEDED the projection (~0.785) — the topical proxy keeps slightly under-predicting (real ≥ proxy,
i.e. NEGATIVE deflation for this dense-topic-invariant family). Leaderboard: son 0.79513 · **us 0.79080**
— now ~0.0043 behind #1, an eligible classical model at #2, only ~0.009 short of 0.80.

**Confirmed pattern over 3 submissions (Iter 9→16→17: 0.75210→0.77913→0.79080):** every topic-invariant
/ transductive lever that clears the TOPICAL lenses transfers to the LB at ≥ its topical-proxy margin.
The +0.081 shift tax is being recovered lever by lever. Stylo scale 0.04 was the TOP of the swept range
and still rising → Stage 2c probes higher scales (0.06/0.08/0.12) next.
Note: a teammate web submission `submission.csv` (0.66084) also landed today between ours.

---

## Iter 18 (2026-07-20) — Stage 2c: tune stylo scale to the peak (`scratch_stage2c_hiscale.py`)

**Idea.** Iter-17 stylo scale 0.04 was the top of its swept range and still rising. Probe higher scales
{0.04,0.06,0.08,0.12} on top of bank_iwst, four lenses, topical-gated.

**Result (Δ vs base):**

| candidate | A | B | C1 | C2 | 4-lens min | topical mean |
|---|---|---|---|---|---|---|
| bankstylo_s0.04 (Iter 17 = 0.79080) | +0.0279 | +0.0348 | +0.0254 | +0.0101 | +0.0101 | +0.0294 |
| bankstylo_s0.06 | +0.0323 | +0.0373 | +0.0268 | +0.0118 | +0.0118 | +0.0321 |
| **bankstylo_s0.08 (peak)** | **+0.0346** | +0.0368 | +0.0268 | +0.0136 | **+0.0136** | **+0.0327** |
| bankstylo_s0.12 | +0.0339 | +0.0358 | +0.0268 | +0.0140 | +0.0140 | +0.0322 |

**Peak at s=0.08** (surface flat 0.06–0.12; 0.08 has best Lens-A and best topical mean, four-lens min
+0.0136 = ~3× bank_iwst). Incremental stylo gain over bank_iwst at s0.08: A +0.0108, B +0.0109,
C1 +0.0072, C2 +0.0090 — positive on ALL four lenses. Topical mean +0.0327 vs Iter-17's +0.0294
(+0.0033 more). Projection via the calibrated transfer (real ≥ topical): **~0.795–0.80**.
Config: `bank(×0.02) + stylo(×0.08) + IW + frac0.5 self-train` → `Task3_BankStylo08_Prediction.csv`.
Submitting as the shot at 0.80.

### Iter 18 — REAL KAGGLE RESULT (submission 54847048, 2026-07-20): **0.79007** ✗ (slight regression)

`Task3_BankStylo08_Prediction.csv` → public **0.79007**, DOWN −0.00073 from Iter-17 (0.79080).
**First proxy improvement this session that did NOT transfer:** the proxy said s0.08 was +0.0033 topical
better than s0.04, but the LB says −0.0007 worse. → **Stylo scale is SATURATED around 0.04–0.06; finer
scale-tuning is now winner's-curse (proxy noise, not real signal).** Iter-17 (s0.04, 0.79080) REMAINS THE
BEST. Lesson: the topical-lens proxy is faithful for NEW signal / structural levers (Iters 16–17 transferred
at ≥1×), but NOT for fine hyperparameter tuning on a saturated axis — for that it overfits like vanilla CV.

**Implication for 0.80:** the cheap scale gains are exhausted. The remaining ~0.009 to 0.80 needs a
genuinely DIFFERENT signal source (richer/new topic-invariant features, stronger transduction, or
shift-aware calibration), each validated offline and submitted only if it clears the topical gate by a
MARGIN well above proxy noise (~>0.005), not by thousandths. Best submittable = Iter-17 `bankstylo_iwst`
(s0.04) = **0.79080**.

---

## Iter 19 (2026-07-20) — Stronger transduction on the strong base — SATURATED / NULL (`scratch_stage3_transduce2.py`)

**Idea.** Self-training was tuned on the weak Stage-1 base; retest transduction DEPTH (self-train rounds
{1,2,3} × frac {0.5,0.7}) + label-spreading on the strong bankstylo (Iter-17) base, four lenses, vs the
f0.5_r1 reference. Only submit if a config beats reference by topical margin >~0.005 (Iter-18 discipline).

**Result (Δ vs f0.5_r1 reference):**

| config | A | B | C1 | C2 | topical_min |
|---|---|---|---|---|---|
| f0.5_r2 | +0.0018 | +0.0006 | −0.0032 | −0.0002 | −0.0032 |
| f0.5_r3 | +0.0034 | +0.0009 | −0.0053 | +0.0001 | −0.0053 |
| f0.7_r1 | +0.0024 | +0.0021 | +0.0027 | +0.0013 | +0.0021 |
| f0.7_r2 | +0.0049 | +0.0032 | +0.0015 | +0.0007 | +0.0015 |
| labelspread | −0.137 | −0.122 | −0.159 | −0.093 | DEAD |

**Verdict: transduction is SATURATED — NULL.** Best config f0.7_r2 = topical mean +0.0031 / min +0.0015,
BELOW the reliable-transfer bar (~0.005) — the same sub-noise zone that made Iter-18 regress, so NOT
submitted. More rounds at frac0.5 go NEGATIVE on C1. Label-spreading catastrophic (−0.09..−0.16): SVD
reduction to feed the graph destroys the 1.2M-dim sparse signal the linear model needs.

**Bottom line for 0.80:** both tunable axes are now exhausted — stylo/bank SCALE (Iter 18) and transduction
DEPTH (Iter 19). The remaining ~0.009 to 0.80 is not reachable by tuning the current levers; it needs a
genuinely new signal source not yet found. **CONSOLIDATE at Iter-17 `bankstylo_iwst` = 0.79080** (eligible
classical #2, ~0.004 behind leader). Session result: 0.75210 → 0.79080, +0.0387, the +0.081 shift tax
recovered from +0.000 to +0.048 of its headroom.

---

## Iter 20 (2026-07-20) — Function-word syntactic-skeleton leg (`scratch_stage4_syntax.py`)

**Idea.** New signal orthogonal to all existing legs: function-word SEQUENCE n-grams = syntax/word-order,
topic-invariant. Skeleton = content words → '#', keep function words + punctuation; TF-IDF word(1,3) on
that. Added on top of bankstylo (Iter-17), four lenses, IW + frac0.5 self-train.

**Result (Δ vs Iter-17):**

| weight | A | B | C1 | C2 | topical_min |
|---|---|---|---|---|---|
| fw x0.5 | +0.0008 | +0.0007 | +0.0037 | +0.0032 | +0.0007 |
| **fw x1.0** | +0.0037 | +0.0035 | +0.0043 | +0.0020 | **+0.0035** |
| fw x1.5 | +0.0019 | +0.0015 | −0.0031 | −0.0046 | −0.0031 (distorts) |

**fw x1.0 is POSITIVE on all four lenses** (topical_min +0.0035) — unlike the saturated tuning axes
(Iter-18 scale, Iter-19 transduction) it is genuinely new *orthogonal* signal (a new feature TYPE, the
category that transferred at ≥1× in Iters 16–17). Below the strict 0.005 bar, so treated as a CALIBRATED
BET, not a sure win. Submitting fw x1.0 → `predictions/Task3_BankStyloFW_Prediction.csv`.

### Iter 20 — SUBMISSION BLOCKED (daily quota 5/5 exhausted, 2026-07-20 UTC)

`Task3_BankStyloFW_Prediction.csv` generated (bankstylo + fw skeleton ×1.0; 318 rows changed vs Iter-17,
machine_frac 0.6074) but Kaggle returned 400 — the shared account already used all 5 daily slots (3 ours:
Iter 16/17/18; 2 teammate web submissions `submission.csv`). **QUEUED as #1 for the next reset (00:00 UTC
2026-07-21).** Best confirmed remains Iter-17 `bankstylo_iwst` = 0.79080. Coordinate with teammate to avoid
burning shared slots. If Iter-20 regresses on the LB (it is only +0.0035 topical, a calibrated bet), revert
to Iter-17.

---

## Iter 21 (2026-07-20) — Does ADDING MORE legs keep growing the stack? — mostly NO (`scratch_stage5_addmore.py`)

**Question (user):** why did bankstylo_iwst win, and does adding more to the stack help? Ablation on top
of Iter-17: + fw (function-word syntax) vs + shape (word-SHAPE/casing n-grams) vs + fw+shape. Lens A
(5 folds; halted after — verdict decisive and uniform):

| candidate | Lens A Δ vs Iter-17 |
|---|---|
| + fw (function-word syntax) | **+0.0053** |
| + shape (casing/word-shape) | **−0.0022** |
| + fw + shape | +0.0002 (worse than fw alone by −0.0051) |

**Verdict: adding more helps ONLY if the leg is genuinely ORTHOGONAL.** fw (syntax over function words)
is orthogonal to the content char/word TF-IDF -> it adds (+0.0053). Word-shape/casing is ALREADY captured
by char_wb(2,6) -> redundant, HURTS (-0.0022), and dilutes fw when stacked (fw+shape < fw). The stack does
NOT grow by piling on feature types — the 1.2M-dim char/word rep + stylo already spans most easy signals.
Orthogonal-feature headroom is nearly exhausted: fw was the last genuinely-new leg (~+0.004 real, queued
as Iter 20).

**Why bankstylo_iwst won (0.79080), quantified:** three COMPLEMENTARY shift-robust legs, each adding
signal-not-capacity, stacked additively — transduction (adapt boundary to test dist, ~+0.012 topical) +
LLR bank (OOV-backstop: speaks when TF-IDF is silent on unseen topics, ~+0.010) + stylo (topic-invariant
syntactic style, ~+0.006). They transferred at >=1x because topical lenses are faithful and topic-invariant
legs don't deflate. Bottom line: ~0.791 (+ fw ~0.794 when submittable) is a genuine PLATEAU for this
representation; crossing 0.80 needs a fundamentally new orthogonal signal, not more of the same.

---

## Iter 22 (2026-07-20) — pseudo-POS morpho-syntactic leg: NEW orthogonal signal, big four-lens WIN (`scratch_stage6_pos.py`)

**Idea (user: build a new orthogonal signal).** No tagger available, so heuristic pseudo-POS: each content
word -> grammatical class by morphology/casing (VBG/VBD/RB/NNZ/JJ/NNP/NNS/NN), function words -> FW,
numbers -> CD, punctuation kept; n-gram the TAG SEQUENCE (syntactic-category rhythm). Purely rule-based +
TF-IDF + Ridge — NO deep learning. Tested ON TOP OF Iter-17 + fw (does it add beyond fw?).

**Result (Δ vs Iter-17; four lenses):**

| candidate | A | B | C1 | C2 | topical_min |
|---|---|---|---|---|---|
| iter17_fw (= Iter-20) | +0.0053 | +0.0027 | +0.0027 | +0.0015 | +0.0027 |
| fw + pos x0.5 | +0.0138 | +0.0097 | +0.0161 | +0.0111 | +0.0097 |
| **fw + pos x1.0** | **+0.0153** | **+0.0118** | **+0.0160** | **+0.0127** | **+0.0118** |

**fw+pos x1.0 is POSITIVE on all four lenses, topical_min +0.0118, topical mean +0.0144 over Iter-17** —
the biggest four-lens gain since the stylo leg (Iter-17). Pseudo-POS is genuinely ORTHOGONAL to fw (which
masks all content as '#'): it adds +0.010 *on top of* fw. pos Pareto-dominates pos0.5 on the full data.
Pos strongly separates certain topic clusters (B-f1 +0.031, C1-f2 +0.049 folds). This is NEW structural
signal (the transferring category), not tuning — high confidence it transfers.

**Projection:** topical mean +0.0144 over Iter-17 (0.79080) → real ≈ **~0.80-0.805** (crossing 0.80) via
the calibrated ≥1× transfer. Config: `bank×0.02 + stylo×0.04 + fw×1.0 + pseudo-POS×1.0 + IW + frac0.5
self-train` → `predictions/Task3_BankStyloFWPOS_Prediction.csv`. QUEUED for the 00:00 UTC reset (daily
quota exhausted today). This supersedes the queued Iter-20 fw-only as the #1 submission at reset.

### Iter 22 — SUBMISSION QUEUED (daily quota still 5/5 as of Jul 20 ~16:02 UTC)

`Task3_BankStyloFWPOS_Prediction.csv` generated (6999 rows, machine_frac 0.6347, 531 changed vs Iter-17,
NO deep learning). Submit attempt returned 400 (quota exhausted, no slot consumed). **QUEUED #1 for the
00:00 UTC reset (~8h)** — supersedes the fw-only Iter-20 (fw+pos strictly dominates: topical +0.0144 vs
fw's +0.0027 over Iter-17). Best confirmed remains Iter-17 0.79080 until this lands. Projected ~0.80-0.805.
