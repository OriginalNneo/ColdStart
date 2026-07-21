# ColdStart — Plan to 0.80–0.82 Macro-F1 (Distribution-Shift Recovery Campaign)

_Author: planning session 2026-07-20. Append-only discipline applies — this file is a living plan,
not a results ledger. All numbers land in `TESTING_REPORT.md`._

> **OUTCOME (2026-07-20, Iters 11–22):** plan executed. Best eligible model went **0.75210 → 0.77913
> → 0.79080** (#2, ~0.004 behind leader) via the distribution-shift-recovery stack
> (`bankstylo_iwst` = base stack + LLR bank ×0.02 + stylo ×0.04 + IW weighting + self-training). A new
> orthogonal **pseudo-POS syntactic leg** (Iter 22, four-lens topical +0.0144) projects **~0.80** and is
> QUEUED for the next quota reset. Validated lessons: topical cluster-lenses are the faithful real-test
> proxy (C2 = floor); tiny/structural topic-invariant legs don't deflate; levers stack additively; only
> genuinely *orthogonal* new signal helps (scale/transduction-depth tuning saturated at ~0.791). Trees,
> co-training, label-spreading, and casing/shape legs were ruled out. 0.80 is in reach; 0.82 was not
> (above the vanilla-CV ceiling / far past the leader). Full detail: `TESTING_REPORT.md` Iters 11–22.

---

## 1. Context — what we are actually fighting

Current best eligible model: **RidgeClassifier(alpha=0.9, balanced) on `[1.6·word(1,3) | char_wb(2,6)]`
uncapped TF-IDF → real Kaggle 0.75210** (Iter 9). Leaders sit at ~0.795 and — per direct team intel —
are **classical, not deep learning**.

The single most important number in this project:

| Signal | Value |
|---|---|
| Vanilla random-5-fold CV of the current stack | **0.8330** |
| Real leaderboard | **0.75210** |
| **Topic-shift tax (the gap)** | **≈ +0.081** |

**Reaching 0.80–0.82 = recovering 60–85% of that +0.081 tax.** The classical ceiling *if the shift were
neutralized* is ~0.833 — above the leaders. So the target is not fantasy, but it is **not reachable by
finding a "better classifier."** It is reachable only by making the model see, or adapt to, the test
distribution.

### Correcting three framing assumptions (so the campaign points the right way)

1. **"Ridge finds different local minima / we can find a higher level of ridge."**
   `RidgeClassifier` is a *convex* linear model — it has exactly **one** global optimum, no local minima.
   There is no deeper optimization to unlock inside it. What we *can* do is give it (a) more
   topic-invariant features, (b) a training distribution reshaped toward the test distribution, or
   (c) test-side pseudo-labels. Those are the levers — not the optimizer.

2. **"We can refine the classifier more."** The ledger is blunt: the classifier-geometry and n-gram
   levers are **exhausted** (Iters 5–9 rode a flat ~0.744–0.752 plateau; C, n-gram range, threshold all
   tapped out). Every attempt to *add capacity* (XGBoost stacking, richer word n-grams, tree meta-models)
   **lost real points** — capacity fits train topics harder and the shifted test punishes it. So we
   deliberately stop tuning the model and start adapting the *distribution*.

3. **"Clustering hasn't worked but there's a method."** Correct on both counts. Blind clustered *experts*
   failed. But clustering is still the right **instrument** — not to split the model, but to (a) build
   honest shift-aware validation (already done: Lens A/B/C1), and (b) drive per-region calibration and
   covariate-shift importance weighting (Stage 3). Clustering as *measurement and weighting*, not as
   *separate models*.

### The 40% / public-LB caveat (your adaptability concern)

Test is 6,999 rows. Kaggle scores a **public subset** of it live and holds the rest for the **private**
final — we do **not** know the public/private ratio. Consequences baked into this plan:
- **Never tune to public-LB wiggles.** A ±0.003 LB move on ~a few thousand rows is inside the noise band.
- **The four-lens cluster-holdout proxy is the judge, not the LB.** The LB only *confirms or refutes the
  deflation calibration* for a given model family.
- Prefer choices that are **robust across all four lenses' worst fold**, because the private set is a
  different topic mix than whatever the public subset happens to be.

---

## 2. Governing rules (unchanged — these are why we win)

- **No deep learning.** Classical only; transformer artifacts are ineligible.
- **Four-lens gate.** A candidate ships only if its *min margin across Lens A, B, C1, and C2 (shift-probe)*
  is **> 0** vs the current anchor. Single-lens wins have deflated repeatedly.
- **Deflation calibration** (decides trust): same-family single sparse-text model ≈ proxy − 0.008;
  ensembles / stacks / dense-stylometric legs ≈ proxy − 0.075..0.08.
- **Append-only ledger.** Every try → new dated Iter in `TESTING_REPORT.md`; never edit old numbers.
- **Kaggle: 5/day, confirm every submission with the user**, resets 00:00 UTC. Log proxy → projected → real
  each time to keep the calibration honest.
- Run everything with `.venv/bin/python` + `PYTHONPATH=.` (the system `python3` has no sklearn).

---

## 3. The campaign — staged, priority-ordered, each stage two/four-lens gated

The through-line: **transductive adaptation + topic-invariant signal + shift-aware calibration**, stacked
the way the winning levers stacked. Each item names its mechanism, expected class, and the harness call
that gates it (`eval_rep` / `eval_lens` in `scratch_lens.py` / `scratch_lensC_combine.py`).

> **Execution decision (user, 2026-07-20): build Stage 1 offline; spend no Kaggle slots yet; skip 0B.**
> Stage 0A stays queued (validated) but is NOT submitted right now — the immediate work is offline
> validation of Stage 1 transduction levers until one clears the four-lens gate.

### Stage 0 — Cash the two bets that are already validated but unsubmitted (queued, not now)

These exist as generated predictions; they cost only Kaggle slots + confirmation.

- **0A. Transductive self-training (SAFE, queued — hold for a slot).**
  `predictions/Task3_SelfTrain_Prediction.csv` (frac 0.7, class-balanced pseudo-labels, 3 rounds).
  Proxy **+0.0124 on both lenses, worst fold ≈ 0** — the largest, most stable lever in the ledger.
  Projects **~0.764**. The real test gives a 6,999-row unlabeled pool (vs ~500/cluster in-proxy), so the
  real gain may *exceed* the proxy. Remains queued #1 for whenever we next spend a slot.

- **0B. Stylo-fusion gamble — DECIDED: SKIP (user, 2026-07-20).** Not spending a Kaggle slot on the
  bimodal ~0.69/~0.76 bet. Consequence: the Stage 2 "does the dense leg deflate?" question is **decided
  offline via proxy** (deflation calibration + four-lens worst-fold) instead of by a live submission. Kept
  in `scratch_agent5_pred.csv` as a dormant option only.

### Stage 1 — Push transduction harder (the core of the +0.081 recovery)

Self-training alone recovered only ~+0.012 of +0.081 — most of the tax is still on the table. Each of these
is classical/eligible and directly targets the covariate shift:

- **1A. Covariate-shift importance weighting on the *current* base.** Reuse the Lens C2 domain classifier's
  `P(test-like)` per train doc as `sample_weight` when fitting the final Ridge, so training emphasizes the
  train docs that look like test. (Old `Task3_IW_*` predictions tried this on the *weak* pre-Iter-9 base —
  revive it on the new stack.) Cheap, in-family; gate on four lenses.
- **1B. Label propagation / spreading** (`sklearn.semi_supervised.LabelSpreading` / `LabelPropagation`) on a
  train+test graph over the sparse features — a stronger transductive signal than confidence self-training.
- **1C. Co-training across the two views we already have** — word block and char block pseudo-label *for each
  other*. Natural fit for `[word | char]`; classic, more robust than single-view self-training.
- **1D. Self-training variants** — more rounds, per-class thresholds, entropy/temperature regularization,
  and **self-training × importance-weighting combined**. Class-balancing is mandatory (unbalanced drifts
  to the majority; worst fold went −0.013 without it).

_Gate:_ any 1x candidate must clear four-lens min-margin > 0 vs the self-training anchor (once 0A lands),
and its projection must beat the live best after deflation. Log each.

**Iter-11 results (2026-07-20, `scratch_stage1_transduce.py`) — first four-lens test of the family:**
- **1A + 1D combined (`iw_selftrain`) is the best candidate: A +0.0142, B +0.0197, C1 +0.0121, but
  C2 −0.0005** → *misses the strict four-lens gate by sub-noise on the shift-probe.* It **Pareto-dominates
  plain self-training on all four lenses**, so it replaces plain self-training as the transductive pick.
- **Mechanism learned:** the transductive family recovers +0.012..0.020 against *topical* shift (A/B/C1)
  but ≈0 on the within-train test-likeness axis (C2). **IW is the shift-*robust* leg** (positive on the
  hard folds where self-training craters, e.g. C1-f4 st −0.0337 vs iw +0.0037); self-training is
  higher-mean but high-variance. Combining smooths the variance.
- **Live fork:** (i) a gentler-config tuning sweep (`scratch_stage1_tune.py`, softened-IW γ × frac ×
  rounds) to convert the −0.0005 C2 miss into ≥0; if it clears → shippable Stage 1 winner. (ii) If nothing
  clears C2 → the family is *topical-only*; pivot to **1C co-training** and **1B label-spreading** (different
  mechanism), and separately decide whether topical lenses (A/B/C1) are the more faithful real-test proxy
  than C2 (a validation-philosophy call for the user).

### Stage 2 — Topic-invariant feature signal (add signal, not capacity)

Only pursue aggressively if **0B shows the dense leg does NOT deflate**. These add signal that barely moves
under topic shift:

- **2A. Stylometry** — promote from gamble to core axis if 0B clears. Highest proxy on the board.
- **2B. Scale up the perplexity/LLR family.** Iter 4 found n-gram-perplexity LLR is the *most topic-robust
  signal measured* (smallest train−cluster gap, +0.199) but "5 features can't match a vocabulary." Fix the
  magnitude: a **bank of dozens** of style/burstiness features — per-POS perplexity, char-compression ratio,
  type-token curves, sentence-length distribution moments, function-word rates, hedging/passive proxies —
  fused (not naively appended) under the Ridge. "Right shape, now with magnitude."
- **2C. Exploit the length shift.** Test median length 1,723 vs train 1,146 is a concrete covariate-shift
  axis. Length-normalized features and length-stratified calibration (Stage 4), so the model doesn't read
  "longer" as "different class."

**Stage 2 RESULT (2026-07-20, Iters 15–16) — 2B works, 2A stays deferred.**
- **2B (LLR/style bank) CLEARS the four-lens gate at scale 0.02** (Iter 15): min +0.0050, worst −0.0009.
  Key mechanism = the **OOV-backstop** — the bank is redundant in-distribution (hurts on CV) but under
  topic shift the test n-grams are OOV (TF-IDF ~0) while the LLR/style stats still compute, so it fills in
  exactly where the sparse stack goes quiet. Scale must be TINY. `scratch_stage2_features.py`.
- **2B STACKS with Stage-1 transduction** (Iter 16, `scratch_stage2_combine.py`): `bank_iwst` = base +
  bank(×0.02) + IW + frac0.5 self-train → **four-lens min +0.0046, worst fold +0.0022 — positive on ALL
  20 folds**, the best-validated candidate in project history. Levers add on the topical lenses
  (A/B/C1 → +0.020..0.026); bank carries C2. Prediction: `Task3_BankIWSelfTrain_Prediction.csv`.
  **This is the recommended next Kaggle submission** — only the LB can calibrate the 3-mechanism deflation.
- **2A (227-dim stylo) still deferred** — we did NOT need it; the tiny 16-feat bank captures the
  topic-invariant signal at far lower deflation risk. Stylo remains the separate bimodal gamble.

### Stage 3 — Clustering done right (your request, as instrumentation)

Not clustered experts (repeatedly failed). Clustering as adaptation machinery:
- **3A. Per-cluster / per-region calibration.** Fit one global Ridge, then calibrate the decision offset
  *per test-cluster* (Platt/isotonic on holdout structure). Global threshold tuning failed because no single
  threshold beat both lenses (Iter 8) — a *region-conditional* one may.
- **3B. Cluster-conditional importance weighting** — 1A, but weights conditioned on cluster membership.
- Honest note: if 3A/3B don't clear the gate quickly, drop clustering — it is the most-failed idea here and
  gets a short leash.

### Stage 4 — Macro-F1-aware, shift-aware threshold/calibration

Metric is Macro-F1 with 62/38 imbalance, so the decision boundary matters. Global threshold tuning failed
(Iter 8); retry only as **shift-aware / per-region** offsets tuned on the **shift-probe holdout** (Lens C2),
never on vanilla CV. Four-lens gated.

### Stage 5 — Variance reduction (only if 1–4 plateau)

Multi-seed / multi-config **averaging of the same in-family model** (decision-function averaging, not a
tree/meta stack) to cut variance without adding capacity. Ensembles deflate — keep it strictly same-family
and four-lens gate the average, or skip.

---

## 4. Expected trajectory (honest milestones)

| Milestone | Lever | Confidence |
|---|---|---|
| **~0.764** | Stage 0A self-training | High — validated, worst fold ≈ 0 |
| **~0.77–0.78** | + Stage 1 transduction (IW / label-spreading / co-training), best one or two stacked | Medium |
| **~0.78–0.80** | + Stage 2 topic-invariant features **iff 0B says dense doesn't deflate** | Medium-low, gated on 0B |
| **0.80–0.82** | All axes stacking additively the way Iter-9 levers did, + shift-aware calibration | Stretch — this is a research bet, not a guarantee |

Straight talk: **0.764 is near-certain; 0.78 is realistic; 0.80–0.82 is the ambitious stretch** that
requires the transduction + topic-invariant + calibration axes to stack additively (they have precedent for
doing so — Iter 9's two levers added cleanly across four lenses) **and** the dense/perplexity signal to not
deflate on real submission. We find out which world we're in the moment 0B lands.

---

## 5. Validation & adaptability protocol (non-negotiable)

- **Judge on the four-lens cluster-holdout proxy + shift-probe. Never vanilla CV** (it over-reads by ~0.09).
- Every candidate: report **min margin across A/B/C1/C2** and **worst single fold**. Ship only if min > 0.
- Convert proxy → projected real via the deflation class, submit, then **record the actual** to re-fit the
  calibration. This is the loop that has kept us honest.
- **Do not chase public-LB noise.** Prefer worst-fold-robust choices for private-set safety.
- One idea per submission where possible, so each Kaggle result cleanly attributes to one lever.

## 6. Files to build on (reuse, don't rewrite)

- `scratch_lens.py` — `load_data`, `get_folds`, `eval_rep`, `wideB_vecs`, `ANCHOR`. Core harness.
- `scratch_lensC_combine.py` — Lens C1/C2 + lever-stacking eval (`eval_lens`, `lensC2_folds` shift-probe).
- `scratch_selftrain*.py` — validated self-training (0A) + tuning + real-pred generator.
- `scratch_agent5_stylo.py` — `_features` / `build_dense` 227-dim block for Stage 2A.
- `scratch_folds.npz` / `scratch_anchor.json` — cached identical splits + anchor; keep every experiment on them.
- New work → new `scratch_*.py`; new results → new dated Iter in `TESTING_REPORT.md` + memory update.

## 7. Verification (how we prove each step end-to-end)

1. `PYTHONPATH=. .venv/bin/python scratch_lens.py` — reproduce anchor (A≈0.7439, B≈0.7640) to confirm the
   harness/folds are intact before trusting any new number.
2. For each lever: new `scratch_*.py` calling `eval_rep`/`eval_lens` on the cached folds → print A/B/C1/C2
   margins + worst fold. Require four-lens min > 0.
3. Generate the real prediction CSV, confirm row count = 6,999 and id format matches `sample_submission.csv`.
4. Confirm with user → submit → record proxy/projected/real in `TESTING_REPORT.md` → update memory.
