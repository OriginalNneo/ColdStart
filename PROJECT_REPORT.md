# Detecting Machine-Generated Text: Full Technical Report

**Course:** 50.007 Machine Learning (May 2026) | **Team:** Cold Start
**Kaggle competition:** `50-007-machine-learning-may-2026` | **Metric:** Macro F1
**Standing public leaderboard score at time of writing: 0.72990 (2nd place; 1st = 0.78326)**

---

## 1. Problem and Data

The task is binary classification: given a scientific abstract, predict whether it was
written by a machine (label 1) or a human (label 0). We were given:

- `train.csv`: 20,000 abstracts with labels (62.5% machine / 37.5% human)
- `test.csv`: 6,999 unlabeled abstracts (note: the brief said ~5,000; we verified 6,999)
- `train_features.csv` / `test_features.csv`: the same rows as 5,000 precomputed
  TF-IDF features (term frequency-inverse document frequency; Salton & Buckley, 1988)

Two data quirks we discovered ourselves, which shaped the entire project:

1. **No dev set exists** despite the brief mentioning one. We substituted a 10%
   stratified holdout (and later k-fold schemes).
2. **The test set is distribution-shifted from train.** Test abstracts are ~25% longer,
   and a classifier trained to distinguish train rows from test rows (adversarial
   validation, a standard Kaggle diagnostic) reaches AUC 0.81 on character n-grams,
   far above the 0.5 of identically-distributed data. This is a covariate-shift setting
   in the sense of Shimodaira (2000). Every headline validation number in this project
   must be read through that lens; ignoring it cost us twice, as documented below.

---

## 2. Task 1: Logistic Regression from Scratch

### 2.1 Implementation

We implemented binary logistic regression in NumPy only: the sigmoid, the binary
cross-entropy loss, its analytic gradients, mini-batch gradient descent (Robbins &
Monro, 1951), and prediction. sklearn appears only as a labeled benchmark (its
`LogisticRegression`) and as a data-splitting utility, never as the submitted model.
Numerical safeguards: logits clipped to ±500 before the sigmoid, probabilities clipped
to [1e-9, 1−1e-9] inside the log.

### 2.2 Hyperparameter search and result

A grid over batch size, epochs, and learning rate settled on **bs=512, 500 epochs,
lr=10.0** (final training loss ≈ 0.30). The large learning rate is not a typo: on
L2-normalized-ish sparse TF-IDF rows, per-feature gradients are tiny and plain SGD
needs an aggressive step size.

| Model | Val macro F1 (90/10 stratified, seed 42) | Kaggle public |
|---|---|---|
| **Ours (from scratch)** | **0.7318** | **0.68578** |
| sklearn LogisticRegression (benchmark) | 0.7061 | n/a |

**PASS:** our implementation beats the library benchmark by +0.026 val F1.

### 2.3 Refinement round 1: FAIL (informative)

We attempted to push past 0.7318 with the standard toolkit, validating every candidate
across 5 random splits (seeds 42/7/123/2024/99) because the seed-to-seed spread alone
is 0.022:

| Lever | Mean Δ vs baseline (5 seeds) | Verdict |
|---|---|---|
| L2 regularization grid (1e-4 to 2e-2) | best +0.0014 (L2=5e-4) | inside noise; FAIL |
| Class weighting (inverse frequency) | ≈ 0 | FAIL |
| Adam (Kingma & Ba, 2015) + feature standardization | −0.01 to −0.02 | actively worse on sparse TF-IDF; FAIL |
| Learning-rate decay, 800 epochs | ≈ 0 | FAIL |
| Decision-threshold tuning (honest: threshold picked on an inner 85/15 split, never on val) | +0.0006 | FAIL |

A subtlety worth recording: naive L2 grids (λ ≥ 0.05) collapse the model, because
the effective end-of-training weight shrinkage compounds as exp(−lr·λ·epochs/bs); with
lr=10 the useful λ regime is 10× smaller than textbook defaults.

### 2.4 Refinement round 2: FAIL (ceiling confirmed)

Round 1 never touched the input representation, so we tested the three standard
TF-IDF preprocessing levers (Manning, Raghavan & Schütze, 2008, ch. 6). First we
checked the data: the provided rows are **not** unit-norm (row L2 norms 0.38-1.00,
consistent with a full TF-IDF matrix normalized before being truncated to 5,000
columns), so renormalization was a genuinely live lever.

Screen on the seed-42 split, then 5-seed robustness for the best candidate:

| Candidate | Screen F1 | 5-seed mean Δ vs baseline |
|---|---|---|
| Row L2 renormalization (lr ∈ {5,10,20}) | 0.7285 / 0.7253 / 0.7162 | worse at screen, not pursued |
| log1p then L2-normalize, lr=5 | 0.7311 | **+0.0002** |
| 3-seed probability-averaged ensemble of the above | n/a | −0.0005 |

Baseline 5-seed mean 0.7291, spread 0.0223. Nothing clears the noise band.

**Conclusion (both rounds):** 0.7318 is the practical ceiling of a linear
decision boundary on these fixed 5,000 features. Two exhaustive refinement rounds
converging on zero is itself the evidence: the model is capacity-limited, not
under-tuned. This also disposes of the idea of distilling a stronger model into the
logistic regression, since distillation cannot add capacity a linear model does not
have.

---

## 3. Task 2: PCA + KNN

PCA (Pearson, 1901; Jolliffe, 2002) for dimensionality reduction of the 5,000
features, then k-nearest neighbours (Cover & Hart, 1967) with k=2. All four variants
were submitted to Kaggle:

| PCA components | Kaggle public F1 |
|---|---|
| 100 | **0.67793** |
| 500 | 0.66373 |
| 1000 | 0.55923 |
| 2000 | 0.41495 |

**Finding:** monotone degradation with dimensionality, a textbook curse-of-
dimensionality effect: in high dimensions, distances concentrate and nearest-neighbour
votes approach noise (Beyer et al., 1999). Fewer components mean denoised, more
discriminative distances. Val-set ordering matched Kaggle ordering exactly here,
which, in hindsight, is because PCA+KNN never sees topic vocabulary. This foreshadows
the shift-robustness of dense-feature models observed later.

---

## 4. Task 3: Open Model Search (the main event)

### 4.1 Baseline sweep and the winning classical model

We benchmarked eight classical models over both the provided features and our own
text-derived representations. The winner, and still our standing best submission:

> **Word 1-2 gram + char_wb 3-5 gram TF-IDF (sublinear TF), concatenated, into
> LinearSVC, C=0.25, class_weight='balanced'** (soft-margin linear SVM: Cortes &
> Vapnik, 1995; LIBLINEAR: Fan et al., 2008; SVMs for text: Joachims, 1998).
> Validation 0.8229, **Kaggle public 0.72990**.

Character n-grams within word boundaries (`char_wb`) are the single most valuable
representation choice in the whole project: they capture sub-word style (morphology,
punctuation habits, tokenization artifacts of generators) rather than topic words.

### 4.2 The shift discovery: the project's central lesson

The drop from 0.8229 in validation to 0.72990 on the leaderboard (−0.09) is not
variance; it reproduces under vanilla stratified 5-fold CV (0.8189). Diagnosis:

- Adversarial validation (train-vs-test classifier) on char n-grams: **AUC 0.8119**
- Test abstracts ~25% longer than train
- Feature-based models (PCA+KNN) scored *above* their validation on the LB, while
  text models scored far below. The shift lives in vocabulary/topics and in length.

We therefore built a **cluster-holdout proxy**: KMeans (MacQueen, 1967) with 10
clusters on word-unigram TF-IDF, clusters merged into 5 balanced groups, each fold
holding out entire topic groups. The baseline scores **0.7383** under this proxy,
within 0.008 of its real 0.7299, validating it as a leaderboard estimator for
single text models.

### 4.3 Shift-aware ensemble (ENSMB): FAIL, submitted and measured

A soft-vote of a calibrated char-only logistic regression (probability calibration:
Platt, 1999) and a LightGBM over 10 stylometric features that we had pruned by
adversarial AUC (18 down to 10, after discovering the raw set was nearly as
shift-prone as raw text, AUC 0.79, driven by length features):

- Cluster-holdout proxy: 0.7884 (vs baseline's 0.7383), which looked like a clear win
- **Real leaderboard: 0.7135, below the baseline.** Deflation −0.075.

**PASS/FAIL verdict: FAIL, and the most valuable failure in the project** (see §4.8).

### 4.4 Fully from-scratch track (stretch goal: ≥0.80 val without any ML library)

Three complete implementations, no sklearn/xgboost/lightgbm in the model path:

| Implementation | Validation | Kaggle public |
|---|---|---|
| Gaussian + Multinomial Naive Bayes hybrid, log-odds pooling | 0.6485 (val) | n/a |
| Hand-rolled TF-IDF (word+char n-grams) + hand-rolled logistic regression | 0.8236 (val) | 0.67283 |
| Hand-rolled gradient-boosted trees, Newton leaf values (Friedman, 2001) on 218 dense features | 0.8320 (5-fold CV) | 0.71464 |

Both headline targets passed validation; both deflated hard on the leaderboard. The
from-scratch TF-IDF model's collapse (0.824 to 0.673) traced to min_df=1: a 640K-term
vocabulary memorizing train-specific words that don't transfer under topic shift. The
library baseline's min_df=2 pruning was doing more shift-protection than we had
credited.

### 4.5 Test-time adaptation ("working backwards from the test set")

Since the failure mode is train-to-test shift, we attacked the training distribution
directly, evaluating on the cluster-holdout proxy (5 folds, baseline reproduced at
0.7383 exactly before trusting the harness):

| Technique | Proxy F1 | Δ vs baseline | Notes |
|---|---|---|---|
| Transductive vocabulary (TF-IDF fit on train+test text; classifier on train only) | 0.7444 | +0.006 | free, safe |
| Pseudo-labeling / self-training (Lee, 2013; Scudder, 1965): margin ≥ 1.0 | 0.7444 | +0.006 | pseudo-labels **98.1% accurate** at 8% coverage, measured against true fold labels |
| **Importance weighting (Shimodaira, 2000; Sugiyama et al., 2007): train rows weighted by out-of-fold adversarial P(test-like)/(1−P), clipped [0.25,4]** | **0.7523** | **+0.014, positive on all 5 folds** | the only lever that wins every fold |
| Importance weighting + pseudo-labeling stacked | 0.7524 | +0.0001 over IW alone | the two levers exploit the same signal and do not compound |

**PASS (proxy-level):** importance weighting is a real, fold-consistent gain.
Projected real LB ≈ 0.744 under the single-model calibration rule (§4.8). Held in
reserve, not yet submitted.

### 4.6 Diverse-model screen, prediction-correlation matrix, and stacking: FAIL, submitted and measured

Following Wolpert's stacked generalization (1992): six diverse base models, each
producing out-of-fold (OOF) scores under stratified 5-fold CV; a Pearson correlation
matrix of those OOF scores used to select a genuinely diverse subset; a meta-learner
fit on OOF scores only (never on self-predictions).

Correlation matrix (abridged): the proven LinearSVC and a RidgeClassifier
(Hoerl & Kennard, 1970) on the same representation correlate at **0.985**, making
them near duplicates, so only ridge was kept. Selected subset: LightGBM (Ke et al.,
2017) on TruncatedSVD-200 (Halko et al., 2011) + robust stylometrics; Multinomial
Naive Bayes on word TF-IDF (r ≈ 0.63-0.75 to everything, the diversity workhorse);
RidgeClassifier on word+char TF-IDF; logistic regression on PCA-120 of the provided
features.

Leakage control: every vectorizer/SVD/PCA/scaler fit only on fold-train indices;
meta-learner evaluated by nested inner-OOF within each cluster-holdout fold.

| Quantity | Value |
|---|---|
| Best single base (LGBM svd+stylo) on cluster-holdout | 0.7717 |
| Stack, logistic meta | **0.8018** (ridge meta: 0.7996, so the two meta-learners agree) |
| Stack train/val gap | +0.055 (vs the failed ENSMB's +0.130), passed our screen |
| **Real leaderboard** | **0.72407, below the 0.72990 incumbent. FAIL.** |

Deflation −0.078, almost exactly the ENSMB's −0.075.

### 4.7 Maxed-out gradient boosting: built, deliberately NOT submitted

~30 configs of XGBoost (Chen & Guestrin, 2016), LightGBM, and Random Forest
(Breiman, 2001) across five representations. Winner: LightGBM, 1200 trees, lr 0.03,
63 leaves, subsample/colsample 0.8, on TruncatedSVD-300 of word+char TF-IDF +
10 robust stylometric features:

- Vanilla CV 0.8586, cluster-holdout **0.8079**, train F1 1.0000 (gap +0.192)
- Consistent secondary findings: adding stylometrics to any SVD representation is
  worth +0.06-0.07; text-derived SVD dominates the provided 5,000 features (best
  0.808 vs 0.686); XGBoost tracked LightGBM ~0.006 behind everywhere; RF clearly worst.

After the stack's leaderboard result landed (§4.6), the calibration rule below
projects this model to ~0.73 real, which is not worth a submission slot. **Withheld.**

### 4.8 The calibration rule: what two paid-for failures bought us

| Model family | Proxy to real deflation | Evidence |
|---|---|---|
| Single sparse-text model | −0.008 | LinearSVC: 0.7383 to 0.7299 |
| Anything containing stylometric/dense-tree legs, or ensembles thereof | **−0.075 to −0.078** | ENSMB: 0.7884 to 0.7135; stack: 0.8018 to 0.7241 |

Mechanism: the cluster-holdout proxy simulates *topic* shift (held-out vocabulary)
but cannot simulate *stylometric* shift. The real test set's abstracts are ~25%
longer, and length-correlated features are precisely what stylometric and tree legs
exploit. The proxy therefore systematically overrates exactly that family. Two
independent submissions deflating by the same amount turn this from a suspicion into
a usable correction factor.

### 4.9 Transformer fine-tuning (in progress at time of writing)

Fine-tuned `distilbert-base-uncased` (DistilBERT: Sanh et al., 2019; BERT: Devlin
et al., 2019; Transformer: Vaswani et al., 2017) with a binary head: max_length 256,
batch 16, lr 2e-5, linear warmup 6%, weight decay 0.01, 2 epochs, on Apple MPS.

**Epoch-level results (stratified 90/10 holdout):**

| Epoch | Val macro F1 |
|---|---|
| 1 | **0.8498** (selected) |
| 2 | 0.8393 |

Train F1 0.9067, giving a gap of +0.057, an order smaller than the classical models'
~0.25 memorization gap on this data.

**Cluster-holdout folds (identical fold definitions as every experiment above):**

| Fold | DistilBERT | LinearSVC baseline (same fold) | Δ |
|---|---|---|---|
| 0 (largest, hardest) | 0.7580 | 0.7331 | +0.025 |
| 1 | 0.8740 | 0.7911 | +0.083 |
| Mean (2 folds) | 0.8160 | 0.7621 | +0.054 |

Known weakness: ~66% of test abstracts exceed 256 tokens and get truncated (test
median 345 tokens vs train 245, the length shift again). A max_length=448 rerun is
training as this report is written. The transformer is a different model family from
both failure cases, its gap is small, and it beats the baseline on every fold tested:
under our calibration it projects to ≈ 0.75+, our best remaining bet against 0.78326.

---

## 5. Complete Kaggle Submission Record

| # | Submission | Public F1 | Verdict |
|---|---|---|---|
| 1-4 | PCA+KNN (2000/1000/500/100) | 0.41495 / 0.55923 / 0.66373 / 0.67793 | Task 2 curve; PASS |
| 5 | **LinearSVC word+char TF-IDF (baseline)** | **0.72990** | **standing best; PASS** |
| 6 | ENSMB shift-aware soft-vote | 0.71351 | FAIL (proxy overrated) |
| 7 | Task 1 from-scratch logistic regression | 0.68578 | Task 1 record; PASS vs sklearn |
| 8 | From-scratch TF-IDF + logistic regression | 0.67283 | FAIL (vocab overfit) |
| 9 | From-scratch gradient-boosted trees | 0.71464 | below baseline |
| 10 | Stack (4 diverse bases, logistic meta) | 0.72407 | FAIL (proxy overrated ensembles again) |

Kaggle scores each submission on a hidden public subset of the 6,999 test rows and
keeps the team's best, so probes 6-10 cost nothing but information. Submissions #6
and #10 bought the calibration rule that now steers everything.

---

## 6. What We Learned (ranked by how much it cost)

1. **Validation is a model of deployment, and models can be wrong in structured
   ways.** Our proxy was accurate for single text models and systematically ~0.075
   optimistic for stylometric/ensemble families. We only learned that by submitting
   and being wrong twice, but we learned it as a *number*, and now use it as a
   correction factor.
2. **Distribution shift dominates model choice.** Every point of leaderboard
   difference among our classical models is smaller than the val-to-LB deflation. The
   representation (char_wb n-grams, min_df pruning) and the training distribution
   (importance weighting) mattered more than the classifier.
3. **Capacity ceilings are real and measurable.** Two refinement rounds, eleven
   levers, five seeds each: the from-scratch logistic regression does not move.
   Knowing when a model is *done* is as important as improving it.
4. **Pseudo-labels can be trusted when you can measure their precision.** 98% label
   accuracy at 8% coverage. The gain was already captured by importance weighting,
   though; the two levers share one underlying signal.
5. **Prediction-correlation matrices are the right way to pick ensemble members.**
   Models correlated at 0.985 are duplicates, not diversity. This holds even though
   the ensemble family itself proved shift-fragile here.

---

## References

- Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? *ICDT*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1).
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3).
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Trans. Information Theory*, 13(1).
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *JMLR*, 9.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5).
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with randomness. *SIAM Review*, 53(2).
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1).
- Joachims, T. (1998). Text categorization with support vector machines. *ECML*.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML Workshop*.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Berkeley Symposium*.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11).
- Platt, J. (1999). Probabilistic outputs for support vector machines. *Advances in Large Margin Classifiers*.
- Robbins, H., & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3).
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5).
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop*.
- Scudder, H. (1965). Probability of error of some adaptive pattern-recognition machines. *IEEE Trans. Information Theory*, 11(3).
- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *J. Statistical Planning and Inference*, 90(2).
- Sugiyama, M., Krauledat, M., & Müller, K.-R. (2007). Covariate shift adaptation by importance weighted cross validation. *JMLR*, 8.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2).
