# 50.007 Machine Learning Project Report — GenAI Content Detection

> **Draft** — personalise the voice, add your team name / member names, fill the
> `TODO` slots with final Kaggle scores, then export to PDF for submission.

**Team:** TODO
**Kaggle team name:** TODO

## 1. Our best performing model and how it works

Our best model is a **Linear Support Vector Machine trained on a custom TF-IDF
representation of the raw text** (validation macro F1 **0.8229**, Kaggle public
leaderboard **0.72990** — 1st place at the time of submission, ahead of both the
Blue Line (0.69361) and red line (0.59044) baselines).

The competition provides 5000 pre-computed TF-IDF features, but it also provides the
raw text. Our key insight was that the fixed 5000-word vocabulary is a bottleneck: it
caps how much stylistic information any downstream model can see. We therefore built
our own richer representation:

- **Word 1–2-grams** (TF-IDF, sublinear TF scaling, min_df=2): captures topical word
  choice and common two-word patterns.
- **Character 3–5-grams** (char_wb analyzer, capped at 300K features): captures
  sub-word style — punctuation habits, suffixes, spacing, tokenisation quirks —
  signals known to be strong for authorship attribution, which is essentially what
  human-vs-machine detection is.

Together this yields ~640K sparse features. A linear SVM is a natural fit: it finds
the maximum-margin hyperplane separating the classes, handles very high-dimensional
sparse input efficiently, and hinge loss focuses training on the hard boundary cases.
With `class_weight="balanced"` the per-class penalties are re-weighted inversely to
class frequency, which matters because the data is imbalanced (62.5% machine / 37.5%
human) and the metric is *macro* F1 — both classes count equally.

## 2. How we got there — our roadmap and tuning

Our journey, in order:

1. **From-scratch logistic regression (Task 1)** on the 5000 provided features gave
   validation macro F1 0.7318 — our first baseline.
2. **PCA + KNN (Task 2)** showed the provided feature space is noisy for
   distance-based methods: performance *improved* as we reduced components
   (Kaggle: 2000 → 0.415, 1000 → 0.559, 500 → 0.664, 100 → 0.678).
3. **Model race on the provided features.** We tuned each with `f1_macro` scoring
   (3-fold CV grids for the fast linear models, manual settings for the tree
   ensembles), comparing on a stratified 10% validation split:

   | model | key hyperparameters tried | best setting | val macro F1 |
   |---|---|---|---|
   | LinearSVC | C ∈ {0.01…5}, class_weight | C=0.5, balanced | 0.7364 |
   | LogisticRegression | C ∈ {0.1…50}, class_weight | C=5.0, balanced | 0.7359 |
   | ComplementNB | alpha ∈ {0.01…1} | alpha=0.01 | 0.6486 |
   | RandomForest | n_estimators=500, max_features=sqrt | — | 0.6986 |
   | XGBoost | 800 trees, lr=0.1, depth=6, subsample=0.8 | — | 0.7399 |
   | LightGBM | 800 trees, lr=0.1, 63 leaves, subsample=0.8 | — | 0.7513 |
   | Soft-voting ensemble | calibrated SVC + LR + LightGBM | — | 0.7578 |

   Even the best ensemble plateaued around 0.76 — models on the same 5000 features
   make correlated errors, so ensembling bought little.
4. **Breaking the plateau with the raw text.** Re-representing the text ourselves
   (word + char n-gram TF-IDF → LinearSVC) jumped validation macro F1 to 0.8207 —
   a +6 point leap over anything on the provided features. This told us the
   representation, not the classifier, was the bottleneck.
5. **Tuning the winner.** We swept C ∈ {0.25, 0.5, 1.0, 2.0} × class_weight ∈
   {None, balanced}: best was C=0.25 with balanced weights (0.8229). We then tried to
   beat it and failed — LogisticRegression on the same features (0.8207), stacking
   the provided 5000 features onto ours (0.8160 — slightly *worse*), and a
   calibrated-SVC + LR soft vote (0.8189). The final model was refit on all 20K
   training rows before predicting the test set.

## 3. Difficulties we faced

- **Our from-scratch logistic regression silently underfit.** Our first
  configuration (lr=1.0, 200 epochs) looked "done" — the loss curve flattened — but
  validation macro F1 (0.690) lagged sklearn's (0.706). The training loss (0.487) was
  the clue: TF-IDF values are ≤1 and mostly zero, so gradients are tiny and
  convergence at ordinary learning rates is very slow. Raising the learning rate to
  10.0 (which sounds reckless but is safe at this feature scale) and training 500
  epochs dropped the loss to 0.306 and lifted validation macro F1 to 0.7318 — ahead
  of the sklearn benchmark. Lesson: a flat loss curve shows convergence *at that
  learning rate*, not convergence of the model.
- **No labelled dev set.** The brief mentions a 2K dev set, but the download contains
  only train and test. We carved a stratified 10% holdout from train for all model
  comparisons and refit on the full 20K rows before submitting. The Task 2 results
  later validated this proxy: the holdout ranking matched the Kaggle ranking exactly.
- **PCA+KNN behaved "backwards".** More components (more retained variance) gave
  *worse* scores, which initially looked like a bug. It is the curse of
  dimensionality: with n_neighbors=2, distances in 2000-d space concentrate and the
  nearest neighbours stop being meaningful. Verifying the trend on both our holdout
  and Kaggle convinced us it was real, and it became a useful insight rather than a
  bug.
- **Ensembles refused to help.** Coming out of the course we expected voting/stacking
  to be the endgame, but our models on the provided features erred on the *same*
  examples. The fix was not a better combiner but a better representation (point 4
  above).
- **The validation–leaderboard gap.** Our best model scored 0.823 on our holdout but
  0.730 on the public leaderboard, while the Task 2 PCA models scored *above* their
  holdout estimates. Since our holdout is drawn from train, this points to
  **distribution shift**: the test texts likely come from partly different generators
  or domains, so a model with a rich vocabulary fitted on train text picks up
  train-specific cues that do not all transfer. It still won — stylistic character
  n-grams transfer better than topical words — but it taught us that a random holdout
  from train is an optimistic proxy when test data is drawn from a shifted
  distribution, and that leaderboard probing of multiple model families is essential.

## 4. What we self-learned beyond the course

- **Character n-gram TF-IDF for style detection.** The course covered TF-IDF at the
  word level; we learned from the authorship-attribution literature that character
  n-grams (with word-boundary awareness, `char_wb`) capture punctuation and sub-word
  style that machine text reproduces differently from humans — this was the single
  biggest scoring lever in the whole project.
- **`sublinear_tf` scaling** (1 + log tf): dampens raw term counts so repeated words
  don't dominate — a small flag with a measurable effect.
- **Calibrating margin classifiers.** SVMs output distances, not probabilities; we
  learned to wrap `LinearSVC` in `CalibratedClassifierCV` (Platt scaling / isotonic)
  to get probabilities usable in soft-voting ensembles.
- **Randomized SVD for PCA at scale.** Exact PCA on a 20000×5000 matrix is slow;
  sklearn's `svd_solver="randomized"` (Halko et al.) approximates the top components
  in a fraction of the time, which made the 2000-component experiments feasible.
- **Numerical stability tricks** for the from-scratch model: clipping the sigmoid
  input to avoid `exp` overflow, and clipping predicted probabilities away from 0/1
  to avoid `log(0)` in the loss.

---

### Appendix — final Kaggle scores

| submission | public macro F1 | private macro F1 |
|---|---|---|
| Task 1: LogReg_Prediction.csv | 0.68578 | (after week 13) |
| Task 2: PCA_2000 | 0.41495 | |
| Task 2: PCA_1000 | 0.55923 | |
| Task 2: PCA_500 | 0.66373 | |
| Task 2: PCA_100 | 0.67793 | |
| Task 3: Task3_Final_Prediction.csv | **0.72990** (best submission; beats Blue Line 0.69361 and red line 0.59044) | |

Additional Task 3 candidates submitted 2026-07-14 (none beat the 0.72990 baseline on the
public leaderboard, but all beat the blue line except the from-scratch TF-IDF):

| submission | public macro F1 | note |
|---|---|---|
| Task3_Improved_Prediction.csv (ENSMB: char-LR + LightGBM stylometric soft-vote) | 0.71351 | shift-aware validation predicted ~0.79 but did not translate — a lesson in proxy limits |
| GradientBoosting_Scratch_Prediction.csv (from-scratch GBT) | 0.71464 | fully from scratch, no libraries |
| TFIDF_Scratch_Prediction.csv (from-scratch TF-IDF + LogReg) | 0.67283 | min_df=1 640K-feature vocab overfit train-specific words |
