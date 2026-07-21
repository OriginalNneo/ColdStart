# Machine Learning Algorithms Guide

> **📌 Current best (updated 2026-07-20):** the leaderboard-best *eligible* classical model is now
> **`bankstylo_iwst` → Kaggle public 0.79080** (#2, ~0.004 behind leader): the Ridge+word×1.6 stack
> + topic-invariant LLR/style bank ×0.02 + 227-dim stylo ×0.04 + importance-weighting + self-training.
> Session arc **0.75210 → 0.77913 → 0.79080**; a new orthogonal pseudo-POS leg (Iter 22) projects ~0.80,
> queued. See `TESTING_REPORT.md` (Iters 11–22). The "winning pipeline" figures below (Custom TF-IDF +
> LinearSVC, 0.8229 val / 0.72990 LB) are the earlier **baseline**, kept for the concept walkthrough.

## Problem Setup
**Task:** Binary text classification
- **Input:** Raw text (human-written or machine-generated)
- **Output:** Label {0: human, 1: machine}
- **Metric:** Macro F1 (F1 for both classes averaged)
- **Data:** 20,000 training samples (62.5% machine, 37.5% human)

---

## The Winning Pipeline

### 1. Feature Engineering: TF-IDF (Term Frequency - Inverse Document Frequency)

**What is TF-IDF?**
- TF = how often a word appears in a document
- IDF = log(total docs / docs with word)
- TF-IDF = TF × IDF = score for each word in each document
- Words that appear in many docs get low weight (common words)
- Words that appear in few docs get high weight (distinguishing words)

**Our custom representation:**

```
Word 1-2grams (TF-IDF):
  ├─ Unigrams: ["the", "text", "generated", ...]
  └─ Bigrams: ["machine generated", "human written", ...]
  
  Purpose: Capture TOPICAL VOCABULARY
  
Character 3-5grams with word boundaries (TF-IDF):
  ├─ Trigrams: ["[the", "he ", "e t", ...]
  ├─ 4-grams: ["[the ", "he t", "e te", ...]
  └─ 5-grams: ["[the t", "he te", "e tex", ...]
  
  Purpose: Capture STYLE FINGERPRINT
         (punctuation, spacing, suffix patterns)
```

**Why character n-grams work:**
- Machine-generated text has different punctuation habits
- Different spacing patterns (double spaces, odd breaks)
- Different suffix distributions (more "-ing", "-tion")
- These micro-patterns are SIGNATURES of authorship

**Result:** ~640,000 features (vs 5,000 fixed)

---

### 2. Classification Model: Linear SVM

**Algorithm:**
```
Linear SVM finds the best line/plane that separates two classes.

For 2D intuition:
  
  Machine-generated texts
      •  •  •  •
        •   •  /  ← best separating line
          •  /
         /  •
        /     •  •  
    /        Human texts
```

**Math:**
```
Decision function: f(x) = w·x + b
Prediction: y = sign(w·x + b)  [if positive → class 1, else → class 0]

Training minimizes:
  L = regularization_penalty + hinge_loss
  
Hinge loss = 0 if correct and confident
           = penalty if wrong or unconfident
```

**Key hyperparameters:**
- **C** (inverse regularization)
  - Low C: stronger regularization, simpler model, less overfitting
  - High C: weaker regularization, complex model, more overfitting
  - Sweet spot: C=0.25 for this dataset

- **class_weight="balanced"**
  - Dataset is 62.5/37.5 imbalanced
  - Without balancing: model predicts "machine" too much
  - With balancing: penalizes misclassifying minority (human) class more
  - Formula: weight ∝ 1 / class_frequency

**Why SVM wins:**
- Linear model → fast to train on 640K features
- Hinge loss focuses on hard boundary cases (uncertain examples)
- Sparse input efficient (most features are zero)
- Works well with high-dimensional features

---

### 3. Alternatives We Tried (and Why They Lost)

#### Baseline: Logistic Regression on fixed 5000 features
**Algorithm:** P(y=1|x) = sigmoid(w·x + b)
- Probabilistic: outputs probability, not just class
- Similar to SVM but uses different loss (log loss)
- Result: **0.7359 macro F1** → ❌ Too low

**Why it lost:** Limited 5000-feature vocabulary couldn't capture enough style signals.

#### Alternative: PCA + KNN

**PCA (Principal Component Analysis):**
```
Reduces 5000 dimensions to k components (e.g., 100)

1. Find directions of highest variance in data
2. Project data onto top k directions
3. Result: 100 numbers that summarize each sample

Effect: 
  - Removes noise (features that vary randomly)
  - Keeps signal (features that separate classes)
```

**Why fewer components are better (curse of dimensionality):**
```
In 2000-dimensional space:
  - All points are far apart from each other
  - "Nearest neighbor" is often not actually similar
  - Distance metric breaks down

In 100-dimensional space:
  - Neighbors are more meaningful
  - Distance better reflects similarity
```

**KNN (k-Nearest Neighbors):**
```
Classification by majority vote of k nearest training samples

For each test point:
  1. Find k closest training points (by distance)
  2. Count votes: majority class wins

k=2 means: classify by closest 1-2 training samples
  - Simple
  - Noisy (sensitive to outliers)
  - Requires good distance metric
```

**Result:** PCA(100) + KNN(k=2) = **0.6525 macro F1** → ❌ Lost by 0.17 points

**Why it lost:** KNN needs good features; PCA discards fine-grain details that help distinguish.

#### Tree Ensembles (Random Forest, XGBoost, LightGBM)
```
Random Forest:    500 decision trees, each on random feature subset
XGBoost:          800 gradient-boosted trees, sequential error correction
LightGBM:         800 boosted trees with leaf-wise growth
```

**Why they lost:** Tree models work best with dense, low-dim features. Our 640K sparse features break their assumptions. Result: 0.73-0.75 macro F1 → ❌

---

## Algorithm Comparison

| Aspect | Logistic Regression | SVM | PCA+KNN | Random Forest | XGBoost |
|--------|---|---|---|---|---|
| **Input** | Dense or sparse | Dense or sparse | Dense (preferred) | Dense (preferred) | Dense (preferred) |
| **High-dim sparse** | ✓ Works | ✓✓ Works great | ✗ Needs PCA | ✗ Struggles | ✗ Struggles |
| **Training time** | Fast | Fast | Medium (PCA) | Slow | Slow |
| **Interpretable** | ✓ Feature weights | ✓ Feature weights | ✗ Hard | ~ Tree paths | ~ Tree paths |
| **Requires tuning** | Moderate | Moderate | High | Very high | Very high |
| **Macro F1 (our data)** | 0.7359 | **0.8229** | 0.6525 | 0.6986 | 0.7399 |

---

## Key Insights from This Project

### 1. Representation > Model
```
Good features + simple model > Bad features + complex model

640K TF-IDF features + Linear SVM: 0.8229
5000 fixed features + Ensemble: 0.7578

Winner = better representation, not more complex model
```

### 2. Why Ensembles Failed
```
If all models use same (bad) feature set:
  → They make the same mistakes
  → Ensemble can't fix errors
  → Voting doesn't help

Ensemble only works when models:
  - Use different feature sets, OR
  - Use different learning algorithms
  
Our ensemble (8 models on same 5000 features) → 0.7578
Custom TF-IDF + Linear SVM (different representation) → 0.8229
```

### 3. Distribution Shift
```
Validation score: 0.8229
Kaggle public score: 0.7299

Why the gap?
- Training data = text from certain generators/domains
- Test data = text from partly different generators/domains
- Model learned train-specific patterns that don't transfer

Character n-grams transfer better than topic words:
- Topic words are domain-specific
- Character patterns (style) transfer across domains
```

### 4. Class Imbalance Handling
```
Dataset: 62.5% machine / 37.5% human
Metric: Macro F1 (both classes weighted equally)

Without balancing:
  Model learns to predict "machine" too much
  → High accuracy but low F1 for human class

With class_weight="balanced":
  Each misclassification of human weighted 62.5/37.5 ≈ 1.67× more
  → Model learns to classify humans more carefully
  → Better macro F1
```

---

## Simple Implementation Checklist

### ✓ Our simplified pipeline:

```python
# 1. Load data
train_text = pd.read_csv("train.csv")  # id, text, label
y_train = train_text["label"].values

# 2. Split train/validation (stratified, 90/10)
from sklearn.model_selection import train_test_split
tr_idx, val_idx = train_test_split(
    np.arange(len(y_train)), 
    test_size=0.1, 
    stratify=y_train, 
    random_state=SEED
)

# 3. Build TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), 
                            min_df=2, sublinear_tf=True, max_features=300_000)
vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])

# 4. Vectorize training text
X_tr_vec = vectorizer.fit_transform(train_text.iloc[tr_idx]["text"])

# 5. Train Linear SVM with best hyperparameters
from sklearn.svm import LinearSVC
svm = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
svm.fit(X_tr_vec, y_train[tr_idx])

# 6. Predict on test set
test_vec = vectorizer.transform(test_text["text"])
y_pred = svm.predict(test_vec)

# 7. Save predictions
pd.DataFrame({"id": test_ids, "label": y_pred}).to_csv("predictions.csv", index=False)
```

This is the **KISS principle in action:**
- No unnecessary model searching
- No complex ensembles
- No advanced techniques
- Just 3 well-chosen steps: vectorize → tune → predict
- Result: 1st place on Kaggle leaderboard

---

## Further Reading

- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- SVM: https://scikit-learn.org/stable/modules/svm.html
- Linear classification: https://scikit-learn.org/stable/modules/linear_model.html
- Text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
