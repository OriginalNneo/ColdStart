# Quick Start Guide

> **📌 Current best (updated 2026-07-20):** the leaderboard-best *eligible* classical model is now
> **`bankstylo_iwst` → Kaggle public 0.79080** (#2, ~0.004 behind the leader). It is the Ridge+word×1.6
> stack + a topic-invariant LLR/style bank ×0.02 + a 227-dim stylo block ×0.04 + covariate-shift
> importance-weighting + one round of class-balanced self-training
> (`predictions/Task3_BankStyloIWSelfTrain_Prediction.csv`). Session arc **0.75210 → 0.77913 → 0.79080**;
> a new orthogonal pseudo-POS syntactic leg (Iter 22) projects ~0.80 and is queued for the next quota
> reset. See `TESTING_REPORT.md` (Iters 11–22) for the full campaign. The "winner" figures below
> (Custom TF-IDF + LinearSVC, 0.8229 val / 0.72990 LB) are the earlier **baseline** this
> guide was written around, kept for the learning walkthrough.

## 🎯 5-Minute Overview

Your GenAI text detection project has 3 versions:

| Version | When to use | Time |
|---------|-----------|------|
| **ALGORITHMS_GUIDE.md** | Want to understand concepts | 10 min |
| **ML_Project_Clean.ipynb** | Want to learn interactively | 20 min |
| **ML_Project_Simple.py** | Want to run it now | 1 min |

---

## 🚀 Run Right Now

```bash
# Make sure data is in place
ls -la data/train.csv data/test.csv data/train_features.csv

# Run the simplified pipeline
python ML_Project_Simple.py

# Check predictions
ls -la predictions/
```

**Expected output:**
```
✓ Loaded: train (20000, 5000), test (6999, 5000)
✓ Class balance: [ 7496 12504] (62.5% machine, 37.5% human)

📊 BASELINE (sklearn LogReg on 5000 features):
   Validation Macro F1: 0.7359

🏆 WINNER: Custom TF-IDF + Linear SVM
   Best config: {'C': 0.25, 'class_weight': 'balanced'} → Validation F1 = 0.8229
   ✓ Saved predictions: predictions/Winner_Prediction.csv

🔄 ALTERNATIVE: PCA(100) + KNN(k=2)
   Validation Macro F1: 0.6525
   ✓ Saved predictions: predictions/Alternative_PCA_KNN_Prediction.csv

SUMMARY
Winner (Custom TF-IDF + SVM):    0.8229
Alternative (PCA + KNN):        0.6525
Baseline (LogReg on 5000):      0.7359
```

✓ Done! Submit `predictions/Winner_Prediction.csv` to Kaggle.

---

## 📚 Learn the Algorithms (15 min)

### Step 1: Read ALGORITHMS_GUIDE.md
```
- What is TF-IDF?
- Why Linear SVM wins
- Why other models lost
- Key insights
```

### Step 2: Understand the flow

```
Raw Text (20K samples)
    ↓
[1] TF-IDF Vectorization (word 1-2grams + char 3-5grams)
    → 640K features
    ↓
[2] Linear SVM Classification (C=0.25, class_weight=balanced)
    → maximize-margin hyperplane
    ↓
[3] Predict on Test Set (6,999 samples)
    → Label as human or machine
    ↓
Submission CSV
```

### Step 3: Run the notebook interactively

```bash
jupyter notebook ML_Project_Clean.ipynb
```

Click through each cell and understand:
- Why we split train/validation
- How TF-IDF represents text
- Why character n-grams matter
- How SVM tuning works

---

## 🔍 Understand What Worked

### The Winning Formula

```python
# 1. Custom feature engineering
vectorizer = TfidfVectorizer(
    word_ngrams=(1, 2),           # topical words
    char_ngrams=(3, 5),           # style patterns
    max_features=640_000
)

# 2. Simple linear model
svm = LinearSVC(
    C=0.25,                       # regularization
    class_weight="balanced"       # handle imbalance
)

# 3. Result: 0.8229 macro F1 → 1st place
```

### Why It Works

| Component | Purpose | Impact |
|-----------|---------|--------|
| Word 1-2grams | Topical vocabulary | Captures "generated", "text", etc. |
| Char 3-5grams | **Style fingerprint** | **Captures punctuation/spacing habits** |
| LinearSVC | Margin maximization | Efficient on sparse high-dim data |
| class_weight="balanced" | Handle imbalance | Don't overpredict machine class |
| C=0.25 | Regularization | Avoid overfitting |

**Key insight:** Character n-grams = authorship fingerprint. Machine text has different punctuation, spacing, suffix patterns. This is what beats all other approaches.

---

## 📊 Key Results

### Validation Performance
```
🏆 Custom TF-IDF + SVM:  0.8229 macro F1
   Baseline (LogReg):    0.7359 macro F1 (-0.0870)
   Alternative (PCA):    0.6525 macro F1 (-0.1704)
```

### Kaggle Performance
```
🏆 Custom TF-IDF + SVM:  0.72990 (1st place)
   Blue Line baseline:   0.69361 (+3.6 points)
   Red Line baseline:    0.59044 (+13.9 points)
```

### Why the gap between validation (0.8229) and Kaggle (0.7299)?
**Distribution shift:** Test data comes from different generators/domains than training data.
- Topical words = domain-specific, don't transfer well (-9.3%)
- Character patterns = universal style signatures, transfer better
- That's why character n-grams save us

---

## 🛠️ Modify & Experiment

### Try different hyperparameters

In `ML_Project_Simple.py`, line 140:

```python
for C in [0.25, 0.5, 1.0, 2.0]:  # ← Change these values
    for class_weight in [None, "balanced"]:  # ← Try new options
        ...
```

**What to try:**
- `C in [0.1, 0.5, 2.0]` — regularization strength
- `class_weight in [None, "balanced"]` — imbalance handling

### Try different n-grams

In line 123:

```python
word_vec = TfidfVectorizer(
    ngram_range=(1, 3),  # ← Try (1,2), (1,3), (2,2)
    ...
)

char_vec = TfidfVectorizer(
    ngram_range=(3, 6),  # ← Try (2,4), (3,5), (4,6)
    ...
)
```

### Add new models

Copy the pattern from Alternative (PCA + KNN):

```python
def my_model(X_text, y, tr_idx, val_idx, X_text_test, test_ids):
    """Your new model here."""
    X_tr, y_tr = X_text[tr_idx], y[tr_idx]
    X_val, y_val = X_text[val_idx], y[val_idx]
    
    # Your training code
    model = ...
    model.fit(X_tr, y_tr)
    
    # Validation
    f1 = f1_score(y_val, model.predict(X_val), average="macro")
    print(f"Your model: {f1:.4f}")
    
    # Predict test
    y_pred = model.predict(X_text_test)
    save_predictions(test_ids, y_pred, f"predictions/your_model.csv")
    
    return f1, y_pred
```

---

## ❓ FAQ

**Q: Why not use deep learning?**  
A: Not needed for this dataset. Simple linear model beats complex approaches when features are good.

**Q: Can I use the original complex notebook?**  
A: Yes! `ML_Project.ipynb` has all 8 models if you want to see detailed comparisons. But it's not needed for winning.

**Q: What if I want to understand the from-scratch logistic regression?**  
A: That's in `ML_Project.ipynb` (Task 1). It's educational but not required for best results.

**Q: How do I submit to Kaggle?**  
A: Upload `predictions/Winner_Prediction.csv` on the Kaggle competition page.

**Q: Why 640K features if models typically prefer <10K?**  
A: SVM is different—linear models handle sparse high-dim input well. Other models (tree-based) would struggle.

**Q: Should I ensemble the predictions?**  
A: Only if you have DIFFERENT models using DIFFERENT features. Ensembling similar models on same features doesn't help.

---

## 📋 Checklist Before Submitting

- [ ] Read ALGORITHMS_GUIDE.md (understand the approach)
- [ ] Run ML_Project_Simple.py (verify it works)
- [ ] Check predictions/Winner_Prediction.csv (has id and label columns)
- [ ] Verify row count = 6,999 (matches test set)
- [ ] Submit to Kaggle competition
- [ ] Check leaderboard (should be competitive, ~0.72-0.73 range)

---

## 🎓 Learning Path

### Beginner
1. Run `python ML_Project_Simple.py`
2. Read ALGORITHMS_GUIDE.md
3. Understand why each piece matters

### Intermediate
1. Run `jupyter notebook ML_Project_Clean.ipynb`
2. Modify hyperparameters in the notebook
3. Experiment with different n-grams
4. Track validation F1 changes

### Advanced
1. Study `ML_Project.ipynb` (original complex version)
2. Understand why each model was tested
3. Explore ensemble strategies
4. Analyze prediction errors

---

## 📞 Next Steps

1. **Run it:** `python ML_Project_Simple.py`
2. **Learn it:** Read ALGORITHMS_GUIDE.md
3. **Master it:** Run ML_Project_Clean.ipynb step-by-step
4. **Submit:** Upload Winner_Prediction.csv to Kaggle
5. **Iterate:** Experiment with hyperparameters

---

Good luck! 🚀

Questions? See ALGORITHMS_GUIDE.md or refer back to the clean notebook.
