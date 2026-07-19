"""
50.007 ML Course — GenAI Text Detection (Simplified)
=====================================================
Kaggle Competition: Binary classification (human vs machine-generated text)
Metric: Macro F1 | Best approach: Custom TF-IDF + Linear SVM

Dataset:
- train_features.csv: 20,000 samples × (5000 pre-computed TF-IDF features)
- train.csv: raw text (id, text, label)
- test_features.csv & test.csv: unlabeled test set (6,999 samples)

Class balance: 62.5% machine-generated, 37.5% human-authored
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
import time

# ============================================================================
# SETUP
# ============================================================================

SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("predictions")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load train/test data and split into features & labels."""
    train_feat = pd.read_csv(DATA_DIR / "train_features.csv")
    test_feat = pd.read_csv(DATA_DIR / "test_features.csv")
    train_text = pd.read_csv(DATA_DIR / "train.csv")
    test_text = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [c for c in train_feat.columns if c not in ("id", "label")]

    X_feat_train = train_feat[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_feat["label"].to_numpy(dtype=int)
    X_feat_test = test_feat[feature_cols].to_numpy(dtype=np.float32)

    X_text_train = train_text["text"].values
    X_text_test = test_text["text"].values
    test_ids = test_feat["id"].values

    print(f"✓ Loaded: train {X_feat_train.shape}, test {X_feat_test.shape}")
    print(f"✓ Class balance: {np.bincount(y_train)} (62.5% machine, 37.5% human)")

    return X_feat_train, X_text_train, y_train, X_feat_test, X_text_test, test_ids


def train_val_split(X_feat, X_text, y):
    """Create 90% train / 10% validation split (stratified)."""
    tr_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.1, stratify=y, random_state=SEED
    )
    return tr_idx, val_idx


# ============================================================================
# BASELINE 1: Logistic Regression on provided 5000 TF-IDF features
# ============================================================================

def baseline_provided_features(X_feat_train, y_train, tr_idx, val_idx):
    """Benchmark: sklearn LogisticRegression on fixed 5000 features."""
    from sklearn.linear_model import LogisticRegression

    X_tr, y_tr = X_feat_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_feat_train[val_idx], y_train[val_idx]

    model = LogisticRegression(
        C=5.0, class_weight="balanced", max_iter=3000, random_state=SEED
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"\n📊 BASELINE (sklearn LogReg on 5000 features):")
    print(f"   Validation Macro F1: {f1_macro:.4f}")

    return model, f1_macro


# ============================================================================
# WINNER: Custom TF-IDF (word + char n-grams) + Linear SVM
# ============================================================================

def build_tfidf_vectorizer():
    """
    Create a rich text representation:
    - Word 1-2grams (TF-IDF, sublinear scaling): captures topical vocabulary
    - Character 3-5grams (char_wb): captures style fingerprint

    Together: ~640K sparse features.
    Why it wins: character n-grams capture punctuation/spacing habits that
    distinguish machine vs human authorship.
    """
    from sklearn.pipeline import FeatureUnion

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,  # Dampen raw term counts
        analyzer="word"
    )

    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        analyzer="char_wb",  # Character n-grams with word boundary
        max_features=300_000  # Cap to avoid memory explosion
    )

    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def winner_model(X_text_train, y_train, tr_idx, val_idx, X_text_test, test_ids):
    """
    Best model: Custom TF-IDF + Linear SVM
    - Validation Macro F1: 0.8229
    - Kaggle Public: 0.72990 (1st place)
    """
    X_text_tr, y_tr = X_text_train[tr_idx], y_train[tr_idx]
    X_text_val, y_val = X_text_train[val_idx], y_train[val_idx]

    print(f"\n🏆 WINNER: Custom TF-IDF + Linear SVM")
    print(f"   Building vectorizer...")

    vectorizer = build_tfidf_vectorizer()

    # Fit vectorizer on training text only
    t0 = time.time()
    X_tr_vec = vectorizer.fit_transform(X_text_tr)
    X_val_vec = vectorizer.transform(X_text_val)
    print(f"   ✓ Vectorized: {X_tr_vec.shape[1]} features in {time.time()-t0:.0f}s")

    # Tune SVM hyperparameters
    print(f"   Tuning SVM (C and class_weight)...")
    best_f1, best_params = -1.0, None

    for C in [0.25, 0.5, 1.0, 2.0]:
        for class_weight in [None, "balanced"]:
            svm = LinearSVC(C=C, class_weight=class_weight, random_state=SEED, max_iter=2000)
            svm.fit(X_tr_vec, y_tr)
            y_pred = svm.predict(X_val_vec)
            f1 = f1_score(y_val, y_pred, average="macro")

            if f1 > best_f1:
                best_f1, best_params = f1, {"C": C, "class_weight": class_weight}

            print(f"      C={C}, class_weight={class_weight}: F1={f1:.4f}")

    print(f"\n   Best config: {best_params} → Validation F1 = {best_f1:.4f}")

    # Refit on all 20K training samples
    print(f"   Refitting on full training set (20K samples)...")
    X_train_vec = vectorizer.fit_transform(X_text_train)
    svm_final = LinearSVC(**best_params, random_state=SEED, max_iter=2000)
    svm_final.fit(X_train_vec, y_train)

    # Predict test set
    X_test_vec = vectorizer.transform(X_text_test)
    y_test_pred = svm_final.predict(X_test_vec)

    # Save predictions
    output_file = OUTPUT_DIR / "Winner_Prediction.csv"
    df_pred = pd.DataFrame({"id": test_ids, "label": y_test_pred.astype(int)})
    df_pred.to_csv(output_file, index=False)
    print(f"   ✓ Saved predictions: {output_file} ({len(df_pred)} rows)")

    return best_f1, y_test_pred


# ============================================================================
# SIMPLE ALTERNATIVE: PCA + KNN (for comparison)
# ============================================================================

def alternative_pca_knn(X_feat_train, y_train, tr_idx, val_idx, X_feat_test, test_ids):
    """
    Alternative approach: dimensionality reduction (PCA) + k-NN classifier.
    Key insight: fewer components perform better (curse of dimensionality).
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    print(f"\n🔄 ALTERNATIVE: PCA(100) + KNN(k=2)")

    X_tr, y_tr = X_feat_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_feat_train[val_idx], y_train[val_idx]

    # PCA: fit on train only, transform train and validation
    pca = PCA(n_components=100, svd_solver="randomized", random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr)
    X_val_pca = pca.transform(X_val)

    print(f"   PCA(100) retains {pca.explained_variance_ratio_.sum():.2%} variance")

    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    knn.fit(X_tr_pca, y_tr)

    y_val_pred = knn.predict(X_val_pca)
    f1_macro = f1_score(y_val, y_val_pred, average="macro")

    print(f"   Validation Macro F1: {f1_macro:.4f}")

    # Predict test set
    pca_full = PCA(n_components=100, svd_solver="randomized", random_state=SEED)
    X_train_pca = pca_full.fit_transform(X_feat_train)
    X_test_pca = pca_full.transform(X_feat_test)

    knn_final = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    knn_final.fit(X_train_pca, y_train)
    y_test_pred = knn_final.predict(X_test_pca)

    # Save predictions
    output_file = OUTPUT_DIR / "Alternative_PCA_KNN_Prediction.csv"
    df_pred = pd.DataFrame({"id": test_ids, "label": y_test_pred.astype(int)})
    df_pred.to_csv(output_file, index=False)
    print(f"   ✓ Saved predictions: {output_file} ({len(df_pred)} rows)")

    return f1_macro, y_test_pred


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GenAI TEXT DETECTION — Simplified ML Pipeline")
    print("="*70)

    # Load data
    X_feat, X_text, y, X_feat_test, X_text_test, test_ids = load_data()
    tr_idx, val_idx = train_val_split(X_feat, X_text, y)

    # Run models
    baseline_model, baseline_f1 = baseline_provided_features(X_feat, y, tr_idx, val_idx)
    winner_f1, winner_pred = winner_model(X_text, y, tr_idx, val_idx, X_text_test, test_ids)
    alt_f1, alt_pred = alternative_pca_knn(X_feat, y, tr_idx, val_idx, X_feat_test, test_ids)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline (LogReg on 5000 features):  {baseline_f1:.4f}")
    print(f"Alternative (PCA + KNN):            {alt_f1:.4f}")
    print(f"🏆 WINNER (Custom TF-IDF + SVM):    {winner_f1:.4f}")
    print("\nPrediction files saved to predictions/")
    print("="*70)
