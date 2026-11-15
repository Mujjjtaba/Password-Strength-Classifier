import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "data.csv",
    BASE_DIR / "data.csv",
]
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Weak", "Moderate", "Strong"]
LABELS = [0, 1, 2]


def char_tokenizer(text):
    """Tokenize a password into individual characters."""
    return list(text)


def load_dataset():
    if DATA_PATH is None:
        expected = "\n".join(str(path) for path in DATA_CANDIDATES)
        raise FileNotFoundError(
            "Dataset not found. Expected one of these locations:\n" + expected
        )

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_columns = {"password", "strength"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    original_rows = len(df)
    df = df.dropna(subset=["strength"]).copy()
    df["password"] = df["password"].fillna("").astype(str)
    df = df[df["password"].str.len() > 0].copy()

    # Project policy: reject all whitespace-containing passwords.
    has_whitespace = df["password"].str.contains(r"\s", regex=True, na=False)
    removed_whitespace = int(has_whitespace.sum())
    df = df.loc[~has_whitespace].copy()

    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    df = df.dropna(subset=["strength"])
    df = df[df["strength"].isin(LABELS)].copy()
    df["strength"] = df["strength"].astype(int)

    if df.empty or df["strength"].nunique() != 3:
        raise ValueError("After cleaning, the dataset must contain all three classes: 0, 1, 2.")

    print(f"Original rows: {original_rows}")
    print(f"Rows removed for whitespace: {removed_whitespace}")
    print(f"Cleaned rows: {len(df)}")
    print("\nClass distribution:")
    print(df["strength"].value_counts().sort_index())

    return df


def main():
    df = load_dataset()
    X = df["password"]
    y = df["strength"]

    print("\nCreating stratified 80:20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print("\nFitting character-level TF-IDF on training data only...")
    tfidf = TfidfVectorizer(
        tokenizer=char_tokenizer,
        token_pattern=None,
        lowercase=False,
        dtype=np.float32,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training feature shape:", X_train_tfidf.shape)
    print("Testing feature shape:", X_test_tfidf.shape)

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
        "naive_bayes": MultinomialNB(),
    }

    metrics = {}
    reports = {}

    print("\nStarting model training...")
    for name, model in models.items():
        print(f"\nTraining: {name}")
        start = time.time()

        if name == "logistic_regression":
            model.fit(X_train_tfidf, y_train)
        else:
            # Both XGBoost and MultinomialNB support fit(sample_weight=...).
            model.fit(
                X_train_tfidf,
                y_train,
                sample_weight=sample_weights,
            )

        training_time = time.time() - start
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision_weighted = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall_weighted = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1_weighted = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        precision_macro = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1_macro = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        metrics[name] = {
            "training_time_sec": round(training_time, 2),
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision_weighted * 100, 2),
            "recall": round(recall_weighted * 100, 2),
            "f1_score": round(f1_weighted * 100, 2),
            "macro_precision": round(precision_macro * 100, 2),
            "macro_recall": round(recall_macro * 100, 2),
            "macro_f1_score": round(f1_macro * 100, 2),
        }

        reports[name] = classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        np.save(MODEL_DIR / f"{name}_confusion_matrix.npy", cm)
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

        print(
            f"{name}: accuracy={metrics[name]['accuracy']}%, "
            f"weighted_f1={metrics[name]['f1_score']}%, "
            f"macro_f1={metrics[name]['macro_f1_score']}%, "
            f"time={metrics[name]['training_time_sec']}s"
        )

    joblib.dump(tfidf, MODEL_DIR / "tfidf.pkl")

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(
        MODEL_DIR / "classification_reports.json", "w", encoding="utf-8"
    ) as file:
        json.dump(reports, file, indent=4)

    joblib.dump(
        {"X_test": X_test.tolist(), "y_test": y_test.tolist()},
        MODEL_DIR / "test_data.pkl",
        compress=3,
    )

    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETED")
    print("=" * 55)
    for name, result in metrics.items():
        print(
            f"{name}: accuracy={result['accuracy']}%, "
            f"precision={result['precision']}%, "
            f"recall={result['recall']}%, "
            f"weighted_f1={result['f1_score']}%, "
            f"macro_f1={result['macro_f1_score']}%"
        )

    print(f"\nAll artifacts saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
