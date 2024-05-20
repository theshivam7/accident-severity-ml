"""
Train a CatBoost classifier on the preprocessed accident data.

Usage:
    python3 train.py

Expects:
    processed_data.csv

Outputs:
    model.cbm
    model_meta.json
"""

import json

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

INPUT_FILE = "processed_data.csv"
MODEL_FILE = "model.cbm"
META_FILE = "model_meta.json"

CAT_FEATURES = ["District_Name", "Road_Type"]
FEATURES = ["District_Name", "Road_Type", "Month", "Year"]
TARGET = "Severity"


def main() -> None:
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows: {len(df):,}")

    X = df[FEATURES].copy()
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=50,
        cat_features=CAT_FEATURES,
        class_weights={label: (len(y) / (2 * (y == label).sum())) for label in y.unique()},
    )
    print("\nTraining ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    model.save_model(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

    meta = {
        "districts": sorted(df["District_Name"].dropna().unique().tolist()),
        "road_types": sorted(df["Road_Type"].dropna().unique().tolist()),
        "severity_classes": sorted(df[TARGET].unique().tolist()),
        "features": FEATURES,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_FILE}")


if __name__ == "__main__":
    main()
