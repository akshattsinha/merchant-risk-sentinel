import json
import os

import joblib
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_FILE = "data/processed/fraud_features.csv"
MODEL_FILE = "reports/fraud_model.joblib"
METADATA_FILE = "reports/fraud_model_metadata.json"


EXCLUDED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "address_id",
    "timestamp",
    "is_fraud",
    "fraud_type",
    "incident_id",
    "incident_type",
    "incident_severity"
]


def main():

    print("Loading feature dataset...")

    df = pd.read_csv(FEATURE_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["is_fraud"].astype(int)

    split_index = int(len(df) * 0.70)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    categorical_features = [
        column
        for column in X.columns
        if not is_numeric_dtype(X[column])
    ]

    numeric_features = [
        column
        for column in X.columns
        if is_numeric_dtype(X[column])
    ]

    print(f"Training features: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    print("Categorical columns:")

    for column in categorical_features:
        print(f"  - {column}")

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print("Training HistGradientBoosting model...")

    pipeline.fit(
        X_train,
        y_train
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    joblib.dump(
        artifact,
        MODEL_FILE
    )

    metadata = {
        "model": "HistGradientBoostingClassifier",
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": 0.30
    }

    with open(
        METADATA_FILE,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print()
    print("===== MODEL ARTIFACT =====")
    print(f"Features: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Operating threshold: 0.30")
    print()
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()