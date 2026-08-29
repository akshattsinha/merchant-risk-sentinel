import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

INPUT_FILE = "data/processed/fraud_features.csv"
OUTPUT_FILE = "reports/threshold_analysis.csv"
METRICS_FILE = "reports/optimized_metrics.json"

FALSE_POSITIVE_COST = 500
FALSE_NEGATIVE_COST = 5000


def load_data():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def split_data(df):
    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[
        train_end:validation_end
    ].copy()
    test = df.iloc[
        validation_end:
    ].copy()

    return train, validation, test


def prepare_features(df):
    excluded_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "timestamp",

        "is_fraud",
        "fraud_type",
        "incident_id",
        "incident_type",
        "incident_severity",

        "device_id",
        "ip_id",
        "address_id"
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    y = df["is_fraud"]

    return X, y


def build_model(X):

    categorical_columns = [
        "payment_method",
        "location"
    ]

    categorical_columns = [
        column
        for column in categorical_columns
        if column in X.columns
    ]

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    return Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            classifier
        )
    ])


def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions
        ).ravel()
    )

    estimated_cost = (
        fp * FALSE_POSITIVE_COST
        +
        fn * FALSE_NEGATIVE_COST
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "estimated_cost": estimated_cost
    }


def main():

    print(
        "Loading feature dataset..."
    )

    df = load_data()

    train, validation, test = (
        split_data(df)
    )

    X_train, y_train = (
        prepare_features(train)
    )

    X_validation, y_validation = (
        prepare_features(validation)
    )

    X_test, y_test = (
        prepare_features(test)
    )

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print()
    print(
        "Training baseline model..."
    )

    model = build_model(
        X_train
    )

    model.fit(
        X_train,
        y_train
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    validation_results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            y_validation,
            validation_probabilities,
            round(
                float(threshold),
                2
            )
        )

        validation_results.append(
            result
        )

    results_df = pd.DataFrame(
        validation_results
    )

    results_df = results_df.sort_values(
        "estimated_cost"
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    best_threshold = float(
        results_df.iloc[0]["threshold"]
    )

    print()
    print(
        "===== VALIDATION THRESHOLD ANALYSIS ====="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected threshold:"
    )

    print(
        f"{best_threshold:.2f}"
    )

    print()
    print(
        "Reason:"
    )

    print(
        "Lowest estimated merchant loss "
        "on validation data."
    )

    # -----------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_result = evaluate_threshold(
        y_test,
        test_probabilities,
        best_threshold
    )

    print()
    print(
        "===== HELD-OUT TEST ====="
    )

    print(
        f"Threshold: "
        f"{test_result['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{test_result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{test_result['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{test_result['f1']:.4f}"
    )

    print(
        f"True Negatives: "
        f"{test_result['true_negatives']:,}"
    )

    print(
        f"False Positives: "
        f"{test_result['false_positives']:,}"
    )

    print(
        f"False Negatives: "
        f"{test_result['false_negatives']:,}"
    )

    print(
        f"True Positives: "
        f"{test_result['true_positives']:,}"
    )

    print(
        f"Estimated Cost: ₹"
        f"{test_result['estimated_cost']:,.0f}"
    )

    output = {
    "false_positive_cost": int(
        FALSE_POSITIVE_COST
    ),
    "false_negative_cost": int(
        FALSE_NEGATIVE_COST
    ),
    "selected_threshold": float(
        best_threshold
    ),
    "validation": (
        results_df.astype(
            object
        ).where(
            pd.notna(results_df),
            None
        ).to_dict(
            orient="records"
        )
    ),
    "held_out_test": {
        key: (
            float(value)
            if isinstance(
                value,
                (np.floating,)
            )
            else int(value)
            if isinstance(
                value,
                (np.integer,)
            )
            else value
        )
        for key, value in test_result.items()
    }
}

    with open(
        METRICS_FILE,
        "w"
    ) as file:
        json.dump(
            output,
            file,
            indent=4
        )

    print()
    print(
        f"Threshold analysis saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Optimized metrics saved to: "
        f"{METRICS_FILE}"
    )


if __name__ == "__main__":
    main()