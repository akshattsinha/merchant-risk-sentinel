import os
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

INPUT_FILE = "data/processed/fraud_features.csv"
OUTPUT_FILE = "reports/strong_model_metrics.json"


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
            SimpleImputer(
                strategy="median"
            )
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

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )

    return Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])


def evaluate(
    model,
    X,
    y,
    dataset_name,
    threshold=0.50
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            predictions
        ).ravel()
    )

    result = {
        "dataset": dataset_name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    print()
    print(
        f"===== {dataset_name.upper()} ====="
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    print(
        f"False Positives: {fp:,}"
    )

    print(
        f"False Negatives: {fn:,}"
    )

    return result


def main():

    print(
        "Loading feature dataset..."
    )

    df = load_data()

    print(
        f"Total rows: {len(df):,}"
    )

    train, validation, test = (
        split_data(df)
    )

    print()
    print("===== TIME SPLIT =====")

    print(
        f"Train: {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test: {len(test):,}"
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

    print()
    print(
        f"Training features: "
        f"{X_train.shape[1]}"
    )

    print()
    print(
        "Building HistGradientBoosting model..."
    )

    model = build_model(
        X_train
    )

    print(
        "Training strong model..."
    )

    model.fit(
        X_train,
        y_train
    )

    train_result = evaluate(
        model,
        X_train,
        y_train,
        "train"
    )

    validation_result = evaluate(
        model,
        X_validation,
        y_validation,
        "validation"
    )

    test_result = evaluate(
        model,
        X_test,
        y_test,
        "held-out test"
    )

    results = {
        "model": "HistGradientBoostingClassifier",
        "train": train_result,
        "validation": validation_result,
        "held_out_test": test_result
    }

    os.makedirs(
        "reports",
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        f"Metrics saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()