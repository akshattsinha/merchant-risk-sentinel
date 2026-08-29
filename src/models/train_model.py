import os
import json
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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

INPUT_FILE = "data/processed/fraud_features.csv"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def build_pipeline(X):
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

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    return pipeline


def evaluate_model(
    model,
    X,
    y,
    dataset_name
):
    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.50
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

    matrix = confusion_matrix(
        y,
        predictions
    )

    tn, fp, fn, tp = matrix.ravel()

    results = {
        "dataset": dataset_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
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

    print(matrix)

    print()
    print(
        f"False Positives: {fp:,}"
    )

    print(
        f"False Negatives: {fn:,}"
    )

    return results


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

    print(
        "Building baseline model..."
    )

    model = build_pipeline(
        X_train
    )

    print(
        "Training Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train
    )

    train_results = evaluate_model(
        model,
        X_train,
        y_train,
        "train"
    )

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "validation"
    )

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "held-out test"
    )

    results = {
        "train": train_results,
        "validation": validation_results,
        "held_out_test": test_results
    }

    output_file = os.path.join(
        OUTPUT_DIR,
        "baseline_metrics.json"
    )

    with open(
        output_file,
        "w"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        f"Metrics saved to: {output_file}"
    )


if __name__ == "__main__":
    main()