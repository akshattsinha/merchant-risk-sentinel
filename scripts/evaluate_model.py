from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "fraud_features.csv"
)

MODEL_FILE = (
    ROOT_DIR
    / "reports"
    / "fraud_model.joblib"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "reports"
    / "strong_optimized_metrics.json"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# ============================================================
# BUSINESS COST MODEL
# ============================================================

# Estimated operational cost for investigating one
# legitimate transaction that was incorrectly flagged.
#
# This is configurable. It is NOT a fraud loss.
FALSE_POSITIVE_INVESTIGATION_COST = 500.0


# A false negative is a real fraudulent transaction
# that the model failed to flag.
#
# We use the actual transaction amount as the
# potential financial exposure.
#
# Example:
# A missed ₹50,000 fraudulent transaction contributes
# ₹50,000 to false-negative exposure.
#
# This is a conservative prototype business-cost model.
FALSE_NEGATIVE_AMOUNT_MULTIPLIER = 1.0


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
    amounts,
):
    """
    Calculate classification metrics and financial costs
    for a particular probability threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    # --------------------------------------------------------
    # False-positive financial cost
    # --------------------------------------------------------

    false_positive_cost = (
        fp
        * FALSE_POSITIVE_INVESTIGATION_COST
    )

    # --------------------------------------------------------
    # False-negative financial cost
    # --------------------------------------------------------
    #
    # For each missed fraudulent transaction, use its
    # actual transaction amount as the potential exposure.
    #
    # FN transactions are:
    # actual fraud + model predicted legitimate.
    # --------------------------------------------------------

    false_negative_mask = (
        (y_true.to_numpy() == 1)
        & (predictions == 0)
    )

    false_negative_amount = (
        amounts.to_numpy()[
            false_negative_mask
        ]
    )

    false_negative_cost = (
        false_negative_amount
        .sum()
        * FALSE_NEGATIVE_AMOUNT_MULTIPLIER
    )

    expected_loss = (
        false_positive_cost
        + false_negative_cost
    )

    return {
        "threshold": float(threshold),

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),

        "false_positive_cost": float(
            false_positive_cost
        ),

        "false_negative_cost": float(
            false_negative_cost
        ),

        "expected_loss": float(
            expected_loss
        ),

        "total_fraud_exposure_missed": float(
            false_negative_amount.sum()
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MERCHANT RISK SENTINEL")
print("THRESHOLD + FINANCIAL COST OPTIMIZATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "timestamp",
    "is_fraud",
    "amount",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"is missing from the dataset."
        )


# ============================================================
# TIMESTAMP
# ============================================================

print(
    "\nSorting transactions chronologically..."
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

df = df.dropna(
    subset=["timestamp"]
).copy()

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

print(
    "\nLoading existing model..."
)

artifact = joblib.load(
    MODEL_FILE
)

if not isinstance(
    artifact,
    dict,
):

    raise ValueError(
        "fraud_model.joblib does not "
        "contain the expected model artifact."
    )


pipeline = artifact[
    "pipeline"
]

feature_columns = artifact[
    "feature_columns"
]

print(
    f"Pipeline type: "
    f"{type(pipeline).__name__}"
)

print(
    f"Model features: "
    f"{len(feature_columns)}"
)


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing model features:\n"
        + "\n".join(
            missing_features
        )
    )


# ============================================================
# PREPARE X / Y
# ============================================================

X = df[
    feature_columns
].copy()

y = (
    pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

amounts = (
    pd.to_numeric(
        df["amount"],
        errors="coerce",
    )
    .fillna(0.0)
    .clip(lower=0.0)
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

total_rows = len(df)

train_end = int(
    total_rows
    * TRAIN_FRACTION
)

validation_end = int(
    total_rows
    * (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
    )
)


train_df = df.iloc[
    :train_end
].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()


X_train = train_df[
    feature_columns
]

y_train = y.iloc[
    :train_end
]


X_validation = validation_df[
    feature_columns
]

y_validation = y.iloc[
    train_end:validation_end
]


validation_amounts = amounts.iloc[
    train_end:validation_end
]


X_test = test_df[
    feature_columns
]

y_test = y.iloc[
    validation_end:
]


test_amounts = amounts.iloc[
    validation_end:
]


# ============================================================
# DISPLAY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL DATA SPLIT")
print("=" * 70)

print(
    f"\nTraining:     {len(train_df):,}"
)

print(
    f"Validation:   {len(validation_df):,}"
)

print(
    f"Held-out:     {len(test_df):,}"
)

print(
    "\nFraud counts:"
)

print(
    f"Training:     {int(y_train.sum()):,}"
)

print(
    f"Validation:   {int(y_validation.sum()):,}"
)

print(
    f"Held-out:     {int(y_test.sum()):,}"
)

print(
    "\nTime ranges:"
)

print(
    f"Training:   "
    f"{train_df['timestamp'].min()}"
    f" → "
    f"{train_df['timestamp'].max()}"
)

print(
    f"Validation: "
    f"{validation_df['timestamp'].min()}"
    f" → "
    f"{validation_df['timestamp'].max()}"
)

print(
    f"Held-out:   "
    f"{test_df['timestamp'].min()}"
    f" → "
    f"{test_df['timestamp'].max()}"
)


# ============================================================
# TRAIN EVALUATION COPY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING EVALUATION COPY")
print("=" * 70)

print(
    "\nOriginal fraud_model.joblib "
    "will NOT be modified."
)

evaluation_pipeline = clone(
    pipeline
)

evaluation_pipeline.fit(
    X_train,
    y_train,
)

print(
    "Evaluation model trained."
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print(
    "\nGenerating validation probabilities..."
)

validation_probabilities = (
    evaluation_pipeline
    .predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# TEST PROBABILITIES
# ============================================================

print(
    "Generating held-out test probabilities..."
)

test_probabilities = (
    evaluation_pipeline
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE: THRESHOLD 0.50")
print("=" * 70)

baseline = calculate_metrics(
    y_validation,
    validation_probabilities,
    0.50,
    validation_amounts,
)

print(
    f"\nPrecision: "
    f"{baseline['precision'] * 100:.2f}%"
)

print(
    f"Recall:    "
    f"{baseline['recall'] * 100:.2f}%"
)

print(
    f"F1:        "
    f"{baseline['f1'] * 100:.2f}%"
)

print(
    f"FP Cost:   "
    f"₹{baseline['false_positive_cost']:,.2f}"
)

print(
    f"FN Cost:   "
    f"₹{baseline['false_negative_cost']:,.2f}"
)

print(
    f"Expected:  "
    f"₹{baseline['expected_loss']:,.2f}"
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

print(
    "\nEvaluating thresholds:"
)

print(
    ", ".join(
        f"{threshold:.2f}"
        for threshold in THRESHOLDS
    )
)


threshold_results = []

for threshold in THRESHOLDS:

    result = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold,
        validation_amounts,
    )

    threshold_results.append(
        result
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# PRINT THRESHOLD TABLE
# ============================================================

display_columns = [
    "threshold",
    "precision",
    "recall",
    "f1",
    "false_positives",
    "false_negatives",
    "false_positive_cost",
    "false_negative_cost",
    "expected_loss",
]

print(
    "\nThreshold comparison:"
)

print(
    threshold_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "threshold":
                lambda x:
                f"{x:.2f}",

            "precision":
                lambda x:
                f"{x * 100:.2f}%",

            "recall":
                lambda x:
                f"{x * 100:.2f}%",

            "f1":
                lambda x:
                f"{x * 100:.2f}%",

            "false_positive_cost":
                lambda x:
                f"₹{x:,.0f}",

            "false_negative_cost":
                lambda x:
                f"₹{x:,.0f}",

            "expected_loss":
                lambda x:
                f"₹{x:,.0f}",
        },
    )
)


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================

best_row = (
    threshold_df
    .sort_values(
        [
            "expected_loss",
            "f1",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .iloc[0]
)


selected_threshold = float(
    best_row[
        "threshold"
    ]
)


print("\n" + "=" * 70)
print("SELECTED OPERATING THRESHOLD")
print("=" * 70)

print(
    f"\nSelected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Validation expected loss: "
    f"₹{best_row['expected_loss']:,.2f}"
)

print(
    f"Validation precision: "
    f"{best_row['precision'] * 100:.2f}%"
)

print(
    f"Validation recall: "
    f"{best_row['recall'] * 100:.2f}%"
)

print(
    f"Validation F1: "
    f"{best_row['f1'] * 100:.2f}%"
)


# ============================================================
# FINAL HELD-OUT TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL HELD-OUT TEST")
print("=" * 70)

print(
    "\nApplying the selected threshold "
    f"{selected_threshold:.2f}"
    " to the untouched test set."
)

final_test = calculate_metrics(
    y_test,
    test_probabilities,
    selected_threshold,
    test_amounts,
)


# ============================================================
# TEST PR-AUC
# ============================================================

test_pr_auc = (
    average_precision_score(
        y_test,
        test_probabilities,
    )
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    f"\nThreshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Precision: "
    f"{final_test['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{final_test['recall'] * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{final_test['f1'] * 100:.2f}%"
)

print(
    f"PR-AUC: "
    f"{test_pr_auc * 100:.2f}%"
)

print(
    f"\nTrue Positives: "
    f"{final_test['true_positives']:,}"
)

print(
    f"False Positives: "
    f"{final_test['false_positives']:,}"
)

print(
    f"True Negatives: "
    f"{final_test['true_negatives']:,}"
)

print(
    f"False Negatives: "
    f"{final_test['false_negatives']:,}"
)

print(
    f"\nFalse-positive cost: "
    f"₹{final_test['false_positive_cost']:,.2f}"
)

print(
    f"False-negative cost: "
    f"₹{final_test['false_negative_cost']:,.2f}"
)

print(
    f"Expected loss: "
    f"₹{final_test['expected_loss']:,.2f}"
)


# ============================================================
# BUILD FINAL METRICS ARTIFACT
# ============================================================

model_name = (
    type(
        pipeline.named_steps[
            list(
                pipeline.named_steps.keys()
            )[-1]
        ]
    ).__name__
)


metrics = {

    "model":
        model_name,

    "evaluation_type":
        "temporal_holdout_with_validation_threshold_optimization",

    "dataset":
        str(
            DATA_FILE.relative_to(
                ROOT_DIR
            )
        ),

    "total_samples":
        int(total_rows),

    "train_samples":
        int(len(train_df)),

    "validation_samples":
        int(len(validation_df)),

    "test_samples":
        int(len(test_df)),

    "train_fraud_count":
        int(y_train.sum()),

    "validation_fraud_count":
        int(y_validation.sum()),

    "test_fraud_count":
        int(y_test.sum()),

    "train_fraction":
        TRAIN_FRACTION,

    "validation_fraction":
        VALIDATION_FRACTION,

    "test_fraction":
        TEST_FRACTION,

    "selected_threshold":
        selected_threshold,

    "threshold_candidates":
        THRESHOLDS,

    "threshold_selection":
        "minimum_validation_expected_loss",

    "business_cost_model": {

        "false_positive_investigation_cost":
            FALSE_POSITIVE_INVESTIGATION_COST,

        "false_negative_amount_multiplier":
            FALSE_NEGATIVE_AMOUNT_MULTIPLIER,

        "false_negative_cost_definition":
            "sum of actual transaction amounts "
            "for missed fraudulent transactions",

        "expected_loss_definition":
            "false_positive_cost + "
            "false_negative_cost",
    },

    "validation_threshold_results":
        threshold_results,

    "baseline_validation":
        baseline,

    "held_out_test": {

        "precision":
            final_test["precision"],

        "recall":
            final_test["recall"],

        "f1":
            final_test["f1"],

        "pr_auc":
            float(test_pr_auc),

        "true_positives":
            final_test["true_positives"],

        "false_positives":
            final_test["false_positives"],

        "true_negatives":
            final_test["true_negatives"],

        "false_negatives":
            final_test["false_negatives"],

        "false_positive_cost":
            final_test[
                "false_positive_cost"
            ],

        "false_negative_cost":
            final_test[
                "false_negative_cost"
            ],

        "expected_loss":
            final_test[
                "expected_loss"
            ],

        "total_fraud_exposure_missed":
            final_test[
                "total_fraud_exposure_missed"
            ],
    },

    "train_start":
        str(
            train_df[
                "timestamp"
            ].min()
        ),

    "train_end":
        str(
            train_df[
                "timestamp"
            ].max()
        ),

    "validation_start":
        str(
            validation_df[
                "timestamp"
            ].min()
        ),

    "validation_end":
        str(
            validation_df[
                "timestamp"
            ].max()
        ),

    "test_start":
        str(
            test_df[
                "timestamp"
            ].min()
        ),

    "test_end":
        str(
            test_df[
                "timestamp"
            ].max()
        ),

    "feature_count":
        len(feature_columns),

    "feature_columns":
        feature_columns,
}


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)

print(
    "\nMetrics saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nOriginal fraud_model.joblib "
    "was NOT modified."
)

print(
    "\nThe threshold was selected using "
    "validation data only."
)

print(
    "The final metrics were calculated "
    "on the untouched temporal test set."
)

print(
    "\nNext step:"
)

print(
    "Update the dashboard to display "
    "these real evaluation results."
)