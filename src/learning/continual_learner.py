"""
Feedback-driven continual learning for Merchant Risk Sentinel.

Responsibilities
----------------
1. Store confirmed analyst feedback in SQLite.
2. Track AI recommendation vs human decision.
3. Store investigation outcomes and notes.
4. Use ONLY confirmed fraud / legitimate outcomes as training labels.
5. Retrain after a minimum number of confirmed labels.
6. Evaluate candidate model on a chronological holdout set.
7. Promote the candidate only when it passes the promotion gate.
8. Version promoted models.
9. Keep a small model registry.
10. Prevent concurrent retraining.

Important
---------
Operational decisions such as HOLD, REVIEW, and ALLOW are NOT treated
as ground truth.

Ground truth comes only from:

    CONFIRMED_FRAUD
    CONFIRMED_LEGITIMATE

INCONCLUSIVE investigations are deliberately excluded from training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

FEEDBACK_DIR = (
    DATA_DIR
    / "feedback"
)

FEEDBACK_DB = (
    FEEDBACK_DIR
    / "feedback.db"
)

FEATURE_FILE = (
    DATA_DIR
    / "processed"
    / "fraud_features.csv"
)

MODEL_FILE = (
    REPORTS_DIR
    / "fraud_model.joblib"
)

MODEL_METADATA_FILE = (
    REPORTS_DIR
    / "fraud_model_metadata.json"
)

ACTIVE_MODEL_FILE = (
    REPORTS_DIR
    / "active_model.json"
)

MODEL_REGISTRY_DIR = (
    REPORTS_DIR
    / "model_versions"
)

LOCK_FILE = (
    REPORTS_DIR
    / ".retrain.lock"
)

MAINTENANCE_STATUS_FILE = (
    REPORTS_DIR
    / "maintenance_status.json"
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MIN_FEEDBACK = 10

DEFAULT_THRESHOLD = 0.30

REGISTRY_RETENTION = 5

MODEL_RANDOM_STATE = 42

MODEL_VERSION_PREFIX = "v"

PROMOTION_TOLERANCE = 0.02


# ============================================================
# DIRECTORY SETUP
# ============================================================

def _ensure_directories():
    """Create directories required by the learning system."""

    FEEDBACK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def _connect():
    """
    Open the SQLite feedback database.

    Row factory allows access using row["column_name"].
    """

    _ensure_directories()

    connection = sqlite3.connect(
        FEEDBACK_DB
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def _initialize_database():
    """
    Create the feedback table if it does not exist.

    Existing installations are migrated by adding any missing
    columns instead of deleting the existing database.
    """

    _ensure_directories()

    with _connect() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (

                transaction_id TEXT PRIMARY KEY,

                label INTEGER NOT NULL,

                ground_truth TEXT,

                ai_recommendation TEXT,

                human_decision TEXT,

                final_decision TEXT NOT NULL,

                reason TEXT NOT NULL,

                investigation_notes TEXT,

                submitted_at TEXT NOT NULL,

                model_version TEXT,

                fraud_probability REAL,

                risk_score REAL,

                risk_level TEXT,

                transaction_json TEXT NOT NULL,

                feature_json TEXT NOT NULL

            )
            """
        )

        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(feedback)"
            ).fetchall()
        }

        migrations = {

            "ground_truth":
                "TEXT",

            "ai_recommendation":
                "TEXT",

            "human_decision":
                "TEXT",

            "investigation_notes":
                "TEXT",

            "fraud_probability":
                "REAL",

            "risk_score":
                "REAL",

            "risk_level":
                "TEXT",

        }

        for column, column_type in migrations.items():

            if column not in existing_columns:

                connection.execute(
                    f"""
                    ALTER TABLE feedback
                    ADD COLUMN {column}
                    {column_type}
                    """
                )

        connection.commit()


# ============================================================
# MODEL VERSION
# ============================================================

def _current_model_version():
    """
    Read the currently active model version.

    Falls back to metadata if active_model.json does not exist.
    """

    if ACTIVE_MODEL_FILE.exists():

        try:

            with open(
                ACTIVE_MODEL_FILE,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            version = data.get(
                "version"
            )

            if version:
                return str(version)

        except Exception:
            pass

    if MODEL_METADATA_FILE.exists():

        try:

            with open(
                MODEL_METADATA_FILE,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            for key in (
                "version",
                "model_version",
                "artifact_version",
            ):

                if data.get(key):

                    return str(
                        data[key]
                    )

        except Exception:
            pass

    return None


# ============================================================
# FEEDBACK STORAGE
# ============================================================

def record_feedback(
    transaction_id,
    label,
    final_decision,
    reason,
    transaction,
    features,
    model_version=None,
    ground_truth=None,
    ai_recommendation=None,
    human_decision=None,
    investigation_notes=None,
    fraud_probability=None,
    risk_score=None,
    risk_level=None,
):
    """
    Store analyst feedback.

    `label` must represent confirmed ground truth:

        1 = confirmed fraud
        0 = confirmed legitimate

    Do NOT call this function with an operational action as the label.
    """

    _initialize_database()

    transaction_id = str(
        transaction_id
    ).strip()

    if not transaction_id:

        raise ValueError(
            "transaction_id cannot be empty."
        )

    label = int(label)

    if label not in (0, 1):

        raise ValueError(
            "label must be either 0 or 1."
        )

    final_decision = (
        str(
            final_decision
        )
        .strip()
        .upper()
    )

    allowed_decisions = {
        "ALLOW",
        "REVIEW",
        "HOLD",
    }

    if final_decision not in allowed_decisions:

        raise ValueError(
            "final_decision must be "
            "ALLOW, REVIEW, or HOLD."
        )

    reason = str(
        reason
    ).strip()

    if len(reason) < 3:

        raise ValueError(
            "reason must contain at least "
            "3 characters."
        )

    if ground_truth is not None:

        ground_truth = (
            str(ground_truth)
            .strip()
            .upper()
        )

        allowed_ground_truth = {
            "CONFIRMED_FRAUD",
            "CONFIRMED_LEGITIMATE",
        }

        if ground_truth not in allowed_ground_truth:

            raise ValueError(
                "ground_truth must be "
                "CONFIRMED_FRAUD or "
                "CONFIRMED_LEGITIMATE."
            )

        expected_label = (
            1
            if ground_truth
            == "CONFIRMED_FRAUD"
            else 0
        )

        if label != expected_label:

            raise ValueError(
                "label does not match ground_truth."
            )

    if ai_recommendation is not None:

        ai_recommendation = (
            str(
                ai_recommendation
            )
            .strip()
            .upper()
        )

    if human_decision is not None:

        human_decision = (
            str(
                human_decision
            )
            .strip()
            .upper()
        )

    if model_version is None:

        model_version = (
            _current_model_version()
        )

    submitted_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    transaction_json = json.dumps(
        transaction or {},
        default=str,
    )

    feature_json = json.dumps(
        features or {},
        default=str,
    )

    with _connect() as connection:

        connection.execute(
            """
            INSERT INTO feedback (

                transaction_id,

                label,

                ground_truth,

                ai_recommendation,

                human_decision,

                final_decision,

                reason,

                investigation_notes,

                submitted_at,

                model_version,

                fraud_probability,

                risk_score,

                risk_level,

                transaction_json,

                feature_json

            )

            VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )

            ON CONFLICT(transaction_id)

            DO UPDATE SET

                label =
                    excluded.label,

                ground_truth =
                    excluded.ground_truth,

                ai_recommendation =
                    excluded.ai_recommendation,

                human_decision =
                    excluded.human_decision,

                final_decision =
                    excluded.final_decision,

                reason =
                    excluded.reason,

                investigation_notes =
                    excluded.investigation_notes,

                submitted_at =
                    excluded.submitted_at,

                model_version =
                    excluded.model_version,

                fraud_probability =
                    excluded.fraud_probability,

                risk_score =
                    excluded.risk_score,

                risk_level =
                    excluded.risk_level,

                transaction_json =
                    excluded.transaction_json,

                feature_json =
                    excluded.feature_json
            """,
            (
                transaction_id,

                label,

                ground_truth,

                ai_recommendation,

                human_decision,

                final_decision,

                reason,

                investigation_notes,

                submitted_at,

                model_version,

                fraud_probability,

                risk_score,

                risk_level,

                transaction_json,

                feature_json,
            ),
        )

        connection.commit()

    return {
        "status":
            "stored",

        "transaction_id":
            transaction_id,

        "label":
            label,

        "ground_truth":
            ground_truth,

        "ai_recommendation":
            ai_recommendation,

        "human_decision":
            human_decision,

        "final_decision":
            final_decision,

        "model_version":
            model_version,

        "submitted_at":
            submitted_at,
    }


# ============================================================
# FEEDBACK STATISTICS
# ============================================================

def feedback_stats():
    """
    Return feedback counts.

    Only confirmed outcomes stored in the feedback table
    are considered training feedback.
    """

    _initialize_database()

    with _connect() as connection:

        total = connection.execute(
            """
            SELECT COUNT(*)
            FROM feedback
            """
        ).fetchone()[0]

        fraud = connection.execute(
            """
            SELECT COUNT(*)
            FROM feedback
            WHERE label = 1
            """
        ).fetchone()[0]

        legitimate = connection.execute(
            """
            SELECT COUNT(*)
            FROM feedback
            WHERE label = 0
            """
        ).fetchone()[0]

        ai_accepted = connection.execute(
            """
            SELECT COUNT(*)
            FROM feedback
            WHERE
                ai_recommendation IS NOT NULL
                AND human_decision =
                    ai_recommendation
            """
        ).fetchone()[0]

        ai_overridden = connection.execute(
            """
            SELECT COUNT(*)
            FROM feedback
            WHERE
                ai_recommendation IS NOT NULL
                AND human_decision IS NOT NULL
                AND human_decision !=
                    ai_recommendation
            """
        ).fetchone()[0]

    agreement_total = (
        ai_accepted
        + ai_overridden
    )

    agreement_rate = (
        ai_accepted / agreement_total
        if agreement_total > 0
        else None
    )

    return {

        "total_feedback":
            int(total),

        "confirmed_fraud":
            int(fraud),

        "confirmed_legitimate":
            int(legitimate),

        "ai_accepted":
            int(ai_accepted),

        "ai_overridden":
            int(ai_overridden),

        "ai_agreement_rate":
            (
                round(
                    agreement_rate,
                    4,
                )
                if agreement_rate is not None
                else None
            ),

    }


# ============================================================
# LOAD FEEDBACK
# ============================================================

def _load_feedback():
    """
    Load confirmed feedback records.

    The database contains only confirmed labels because
    INCONCLUSIVE cases are rejected before training storage.
    """

    _initialize_database()

    with _connect() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM feedback
            WHERE label IN (0, 1)
            ORDER BY submitted_at ASC
            """
        ).fetchall()

    if not rows:

        return pd.DataFrame()

    records = []

    for row in rows:

        record = dict(row)

        try:

            features = json.loads(
                record.get(
                    "feature_json",
                    "{}",
                )
            )

        except Exception:

            features = {}

        record["parsed_features"] = (
            features
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# LOAD ACTIVE ARTIFACT
# ============================================================

def _load_active_artifact():
    """
    Load the existing model artifact.

    Expected artifact structure:

        {
            "pipeline": sklearn_pipeline,
            "feature_columns": [...]
        }
    """

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Active model not found: "
            f"{MODEL_FILE}"
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
            "contain the expected artifact dictionary."
        )

    if "pipeline" not in artifact:

        raise ValueError(
            "Model artifact is missing "
            "'pipeline'."
        )

    if "feature_columns" not in artifact:

        raise ValueError(
            "Model artifact is missing "
            "'feature_columns'."
        )

    return artifact


# ============================================================
# BUILD FEEDBACK DATASET
# ============================================================

def _prepare_training_data():
    """
    Combine the original labeled dataset with confirmed analyst
    feedback.

    Existing historical data remains the foundation.

    Confirmed feedback is appended using the exact feature schema
    from the active model.

    Transaction timestamps are retained separately for chronological
    validation and are NOT used as model features.
    """

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{FEATURE_FILE}"
        )

    base_df = pd.read_csv(
        FEATURE_FILE
    )

    artifact = (
        _load_active_artifact()
    )

    feature_columns = list(
        artifact[
            "feature_columns"
        ]
    )

    if "is_fraud" not in base_df.columns:
        raise ValueError(
            "Base feature dataset is missing "
            "'is_fraud'."
        )

    if "timestamp" not in base_df.columns:
        raise ValueError(
            "Base feature dataset is missing "
            "'timestamp'."
        )

    missing_base_features = [
        column
        for column in feature_columns
        if column not in base_df.columns
    ]

    if missing_base_features:
        raise ValueError(
            "Base feature dataset is missing "
            f"model features: "
            f"{missing_base_features}"
        )

    # --------------------------------------------------------
    # Original historical data
    # --------------------------------------------------------

    X_base = (
        base_df[
            feature_columns
        ]
        .copy()
    )

    y_base = (
        pd.to_numeric(
            base_df[
                "is_fraud"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    timestamps_base = pd.to_datetime(
        base_df["timestamp"],
        errors="coerce",
    )

    if timestamps_base.isna().any():
        raise ValueError(
            "Base feature dataset contains "
            "invalid timestamps."
        )

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    feedback_df = (
        _load_feedback()
    )

    if feedback_df.empty:
        return (
            X_base,
            y_base,
            feature_columns,
            timestamps_base,
        )

    feedback_rows = []
    feedback_labels = []
    feedback_timestamps = []

    for _, feedback in (
        feedback_df.iterrows()
    ):

        features = (
            feedback.get(
                "parsed_features",
                {},
            )
        )

        if not isinstance(
            features,
            dict,
        ):
            continue

        try:
            transaction = json.loads(
                feedback.get(
                    "transaction_json",
                    "{}",
                )
            )
        except Exception:
            transaction = {}

        timestamp = transaction.get(
            "timestamp"
        )

        if timestamp is None:
            continue

        timestamp = pd.to_datetime(
            timestamp,
            errors="coerce",
        )

        if pd.isna(timestamp):
            continue

        row = {}
        valid = True

        for column in feature_columns:

            if column not in features:
                valid = False
                break

            row[column] = (
                features[column]
            )

        if not valid:
            continue

        feedback_rows.append(
            row
        )

        feedback_labels.append(
            int(
                feedback[
                    "label"
                ]
            )
        )

        feedback_timestamps.append(
            timestamp
        )

    if not feedback_rows:
        return (
            X_base,
            y_base,
            feature_columns,
            timestamps_base,
        )

    X_feedback = pd.DataFrame(
        feedback_rows,
        columns=feature_columns,
    )

    y_feedback = pd.Series(
        feedback_labels,
        name="is_fraud",
        dtype=int,
    )

    timestamps_feedback = pd.Series(
        feedback_timestamps,
        name="timestamp",
    )

    X_combined = pd.concat(
        [
            X_base,
            X_feedback,
        ],
        ignore_index=True,
    )

    y_combined = pd.concat(
        [
            y_base,
            y_feedback,
        ],
        ignore_index=True,
    )

    timestamps_combined = pd.concat(
        [
            timestamps_base,
            timestamps_feedback,
        ],
        ignore_index=True,
    )

    return (
        X_combined,
        y_combined,
        feature_columns,
        timestamps_combined,
    )


# ============================================================
# MODEL PIPELINE
# ============================================================

def _build_pipeline(
    X,
):
    """
    Build the continual-learning HistGradientBoosting pipeline.

    Numeric features are median-imputed.

    Categorical features are most-frequent imputed and one-hot encoded.
    """

    categorical_columns = [
        column
        for column in [
            "payment_method",
            "location",
        ]
        if column in X.columns
    ]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    classifier = (
        HistGradientBoostingClassifier(
            max_iter=250,

            learning_rate=0.08,

            max_leaf_nodes=31,

            min_samples_leaf=20,

            l2_regularization=1.0,

            random_state=MODEL_RANDOM_STATE,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def _metrics(
    model,
    X,
    y,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Calculate ROC-AUC, PR-AUC and thresholded F1.
    """

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = {}

    unique_classes = (
        np.unique(y)
    )

    if len(unique_classes) >= 2:

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y,
                probabilities,
            )
        )

        result[
            "pr_auc"
        ] = float(
            average_precision_score(
                y,
                probabilities,
            )
        )

    else:

        result[
            "roc_auc"
        ] = None

        result[
            "pr_auc"
        ] = None

    result[
        "f1"
    ] = float(
        f1_score(
            y,
            predictions,
            zero_division=0,
        )
    )

    result[
        "threshold"
    ] = float(
        threshold
    )

    return result


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def _chronological_split(
    X,
    y,
    timestamps,
):
    """
    Create an 80/20 chronological holdout.

    Transaction timestamps are used only for ordering and
    are never passed to the model.
    """

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if len(X) != len(timestamps):
        raise ValueError(
            "X and timestamps must contain the same number "
            "of rows."
        )

    timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Chronological split contains invalid timestamps."
        )

    order = (
        timestamps
        .sort_values()
        .index
    )

    X = X.loc[
        order
    ].reset_index(
        drop=True
    )

    y = y.loc[
        order
    ].reset_index(
        drop=True
    )

    split_index = int(
        len(X) * 0.80
    )

    if split_index <= 0:
        raise ValueError(
            "Not enough rows for training."
        )

    if split_index >= len(X):
        split_index = (
            len(X) - 1
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# RETRAIN LOCK
# ============================================================

def _acquire_lock():
    """
    Acquire a filesystem lock.

    Prevents two maintenance workers from retraining
    simultaneously.
    """

    _ensure_directories()

    try:

        fd = os.open(
            LOCK_FILE,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY,
        )

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    {
                        "pid":
                            os.getpid(),

                        "created_at":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
                    }
                )
            )

        return True

    except FileExistsError:

        return False


def _release_lock():

    try:

        LOCK_FILE.unlink()

    except FileNotFoundError:

        pass


# ============================================================
# MODEL VERSION
# ============================================================

def _new_model_version():

    return (
        MODEL_VERSION_PREFIX
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d%H%M%S"
        )
    )


# ============================================================
# ATOMIC MODEL SAVE
# ============================================================

def _atomic_joblib_save(
    artifact,
    destination,
):
    """
    Atomically save a joblib artifact.
    """

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".joblib",
        dir=destination.parent,
        delete=False,
    )

    temporary_path = Path(
        temporary_file.name
    )

    try:

        temporary_file.close()

        joblib.dump(
            artifact,
            temporary_path,
        )

        os.replace(
            temporary_path,
            destination,
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink(
                missing_ok=True
            )


# ============================================================
# ATOMIC JSON SAVE
# ============================================================

def _atomic_json_save(
    data,
    destination,
):
    """
    Atomically write JSON.
    """

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    )

    temporary_path = Path(
        temporary_file.name
    )

    try:

        json.dump(
            data,
            temporary_file,
            indent=2,
            default=str,
        )

        temporary_file.flush()

        os.fsync(
            temporary_file.fileno()
        )

        temporary_file.close()

        os.replace(
            temporary_path,
            destination,
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink(
                missing_ok=True
            )


# ============================================================
# ACTIVE MODEL METRICS
# ============================================================

def _evaluate_active_model(
    X_test,
    y_test,
    threshold,
):
    """
    Evaluate the currently active model on the same holdout
    used for candidate evaluation.
    """

    artifact = (
        _load_active_artifact()
    )

    active_model = (
        artifact[
            "pipeline"
        ]
    )

    return _metrics(
        active_model,
        X_test,
        y_test,
        threshold=threshold,
    )


# ============================================================
# PROMOTION GATE
# ============================================================

def _candidate_passes_gate(
    active_metrics,
    candidate_metrics,
):
    """
    Decide whether the candidate should become active.

    Candidate is rejected if BOTH PR-AUC and F1 materially
    degrade beyond the configured tolerance.

    This prevents a single noisy metric from blocking
    potentially useful model improvements.
    """

    active_pr = (
        active_metrics.get(
            "pr_auc"
        )
    )

    candidate_pr = (
        candidate_metrics.get(
            "pr_auc"
        )
    )

    active_f1 = (
        active_metrics.get(
            "f1"
        )
    )

    candidate_f1 = (
        candidate_metrics.get(
            "f1"
        )
    )

    if (
        active_pr is None
        or candidate_pr is None
    ):

        pr_degradation = False

    else:

        pr_degradation = (
            candidate_pr
            <
            active_pr
            * (
                1.0
                - PROMOTION_TOLERANCE
            )
        )

    f1_degradation = (
        candidate_f1
        <
        active_f1
        * (
            1.0
            - PROMOTION_TOLERANCE
        )
    )

    reject = (
        pr_degradation
        and f1_degradation
    )

    return not reject


# ============================================================
# MODEL REGISTRY
# ============================================================

def _write_registry_version(
    version,
    artifact,
    metadata,
):
    """
    Save a versioned copy of the promoted model.
    """

    version_dir = (
        MODEL_REGISTRY_DIR
        / version
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        version_dir
        / "fraud_model.joblib"
    )

    metadata_path = (
        version_dir
        / "metadata.json"
    )

    _atomic_joblib_save(
        artifact,
        model_path,
    )

    _atomic_json_save(
        metadata,
        metadata_path,
    )

    return version_dir


# ============================================================
# CLEAN REGISTRY
# ============================================================

def _cleanup_registry():

    if not MODEL_REGISTRY_DIR.exists():

        return

    directories = [
        directory
        for directory
        in MODEL_REGISTRY_DIR.iterdir()
        if directory.is_dir()
    ]

    directories.sort(
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    for old_directory in directories[
        REGISTRY_RETENTION:
    ]:

        shutil.rmtree(
            old_directory,
            ignore_errors=True,
        )


# ============================================================
# PROMOTE MODEL
# ============================================================

def _promote_model(
    candidate_artifact,
    version,
    metadata,
):
    """
    Atomically promote candidate to active model.

    The live predictor watches the active model metadata and
    can reload after promotion.
    """

    _ensure_directories()

    candidate_model_path = (
        REPORTS_DIR
        / f".candidate_{version}.joblib"
    )

    candidate_metadata_path = (
        REPORTS_DIR
        / f".candidate_{version}.json"
    )

    try:

        _atomic_joblib_save(
            candidate_artifact,
            candidate_model_path,
        )

        _atomic_json_save(
            metadata,
            candidate_metadata_path,
        )

        # ----------------------------------------------------
        # Main active model
        # ----------------------------------------------------

        os.replace(
            candidate_model_path,
            MODEL_FILE,
        )

        os.replace(
            candidate_metadata_path,
            MODEL_METADATA_FILE,
        )

        # ----------------------------------------------------
        # Active model pointer
        # ----------------------------------------------------

        active_model = {
            "version":
                version,

            "model_path":
                str(
                    MODEL_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "metadata_path":
                str(
                    MODEL_METADATA_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "promoted_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        _atomic_json_save(
            active_model,
            ACTIVE_MODEL_FILE,
        )

        # ----------------------------------------------------
        # Version registry
        # ----------------------------------------------------

        _write_registry_version(
            version,
            candidate_artifact,
            metadata,
        )

        _cleanup_registry()

    finally:

        candidate_model_path.unlink(
            missing_ok=True
        )

        candidate_metadata_path.unlink(
            missing_ok=True
        )


# ============================================================
# RETRAIN
# ============================================================

def retrain_if_needed(
    min_feedback=DEFAULT_MIN_FEEDBACK,
    force=False,
):
    """
    Retrain the model when enough confirmed feedback exists.

    Returns a structured result rather than raising for normal
    skip/rejection conditions.
    """

    _initialize_database()

    stats = feedback_stats()

    feedback_count = (
        stats[
            "total_feedback"
        ]
    )

    if (
        not force
        and feedback_count
        < min_feedback
    ):

        return {

            "status":
                "skipped",

            "reason":
                (
                    f"Need {min_feedback} "
                    f"feedback labels; "
                    f"have {feedback_count}."
                ),

            "feedback_count":
                feedback_count,

        }

    if not _acquire_lock():

        return {

            "status":
                "skipped",

            "reason":
                "Another retraining job is already running.",

            "feedback_count":
                feedback_count,

        }

    try:

        # ====================================================
        # LOAD DATA
        # ====================================================

        X, y, feature_columns, timestamps = (
            _prepare_training_data()
        )

        if len(X) < 20:

            return {

                "status":
                    "skipped",

                "reason":
                    "Not enough training rows.",

                "training_rows":
                    len(X),

            }

        if y.nunique() < 2:

            return {

                "status":
                    "skipped",

                "reason":
                    "Training data contains only one class.",

            }

        # ====================================================
        # CHRONOLOGICAL HOLDOUT
        # ====================================================

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = _chronological_split(
            X,
            y,
            timestamps,
        )

        if y_train.nunique() < 2:

            return {

                "status":
                    "skipped",

                "reason":
                    (
                        "Chronological training split "
                        "contains only one class."
                    ),

            }

        if y_test.nunique() < 2:

            return {

                "status":
                    "skipped",

                "reason":
                    (
                        "Chronological holdout "
                        "contains only one class."
                    ),

            }

        # ====================================================
        # BUILD CANDIDATE
        # ====================================================

        candidate = _build_pipeline(
            X_train
        )

        candidate.fit(
            X_train,
            y_train,
        )

        # ====================================================
        # METRICS
        # ====================================================

        candidate_metrics = _metrics(
            candidate,
            X_test,
            y_test,
            threshold=DEFAULT_THRESHOLD,
        )

        active_metrics = (
            _evaluate_active_model(
                X_test,
                y_test,
                DEFAULT_THRESHOLD,
            )
        )

        # ====================================================
        # PROMOTION GATE
        # ====================================================

        passes_gate = (
            _candidate_passes_gate(
                active_metrics,
                candidate_metrics,
            )
        )

        version = (
            _new_model_version()
        )

        metadata = {

            "version":
                version,

            "model_version":
                version,

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "learning_mode":
                (
                    "feedback_driven_"
                    "continual_learning"
                ),

            "training_rows":
                int(
                    len(X_train)
                ),

            "holdout_rows":
                int(
                    len(X_test)
                ),

            "feedback_count":
                int(
                    feedback_count
                ),

            "confirmed_fraud":
                int(
                    stats[
                        "confirmed_fraud"
                    ]
                ),

            "confirmed_legitimate":
                int(
                    stats[
                        "confirmed_legitimate"
                    ]
                ),

            "ai_accepted":
                int(
                    stats[
                        "ai_accepted"
                    ]
                ),

            "ai_overridden":
                int(
                    stats[
                        "ai_overridden"
                    ]
                ),

            "threshold":
                DEFAULT_THRESHOLD,

            "feature_columns":
                feature_columns,

            "candidate_metrics":
                candidate_metrics,

            "active_metrics":
                active_metrics,

            "promotion_gate":
                {
                    "passes":
                        passes_gate,

                    "tolerance":
                        PROMOTION_TOLERANCE,

                    "rule":
                        (
                            "Reject candidate only when "
                            "both PR-AUC and F1 materially "
                            "degrade."
                        ),
                },

        }

        # ====================================================
        # REJECT
        # ====================================================

        if not passes_gate:

            return {

                "status":
                    "rejected",

                "reason":
                    (
                        "Candidate failed the "
                        "promotion gate."
                    ),

                "version":
                    version,

                "feedback_count":
                    feedback_count,

                "candidate_metrics":
                    candidate_metrics,

                "active_metrics":
                    active_metrics,

            }

        # ====================================================
        # CRITICAL:
        #
        # Preserve the artifact format expected by the
        # live predictor.
        # ====================================================

        candidate_artifact = {

            "pipeline":
                candidate,

            "feature_columns":
                feature_columns,

        }

        # ====================================================
        # PROMOTE
        # ====================================================

        _promote_model(
            candidate_artifact,
            version,
            metadata,
        )

        return {

            "status":
                "promoted",

            "version":
                version,

            "feedback_count":
                feedback_count,

            "candidate_metrics":
                candidate_metrics,

            "active_metrics":
                active_metrics,

            "feature_count":
                len(
                    feature_columns
                ),

        }

    except Exception as exc:

        return {

            "status":
                "error",

            "reason":
                str(exc),

        }

    finally:

        _release_lock()


# ============================================================
# LEARNING STATUS
# ============================================================

def learning_status(
    min_feedback=DEFAULT_MIN_FEEDBACK,
):
    """
    Return current continual-learning status.
    """

    stats = feedback_stats()

    active_model = (
        _current_model_version()
    )

    return {

        **stats,

        "active_model":
            active_model,

        "feedback_database":
            str(
                FEEDBACK_DB.relative_to(
                    PROJECT_ROOT
                )
            ),

        "minimum_feedback_for_retraining":
            min_feedback,

        "ready_for_retraining":
            (
                stats[
                    "total_feedback"
                ]
                >= min_feedback
            ),

        "learning_mode":
            (
                "feedback_driven_"
                "continual_learning"
            ),

    }


# ============================================================
# MAINTENANCE STATUS
# ============================================================

def _write_maintenance_status(
    status,
):

    data = {

        "updated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        **status,

    }

    _atomic_json_save(
        data,
        MAINTENANCE_STATUS_FILE,
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Merchant Risk Sentinel "
            "continual-learning worker"
        )
    )

    parser.add_argument(
        "--min-feedback",
        type=int,
        default=DEFAULT_MIN_FEEDBACK,
        help=(
            "Minimum confirmed analyst "
            "labels required before retraining."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force retraining even when "
            "the feedback threshold has not "
            "been reached."
        ),
    )

    args = parser.parse_args()

    result = retrain_if_needed(
        min_feedback=args.min_feedback,
        force=args.force,
    )

    _write_maintenance_status(
        result
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":

    main()