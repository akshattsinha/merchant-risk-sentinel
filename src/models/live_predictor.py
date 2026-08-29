import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    BASE_DIR
    / "reports"
    / "fraud_model.joblib"
)

METADATA_FILE = (
    BASE_DIR
    / "reports"
    / "fraud_model_metadata.json"
)


# ============================================================
# LIVE RISK PREDICTOR
# ============================================================

class LiveRiskPredictor:

    def __init__(self):

        # ----------------------------------------------------
        # Load model artifact
        # ----------------------------------------------------

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                f"Model artifact not found at: {MODEL_FILE}"
            )

        artifact = joblib.load(
            MODEL_FILE
        )

        self.pipeline = artifact["pipeline"]

        self.feature_columns = artifact[
            "feature_columns"
        ]

        self.numeric_features = artifact.get(
            "numeric_features",
            []
        )

        self.categorical_features = artifact.get(
            "categorical_features",
            []
        )

        # ----------------------------------------------------
        # Load model metadata
        # ----------------------------------------------------

        if not METADATA_FILE.exists():

            raise FileNotFoundError(
                f"Model metadata not found at: {METADATA_FILE}"
            )

        with open(
            METADATA_FILE,
            "r"
        ) as file:

            metadata = json.load(file)

        self.metadata = metadata

        # ----------------------------------------------------
        # Cost-sensitive operating threshold
        # ----------------------------------------------------

        self.operating_threshold = float(
            metadata.get(
                "threshold",
                0.30
            )
        )


    # ========================================================
    # BUILD LIVE FEATURES
    # ========================================================

    def build_features(
        self,
        transaction
    ):

        # ----------------------------------------------------
        # Convert transaction to DataFrame
        # ----------------------------------------------------

        if isinstance(
            transaction,
            pd.DataFrame
        ):

            df = transaction.copy()

        else:

            df = pd.DataFrame(
                [transaction]
            )


        # ====================================================
        # REQUIRED INPUT COLUMNS
        # ====================================================

        required_columns = [
            "amount",
            "customer_id",
            "merchant_id",
            "timestamp",
            "payment_method",
            "device_id",
            "ip_id",
            "address_id",
            "account_age_days",
            "location"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required transaction columns: "
                + ", ".join(
                    missing_columns
                )
            )


        # ====================================================
        # BASIC TYPES
        # ====================================================

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        if df["timestamp"].isna().any():

            raise ValueError(
                "Transaction timestamp must be valid."
            )


        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce"
        )

        if df["amount"].isna().any():

            raise ValueError(
                "Transaction amount must be numeric."
            )


        df["account_age_days"] = pd.to_numeric(
            df["account_age_days"],
            errors="coerce"
        )

        df["account_age_days"] = (
            df["account_age_days"]
            .fillna(0)
        )


        # ====================================================
        # BASIC TRANSACTION FEATURE
        # ====================================================

        df["log_amount"] = np.log1p(
            df["amount"]
        )


        # ====================================================
        # CUSTOMER HISTORICAL FEATURES
        # ====================================================

        if (
            "customer_transaction_count_before"
            not in df.columns
        ):

            df[
                "customer_transaction_count_before"
            ] = 0


        if (
            "customer_avg_amount_before"
            not in df.columns
        ):

            df[
                "customer_avg_amount_before"
            ] = df["amount"]


        if (
            "amount_vs_customer_average"
            not in df.columns
        ):

            df[
                "amount_vs_customer_average"
            ] = (
                df["amount"]
                / (
                    df[
                        "customer_avg_amount_before"
                    ]
                    + 1
                )
            )


        # ====================================================
        # MERCHANT HISTORICAL FEATURES
        # ====================================================

        if (
            "merchant_transaction_count_before"
            not in df.columns
        ):

            df[
                "merchant_transaction_count_before"
            ] = 0


        if (
            "merchant_avg_amount_before"
            not in df.columns
        ):

            df[
                "merchant_avg_amount_before"
            ] = df["amount"]


        if (
            "amount_vs_merchant_average"
            not in df.columns
        ):

            df[
                "amount_vs_merchant_average"
            ] = (
                df["amount"]
                / (
                    df[
                        "merchant_avg_amount_before"
                    ]
                    + 1
                )
            )


        # ====================================================
        # ADDITIONAL MERCHANT FEATURES
        # ====================================================

        #
        # These features are present in the trained model
        # artifact but are not available from a single live
        # transaction.
        #
        # Neutral defaults are therefore used unless the
        # caller supplies historical values explicitly.
        #

        if (
            "merchant_average_amount"
            not in df.columns
        ):

            df[
                "merchant_average_amount"
            ] = df["amount"]


        if (
            "amount_to_merchant_average"
            not in df.columns
        ):

            df[
                "amount_to_merchant_average"
            ] = (
                df["amount"]
                / (
                    df[
                        "merchant_average_amount"
                    ]
                    + 1
                )
            )


        if (
            "merchant_refund_ratio"
            not in df.columns
        ):

            df[
                "merchant_refund_ratio"
            ] = 0.0


        if (
            "merchant_chargeback_ratio"
            not in df.columns
        ):

            df[
                "merchant_chargeback_ratio"
            ] = 0.0


        # ====================================================
        # PAYMENT METHOD FEATURES
        # ====================================================

        if (
            "payment_method_count_before"
            not in df.columns
        ):

            df[
                "payment_method_count_before"
            ] = 0


        if (
            "payment_method_frequency"
            not in df.columns
        ):

            df[
                "payment_method_frequency"
            ] = 1.0


        # ====================================================
        # DEVICE / IP / ADDRESS FEATURES
        # ====================================================

        if (
            "device_customer_count"
            not in df.columns
        ):

            df[
                "device_customer_count"
            ] = 1


        if (
            "ip_customer_count"
            not in df.columns
        ):

            df[
                "ip_customer_count"
            ] = 1


        if (
            "address_customer_count"
            not in df.columns
        ):

            df[
                "address_customer_count"
            ] = 1


        # ====================================================
        # ACCOUNT AGE FEATURES
        # ====================================================

        if (
            "is_new_account"
            not in df.columns
        ):

            df[
                "is_new_account"
            ] = (
                df["account_age_days"] <= 14
            ).astype(int)


        if (
            "is_very_new_account"
            not in df.columns
        ):

            df[
                "is_very_new_account"
            ] = (
                df["account_age_days"] <= 7
            ).astype(int)


        # ====================================================
        # REFUND / CHARGEBACK INPUTS
        # ====================================================

        if "refund_count" not in df.columns:

            df["refund_count"] = 0


        if "refund_amount" not in df.columns:

            df["refund_amount"] = 0.0


        if "chargeback_count" not in df.columns:

            df["chargeback_count"] = 0


        if "chargeback_amount" not in df.columns:

            df["chargeback_amount"] = 0.0


        if "is_refund" not in df.columns:

            df["is_refund"] = 0


        if "is_chargeback" not in df.columns:

            df["is_chargeback"] = 0


        # ====================================================
        # CUSTOMER REFUND / CHARGEBACK FEATURES
        # ====================================================

        if (
            "refund_to_transaction_ratio"
            not in df.columns
        ):

            df[
                "refund_to_transaction_ratio"
            ] = (
                df["refund_count"]
                / (
                    df[
                        "customer_transaction_count_before"
                    ]
                    + 1
                )
            )


        if (
            "chargeback_to_transaction_ratio"
            not in df.columns
        ):

            df[
                "chargeback_to_transaction_ratio"
            ] = (
                df["chargeback_count"]
                / (
                    df[
                        "customer_transaction_count_before"
                    ]
                    + 1
                )
            )


        if (
            "refund_amount_ratio"
            not in df.columns
        ):

            df[
                "refund_amount_ratio"
            ] = (
                df["refund_amount"]
                / (
                    df["amount"]
                    + 1
                )
            )


        if (
            "chargeback_amount_ratio"
            not in df.columns
        ):

            df[
                "chargeback_amount_ratio"
            ] = (
                df["chargeback_amount"]
                / (
                    df["amount"]
                    + 1
                )
            )


        # ====================================================
        # TEMPORAL FEATURES
        # ====================================================

        df["hour"] = (
            df["timestamp"].dt.hour
        )

        df["day_of_week"] = (
            df["timestamp"].dt.dayofweek
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)


        # ====================================================
        # TIME SINCE CUSTOMER TRANSACTION
        # ====================================================

        #
        # A single live transaction does not contain
        # historical customer transactions.
        #
        # 999999 represents "no previous transaction
        # available".
        #

        if (
            "seconds_since_customer_transaction"
            not in df.columns
        ):

            df[
                "seconds_since_customer_transaction"
            ] = 999999


        # ====================================================
        # VELOCITY FEATURES
        # ====================================================

        if (
            "transaction_count_last_hour"
            not in df.columns
        ):

            df[
                "transaction_count_last_hour"
            ] = 0


        if (
            "high_velocity_flag"
            not in df.columns
        ):

            df[
                "high_velocity_flag"
            ] = (
                df[
                    "seconds_since_customer_transaction"
                ] < 300
            ).astype(int)


        if (
            "very_high_velocity_flag"
            not in df.columns
        ):

            df[
                "very_high_velocity_flag"
            ] = (
                df[
                    "seconds_since_customer_transaction"
                ] < 60
            ).astype(int)


        # ====================================================
        # GEOGRAPHIC CHANGE
        # ====================================================

        if (
            "location_changed"
            not in df.columns
        ):

            df[
                "location_changed"
            ] = 0


        # ====================================================
        # DEVICE CHANGE
        # ====================================================

        if (
            "device_changed"
            not in df.columns
        ):

            df[
                "device_changed"
            ] = 0


        # ====================================================
        # IP CHANGE
        # ====================================================

        if (
            "ip_changed"
            not in df.columns
        ):

            df[
                "ip_changed"
            ] = 0


        # ====================================================
        # BEHAVIORAL RISK SIGNAL COUNT
        # ====================================================

        risk_flags = [
            "is_new_account",
            "is_very_new_account",
            "high_velocity_flag",
            "very_high_velocity_flag",
            "location_changed",
            "device_changed",
            "ip_changed"
        ]

        df[
            "behavioral_risk_signal_count"
        ] = (
            df[risk_flags]
            .sum(axis=1)
        )


        # ====================================================
        # INCIDENT / LABEL COLUMNS
        # ====================================================

        if "is_fraud" not in df.columns:

            df["is_fraud"] = 0


        if "fraud_type" not in df.columns:

            df["fraud_type"] = "normal"


        if "incident_id" not in df.columns:

            df["incident_id"] = ""


        if "incident_type" not in df.columns:

            df["incident_type"] = ""


        if (
            "incident_severity"
            not in df.columns
        ):

            df["incident_severity"] = ""


        # ====================================================
        # NUMERIC TYPE CLEANUP
        # ====================================================

        for column in self.numeric_features:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )


        # ====================================================
        # MODEL FEATURE VALIDATION
        # ====================================================

        missing_features = [
            column
            for column in self.feature_columns
            if column not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Unable to construct model features. "
                "Missing columns: "
                + ", ".join(
                    missing_features
                )
            )


        # ====================================================
        # SELECT EXACT MODEL FEATURES
        # ====================================================

        features = df[
            self.feature_columns
        ].copy()


        return features


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        transaction
    ):

        # ----------------------------------------------------
        # Build features
        # ----------------------------------------------------

        features = self.build_features(
            transaction
        )


        # ----------------------------------------------------
        # Model probability
        # ----------------------------------------------------

        probability = float(
            self.pipeline.predict_proba(
                features
            )[0][1]
        )


        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )


        # ====================================================
        # RISK SCORE
        # ====================================================

        risk_score = round(
            probability * 100
        )


        # ====================================================
        # RISK SEVERITY
        # ====================================================

        if probability >= 0.75:

            risk_level = "CRITICAL"

        elif probability >= 0.40:

            risk_level = "HIGH"

        elif probability >= 0.10:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ====================================================
        # OPERATIONAL ACTION
        # ====================================================

        #
        # The operating threshold is loaded from:
        #
        # reports/fraud_model_metadata.json
        #
        # Current optimized threshold:
        #
        # 0.30
        #

        if probability >= 0.75:

            action = (
                "HOLD_AND_INVESTIGATE"
            )

        elif (
            probability
            >= self.operating_threshold
        ):

            action = (
                "STEP_UP_VERIFICATION"
            )

        elif (
            probability
            >= self.operating_threshold * 0.5
        ):

            action = "MONITOR"

        else:

            action = "ALLOW"


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "fraud_probability":
                probability,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "recommended_action":
                action,

            "operating_threshold":
                self.operating_threshold,

            "features":
                features.iloc[
                    0
                ].to_dict()
        }