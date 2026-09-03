from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "fraud_model.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud_features.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "reports"
    / "fraud_model_metadata.json"
)


# ============================================================
# LIVE RISK PREDICTOR
# ============================================================

class LiveRiskPredictor:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        # ----------------------------------------------------
        # MODEL FILE CHECK
        # ----------------------------------------------------

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                f"Model artifact not found at: "
                f"{MODEL_FILE}"
            )


        # ----------------------------------------------------
        # FEATURE DATASET CHECK
        # ----------------------------------------------------

        if not FEATURE_FILE.exists():

            raise FileNotFoundError(
                f"Feature dataset not found at: "
                f"{FEATURE_FILE}"
            )


        # ----------------------------------------------------
        # LOAD MODEL ARTIFACT
        # ----------------------------------------------------

        self.artifact = joblib.load(
            MODEL_FILE
        )


        # ----------------------------------------------------
        # LOAD PIPELINE
        # ----------------------------------------------------

        self.pipeline = self.artifact[
            "pipeline"
        ]


        # ----------------------------------------------------
        # LOAD FEATURE COLUMNS
        # ----------------------------------------------------

        self.feature_columns = self.artifact[
            "feature_columns"
        ]


        # ----------------------------------------------------
        # LOAD HISTORICAL TRANSACTION DATA
        # ----------------------------------------------------

        self.history = pd.read_csv(
            FEATURE_FILE
        )


        # ----------------------------------------------------
        # CONVERT TIMESTAMP
        # ----------------------------------------------------

        self.history["timestamp"] = (
            pd.to_datetime(
                self.history["timestamp"]
            )
        )


        # ----------------------------------------------------
        # SORT HISTORY CHRONOLOGICALLY
        # ----------------------------------------------------

        self.history = (
            self.history
            .sort_values("timestamp")
            .reset_index(drop=True)
        )


        # ----------------------------------------------------
        # STORE MODEL METADATA TIMESTAMP
        # ----------------------------------------------------
        #
        # The continual-learning system replaces the model
        # artifact and metadata when a new model is promoted.
        #
        # We use the metadata modification time as a lightweight
        # signal that the active model has changed.
        #
        # ----------------------------------------------------

        self._metadata_mtime = (
            METADATA_FILE.stat().st_mtime
            if METADATA_FILE.exists()
            else None
        )


        # ----------------------------------------------------
        # LOAD MODEL VERSION
        # ----------------------------------------------------

        self.model_version = (
            self._load_model_version()
        )


    # ========================================================
    # MODEL VERSION
    # ========================================================

    def _load_model_version(self):

        """
        Read the currently active model version
        from fraud_model_metadata.json.

        Falls back to 'unknown' if metadata is unavailable.
        """

        if not METADATA_FILE.exists():

            return "unknown"


        try:

            import json

            metadata = json.loads(
                METADATA_FILE.read_text()
            )

            return str(
                metadata.get(
                    "model_version",
                    "unknown",
                )
            )

        except Exception:

            return "unknown"


    # ========================================================
    # CHECK FOR MODEL CHANGES
    # ========================================================

    def reload_if_changed(
        self,
        force=False,
    ):

        """
        Detect whether continual learning has promoted
        a new model.

        If the metadata file changed, reload the entire
        predictor.

        Returns:

            True  -> model was reloaded
            False -> model was unchanged
        """

        current_mtime = (
            METADATA_FILE.stat().st_mtime
            if METADATA_FILE.exists()
            else None
        )


        previous_mtime = getattr(
            self,
            "_metadata_mtime",
            None,
        )


        # ----------------------------------------------------
        # MODEL CHANGED
        # ----------------------------------------------------

        if (
            force
            or current_mtime != previous_mtime
        ):

            self.__init__()

            return True


        return False


    # ========================================================
    # BUILD BEHAVIORAL FEATURES
    # ========================================================

    def build_features(
        self,
        transaction,
    ):

        history = self.history


        # ----------------------------------------------------
        # TRANSACTION TIMESTAMP
        # ----------------------------------------------------

        timestamp = pd.Timestamp(
            transaction["timestamp"]
        )


        # ----------------------------------------------------
        # TRANSACTION IDENTIFIERS
        # ----------------------------------------------------

        customer_id = (
            transaction["customer_id"]
        )

        merchant_id = (
            transaction["merchant_id"]
        )

        device_id = (
            transaction["device_id"]
        )

        ip_id = (
            transaction["ip_id"]
        )

        address_id = (
            transaction["address_id"]
        )


        # ----------------------------------------------------
        # BASIC TRANSACTION INFORMATION
        # ----------------------------------------------------

        amount = float(
            transaction["amount"]
        )

        payment_method = (
            transaction["payment_method"]
        )

        location = (
            transaction["location"]
        )


        # ----------------------------------------------------
        # CUSTOMER HISTORY
        # ----------------------------------------------------

        customer_history = history[
            history["customer_id"]
            == customer_id
        ]


        # ----------------------------------------------------
        # MERCHANT HISTORY
        # ----------------------------------------------------

        merchant_history = history[
            history["merchant_id"]
            == merchant_id
        ]


        # ----------------------------------------------------
        # PAYMENT METHOD HISTORY
        # ----------------------------------------------------

        payment_history = history[
            history["payment_method"]
            == payment_method
        ]


        # ----------------------------------------------------
        # ONLY USE INFORMATION BEFORE TRANSACTION
        # ----------------------------------------------------

        before_customer = (
            customer_history[
                customer_history["timestamp"]
                < timestamp
            ]
        )


        before_merchant = (
            merchant_history[
                merchant_history["timestamp"]
                < timestamp
            ]
        )


        # ----------------------------------------------------
        # CUSTOMER VELOCITY
        # ----------------------------------------------------

        recent_customer = (
            before_customer[
                before_customer["timestamp"]
                >= timestamp
                - pd.Timedelta(hours=1)
            ]
        )


        # ----------------------------------------------------
        # DEVICE SHARING
        # ----------------------------------------------------

        device_customers = (
            history[
                history["device_id"]
                == device_id
            ]["customer_id"]
            .nunique()
        )


        # ----------------------------------------------------
        # IP SHARING
        # ----------------------------------------------------

        ip_customers = (
            history[
                history["ip_id"]
                == ip_id
            ]["customer_id"]
            .nunique()
        )


        # ----------------------------------------------------
        # ADDRESS SHARING
        # ----------------------------------------------------

        address_customers = (
            history[
                history["address_id"]
                == address_id
            ]["customer_id"]
            .nunique()
        )


        # ----------------------------------------------------
        # CUSTOMER AVERAGE
        # ----------------------------------------------------

        if len(before_customer) > 0:

            customer_avg = (
                before_customer["amount"]
                .mean()
            )

        else:

            customer_avg = (
                history["amount"]
                .median()
            )


        # ----------------------------------------------------
        # MERCHANT AVERAGE
        # ----------------------------------------------------

        if len(before_merchant) > 0:

            merchant_avg = (
                before_merchant["amount"]
                .mean()
            )

        else:

            merchant_avg = (
                history["amount"]
                .median()
            )


        # ----------------------------------------------------
        # PAYMENT METHOD FREQUENCY
        # ----------------------------------------------------

        payment_frequency = len(
            payment_history[
                payment_history["timestamp"]
                < timestamp
            ]
        )


        # ----------------------------------------------------
        # MERCHANT REFUND RATIO
        # ----------------------------------------------------

        if (
            "is_refund"
            in merchant_history.columns
            and len(merchant_history) > 0
        ):

            merchant_refund_ratio = (
                merchant_history[
                    "is_refund"
                ].mean()
            )

        else:

            merchant_refund_ratio = 0.0


        # ----------------------------------------------------
        # MERCHANT CHARGEBACK RATIO
        # ----------------------------------------------------

        if (
            "is_chargeback"
            in merchant_history.columns
            and len(merchant_history) > 0
        ):

            merchant_chargeback_ratio = (
                merchant_history[
                    "is_chargeback"
                ].mean()
            )

        else:

            merchant_chargeback_ratio = 0.0


        # ----------------------------------------------------
        # REFUND / CHARGEBACK HISTORY
        # ----------------------------------------------------

        if "is_refund" in before_customer.columns:

            refund_count = (
                before_customer[
                    "is_refund"
                ].sum()
            )

            refund_amount = (
                before_customer.loc[
                    before_customer[
                        "is_refund"
                    ] == 1,
                    "amount",
                ].sum()
            )

        else:

            refund_count = 0
            refund_amount = 0.0


        if "is_chargeback" in before_customer.columns:

            chargeback_count = (
                before_customer[
                    "is_chargeback"
                ].sum()
            )

            chargeback_amount = (
                before_customer.loc[
                    before_customer[
                        "is_chargeback"
                    ] == 1,
                    "amount",
                ].sum()
            )

        else:

            chargeback_count = 0
            chargeback_amount = 0.0


        # ----------------------------------------------------
        # PREVIOUS CUSTOMER TRANSACTION
        # ----------------------------------------------------

        previous_transactions = (
            before_customer
            .sort_values("timestamp")
        )


        if len(previous_transactions) > 0:

            previous = (
                previous_transactions
                .iloc[-1]
            )


            seconds_since = (
                timestamp
                - previous["timestamp"]
            ).total_seconds()


            location_changed = int(
                previous["location"]
                != location
            )


            device_changed = int(
                previous["device_id"]
                != device_id
            )


            ip_changed = int(
                previous["ip_id"]
                != ip_id
            )

        else:

            seconds_since = 999999

            location_changed = 1

            device_changed = 1

            ip_changed = 1


        # ----------------------------------------------------
        # VELOCITY
        # ----------------------------------------------------

        transaction_count = len(
            recent_customer
        )


        high_velocity = int(
            transaction_count >= 5
        )


        very_high_velocity = int(
            transaction_count >= 10
        )


        # ----------------------------------------------------
        # ACCOUNT AGE
        # ----------------------------------------------------

        account_age = int(
            transaction[
                "account_age_days"
            ]
        )


        is_new_account = int(
            account_age <= 30
        )


        is_very_new_account = int(
            account_age <= 7
        )


        # ----------------------------------------------------
        # CUSTOMER TRANSACTION COUNT
        # ----------------------------------------------------

        customer_count = len(
            before_customer
        )


        # ----------------------------------------------------
        # MERCHANT TRANSACTION COUNT
        # ----------------------------------------------------

        merchant_count = len(
            before_merchant
        )


        # ----------------------------------------------------
        # CUSTOMER TOTAL AMOUNT
        # ----------------------------------------------------

        customer_total_amount = (

            before_customer["amount"].sum()

            if customer_count > 0

            else 0.0
        )


        # ----------------------------------------------------
        # REFUND RATIO
        # ----------------------------------------------------

        refund_ratio = (

            refund_count
            / customer_count

            if customer_count > 0

            else 0.0
        )


        # ----------------------------------------------------
        # CHARGEBACK RATIO
        # ----------------------------------------------------

        chargeback_ratio = (

            chargeback_count
            / customer_count

            if customer_count > 0

            else 0.0
        )


        # ----------------------------------------------------
        # REFUND AMOUNT RATIO
        # ----------------------------------------------------

        refund_amount_ratio = (

            refund_amount
            / customer_total_amount

            if customer_total_amount > 0

            else 0.0
        )


        # ----------------------------------------------------
        # CHARGEBACK AMOUNT RATIO
        # ----------------------------------------------------

        chargeback_amount_ratio = (

            chargeback_amount
            / customer_total_amount

            if customer_total_amount > 0

            else 0.0
        )


        # ----------------------------------------------------
        # AMOUNT VS CUSTOMER AVERAGE
        # ----------------------------------------------------

        amount_vs_customer_average = (

            amount / customer_avg

            if customer_avg > 0

            else 1.0
        )


        # ----------------------------------------------------
        # AMOUNT VS MERCHANT AVERAGE
        # ----------------------------------------------------

        amount_vs_merchant_average = (

            amount / merchant_avg

            if merchant_avg > 0

            else 1.0
        )


        # ----------------------------------------------------
        # BEHAVIORAL RISK SIGNAL COUNT
        # ----------------------------------------------------

        behavioral_signals = sum(
            [
                int(
                    amount_vs_customer_average
                    > 3
                ),

                int(
                    amount_vs_merchant_average
                    > 3
                ),

                is_new_account,

                is_very_new_account,

                high_velocity,

                very_high_velocity,

                int(
                    device_customers >= 3
                ),

                int(
                    ip_customers >= 3
                ),

                int(
                    address_customers >= 3
                ),

                location_changed,

                device_changed,

                ip_changed,

                int(
                    merchant_refund_ratio
                    > 0.05
                ),

                int(
                    merchant_chargeback_ratio
                    > 0.02
                ),
            ]
        )


        # ====================================================
        # FINAL FEATURE ROW
        # ====================================================

        row = {

            "amount":
                amount,

            "payment_method":
                payment_method,

            "account_age_days":
                account_age,

            "location":
                location,

            "refund_count":
                refund_count,

            "refund_amount":
                refund_amount,

            "chargeback_count":
                chargeback_count,

            "chargeback_amount":
                chargeback_amount,

            "is_refund":
                0,

            "is_chargeback":
                0,

            "transaction_count_last_hour":
                transaction_count,

            "device_customer_count":
                device_customers,

            "ip_customer_count":
                ip_customers,

            "address_customer_count":
                address_customers,

            "payment_method_frequency":
                payment_frequency,

            "merchant_average_amount":
                merchant_avg,

            "amount_to_merchant_average":
                (
                    amount / merchant_avg
                    if merchant_avg > 0
                    else 1.0
                ),

            "merchant_refund_ratio":
                merchant_refund_ratio,

            "merchant_chargeback_ratio":
                merchant_chargeback_ratio,

            "log_amount":
                np.log1p(amount),

            "customer_transaction_count_before":
                customer_count,

            "customer_avg_amount_before":
                customer_avg,

            "amount_vs_customer_average":
                amount_vs_customer_average,

            "merchant_transaction_count_before":
                merchant_count,

            "merchant_avg_amount_before":
                merchant_avg,

            "amount_vs_merchant_average":
                amount_vs_merchant_average,

            "payment_method_count_before":
                payment_frequency,

            "is_new_account":
                is_new_account,

            "is_very_new_account":
                is_very_new_account,

            "refund_to_transaction_ratio":
                refund_ratio,

            "chargeback_to_transaction_ratio":
                chargeback_ratio,

            "refund_amount_ratio":
                refund_amount_ratio,

            "chargeback_amount_ratio":
                chargeback_amount_ratio,

            "hour":
                timestamp.hour,

            "day_of_week":
                timestamp.dayofweek,

            "is_weekend":
                int(
                    timestamp.dayofweek >= 5
                ),

            "seconds_since_customer_transaction":
                seconds_since,

            "high_velocity_flag":
                high_velocity,

            "very_high_velocity_flag":
                very_high_velocity,

            "location_changed":
                location_changed,

            "device_changed":
                device_changed,

            "ip_changed":
                ip_changed,

            "behavioral_risk_signal_count":
                behavioral_signals,
        }


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        features = pd.DataFrame(
            [row]
        )


        # ----------------------------------------------------
        # CHECK REQUIRED FEATURES
        # ----------------------------------------------------

        missing_columns = [

            column

            for column
            in self.feature_columns

            if column
            not in features.columns
        ]


        if missing_columns:

            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    missing_columns
                )
            )


        # ----------------------------------------------------
        # EXACT MODEL FEATURE ORDER
        # ----------------------------------------------------

        features = features[
            self.feature_columns
        ]


        return features


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        transaction,
    ):

        # ----------------------------------------------------
        # CHECK WHETHER A NEW MODEL WAS PROMOTED
        # ----------------------------------------------------

        self.reload_if_changed()


        # ----------------------------------------------------
        # BUILD FEATURES
        # ----------------------------------------------------

        features = self.build_features(
            transaction
        )


        # ----------------------------------------------------
        # FRAUD PROBABILITY
        # ----------------------------------------------------

        probability = float(
            self.pipeline
            .predict_proba(
                features
            )[0][1]
        )


        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        risk_score = round(
            probability * 100
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability >= 0.75:

            risk_level = "CRITICAL"

        elif probability >= 0.40:

            risk_level = "HIGH"

        elif probability >= 0.10:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ----------------------------------------------------
        # RECOMMENDED ACTION
        # ----------------------------------------------------

        if probability >= 0.05:

            action = (
                "HOLD_AND_INVESTIGATE"
            )

        elif probability >= 0.02:

            action = "MONITOR"

        else:

            action = "ALLOW"


        # ====================================================
        # RESULT
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

            "model_version":
                self.model_version,

            "features":
                features.iloc[
                    0
                ].to_dict(),
        }