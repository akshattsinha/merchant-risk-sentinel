from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "reports" / "fraud_model.joblib"
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "fraud_features.csv"


class LiveRiskPredictor:

    def __init__(self):

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model artifact not found at: {MODEL_FILE}"
            )

        if not FEATURE_FILE.exists():
            raise FileNotFoundError(
                f"Feature dataset not found at: {FEATURE_FILE}"
            )

        self.artifact = joblib.load(
            MODEL_FILE
        )

        self.pipeline = self.artifact[
            "pipeline"
        ]

        self.feature_columns = self.artifact[
            "feature_columns"
        ]

        self.history = pd.read_csv(
            FEATURE_FILE
        )

        self.history["timestamp"] = pd.to_datetime(
            self.history["timestamp"]
        )

        self.history = self.history.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    def build_features(self, transaction):

        history = self.history

        timestamp = pd.Timestamp(
            transaction["timestamp"]
        )

        customer_id = transaction["customer_id"]
        merchant_id = transaction["merchant_id"]
        device_id = transaction["device_id"]
        ip_id = transaction["ip_id"]
        address_id = transaction["address_id"]

        amount = float(
            transaction["amount"]
        )

        payment_method = transaction[
            "payment_method"
        ]

        location = transaction[
            "location"
        ]

        customer_history = history[
            history["customer_id"] == customer_id
        ]

        merchant_history = history[
            history["merchant_id"] == merchant_id
        ]

        payment_history = history[
            history["payment_method"] == payment_method
        ]

        before_customer = customer_history[
            customer_history["timestamp"] < timestamp
        ]

        before_merchant = merchant_history[
            merchant_history["timestamp"] < timestamp
        ]

        recent_customer = before_customer[
            before_customer["timestamp"]
            >= timestamp - pd.Timedelta(hours=1)
        ]

        device_customers = history[
            history["device_id"] == device_id
        ]["customer_id"].nunique()

        ip_customers = history[
            history["ip_id"] == ip_id
        ]["customer_id"].nunique()

        address_customers = history[
            history["address_id"] == address_id
        ]["customer_id"].nunique()

        if len(before_customer) > 0:
            customer_avg = before_customer[
                "amount"
            ].mean()
        else:
            customer_avg = history[
                "amount"
            ].median()

        if len(before_merchant) > 0:
            merchant_avg = before_merchant[
                "amount"
            ].mean()
        else:
            merchant_avg = history[
                "amount"
            ].median()

        payment_frequency = len(
            payment_history[
                payment_history["timestamp"] < timestamp
            ]
        )

        if (
            "is_refund" in merchant_history.columns
            and len(merchant_history) > 0
        ):
            merchant_refund_ratio = merchant_history[
                "is_refund"
            ].mean()
        else:
            merchant_refund_ratio = 0.0

        if (
            "is_chargeback" in merchant_history.columns
            and len(merchant_history) > 0
        ):
            merchant_chargeback_ratio = merchant_history[
                "is_chargeback"
            ].mean()
        else:
            merchant_chargeback_ratio = 0.0

        if "is_refund" in before_customer.columns:

            refund_count = before_customer[
                "is_refund"
            ].sum()

            refund_amount = before_customer.loc[
                before_customer["is_refund"] == 1,
                "amount"
            ].sum()

        else:

            refund_count = 0
            refund_amount = 0.0

        if "is_chargeback" in before_customer.columns:

            chargeback_count = before_customer[
                "is_chargeback"
            ].sum()

            chargeback_amount = before_customer.loc[
                before_customer["is_chargeback"] == 1,
                "amount"
            ].sum()

        else:

            chargeback_count = 0
            chargeback_amount = 0.0

        previous_transactions = (
            before_customer
            .sort_values("timestamp")
        )

        if len(previous_transactions) > 0:

            previous = previous_transactions.iloc[-1]

            seconds_since = (
                timestamp - previous["timestamp"]
            ).total_seconds()

            location_changed = int(
                previous["location"] != location
            )

            device_changed = int(
                previous["device_id"] != device_id
            )

            ip_changed = int(
                previous["ip_id"] != ip_id
            )

        else:

            seconds_since = 999999

            location_changed = 1
            device_changed = 1
            ip_changed = 1

        transaction_count = len(
            recent_customer
        )

        high_velocity = int(
            transaction_count >= 5
        )

        very_high_velocity = int(
            transaction_count >= 10
        )

        account_age = int(
            transaction["account_age_days"]
        )

        is_new_account = int(
            account_age <= 30
        )

        is_very_new_account = int(
            account_age <= 7
        )

        customer_count = len(
            before_customer
        )

        merchant_count = len(
            before_merchant
        )

        customer_total_amount = (
            before_customer["amount"].sum()
            if customer_count > 0
            else 0.0
        )

        refund_ratio = (
            refund_count / customer_count
            if customer_count > 0
            else 0.0
        )

        chargeback_ratio = (
            chargeback_count / customer_count
            if customer_count > 0
            else 0.0
        )

        refund_amount_ratio = (
            refund_amount / customer_total_amount
            if customer_total_amount > 0
            else 0.0
        )

        chargeback_amount_ratio = (
            chargeback_amount / customer_total_amount
            if customer_total_amount > 0
            else 0.0
        )

        amount_vs_customer_average = (
            amount / customer_avg
            if customer_avg > 0
            else 1.0
        )

        amount_vs_merchant_average = (
            amount / merchant_avg
            if merchant_avg > 0
            else 1.0
        )

        behavioral_signals = sum(
            [
                int(
                    amount_vs_customer_average > 3
                ),
                int(
                    amount_vs_merchant_average > 3
                ),
                is_new_account,
                is_very_new_account,
                high_velocity,
                very_high_velocity,
                int(device_customers >= 3),
                int(ip_customers >= 3),
                int(address_customers >= 3),
                location_changed,
                device_changed,
                ip_changed,
                int(
                    merchant_refund_ratio > 0.05
                ),
                int(
                    merchant_chargeback_ratio > 0.02
                )
            ]
        )

        row = {
            "amount": amount,
            "payment_method": payment_method,
            "account_age_days": account_age,
            "location": location,
            "refund_count": refund_count,
            "refund_amount": refund_amount,
            "chargeback_count": chargeback_count,
            "chargeback_amount": chargeback_amount,
            "is_refund": 0,
            "is_chargeback": 0,
            "transaction_count_last_hour": transaction_count,
            "device_customer_count": device_customers,
            "ip_customer_count": ip_customers,
            "address_customer_count": address_customers,
            "payment_method_frequency": payment_frequency,
            "merchant_average_amount": merchant_avg,
            "amount_to_merchant_average": (
                amount / merchant_avg
                if merchant_avg > 0
                else 1.0
            ),
            "merchant_refund_ratio": merchant_refund_ratio,
            "merchant_chargeback_ratio": merchant_chargeback_ratio,
            "log_amount": np.log1p(amount),
            "customer_transaction_count_before": customer_count,
            "customer_avg_amount_before": customer_avg,
            "amount_vs_customer_average": amount_vs_customer_average,
            "merchant_transaction_count_before": merchant_count,
            "merchant_avg_amount_before": merchant_avg,
            "amount_vs_merchant_average": amount_vs_merchant_average,
            "payment_method_count_before": payment_frequency,
            "is_new_account": is_new_account,
            "is_very_new_account": is_very_new_account,
            "refund_to_transaction_ratio": refund_ratio,
            "chargeback_to_transaction_ratio": chargeback_ratio,
            "refund_amount_ratio": refund_amount_ratio,
            "chargeback_amount_ratio": chargeback_amount_ratio,
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "is_weekend": int(
                timestamp.dayofweek >= 5
            ),
            "seconds_since_customer_transaction": seconds_since,
            "high_velocity_flag": high_velocity,
            "very_high_velocity_flag": very_high_velocity,
            "location_changed": location_changed,
            "device_changed": device_changed,
            "ip_changed": ip_changed,
            "behavioral_risk_signal_count": behavioral_signals
        }

        features = pd.DataFrame(
            [row]
        )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in features.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_columns)
            )

        features = features[
            self.feature_columns
        ]

        return features

    def predict(self, transaction):

        features = self.build_features(
            transaction
        )

        probability = float(
            self.pipeline.predict_proba(
                features
            )[0][1]
        )

        risk_score = round(
            probability * 100
        )

        if probability >= 0.75:

            risk_level = "CRITICAL"

        elif probability >= 0.40:

            risk_level = "HIGH"

        elif probability >= 0.10:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        if probability >= 0.05:

            action = "HOLD_AND_INVESTIGATE"

        elif probability >= 0.02:

            action = "MONITOR"

        else:

            action = "ALLOW"

        return {
            "fraud_probability": probability,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "features": features.iloc[0].to_dict()
        }