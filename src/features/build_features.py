import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/raw/transactions.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "fraud_features.csv"
)


def build_features(df):
    df = df.copy()

    # -----------------------------------------
    # 1. SORT BY TIME
    # -----------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # -----------------------------------------
    # 2. BASIC TRANSACTION FEATURES
    # -----------------------------------------

    df["log_amount"] = np.log1p(
        df["amount"]
    )

    # -----------------------------------------
    # 3. CUSTOMER HISTORICAL FEATURES
    # -----------------------------------------
    #
    # shift(1) ensures that the current
    # transaction is NOT included.
    #

    df["customer_transaction_count_before"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    df["customer_avg_amount_before"] = (
        df.groupby("customer_id")["amount"]
        .transform(
            lambda s: s.shift(1).expanding().mean()
        )
    )

    df["customer_avg_amount_before"] = (
        df["customer_avg_amount_before"]
        .fillna(
            df["amount"].median()
        )
    )

    df["amount_vs_customer_average"] = (
        df["amount"]
        / (
            df["customer_avg_amount_before"]
            + 1
        )
    )

    # -----------------------------------------
    # 4. MERCHANT HISTORICAL FEATURES
    # -----------------------------------------

    df["merchant_transaction_count_before"] = (
        df.groupby("merchant_id")
        .cumcount()
    )

    df["merchant_avg_amount_before"] = (
        df.groupby("merchant_id")["amount"]
        .transform(
            lambda s: s.shift(1).expanding().mean()
        )
    )

    df["merchant_avg_amount_before"] = (
        df["merchant_avg_amount_before"]
        .fillna(
            df["amount"].median()
        )
    )

    df["amount_vs_merchant_average"] = (
        df["amount"]
        / (
            df["merchant_avg_amount_before"]
            + 1
        )
    )

    # -----------------------------------------
    # 5. PAYMENT METHOD HISTORY
    # -----------------------------------------

    df["payment_method_count_before"] = (
        df.groupby(
            [
                "merchant_id",
                "payment_method"
            ]
        ).cumcount()
    )

    # -----------------------------------------
    # 6. DEVICE / IP / ADDRESS FEATURES
    # -----------------------------------------

    # Number of customers associated with
    # the device.
    #
    # These are calculated from the dataset
    # as an identity/network signal.
    #

    df["device_customer_count"] = (
        df.groupby("device_id")
        ["customer_id"]
        .transform("nunique")
    )

    df["ip_customer_count"] = (
        df.groupby("ip_id")
        ["customer_id"]
        .transform("nunique")
    )

    df["address_customer_count"] = (
        df.groupby("address_id")
        ["customer_id"]
        .transform("nunique")
    )

    # -----------------------------------------
    # 7. ACCOUNT AGE FEATURES
    # -----------------------------------------

    df["is_new_account"] = (
        df["account_age_days"] <= 14
    ).astype(int)

    df["is_very_new_account"] = (
        df["account_age_days"] <= 7
    ).astype(int)

    # -----------------------------------------
    # 8. REFUND / CHARGEBACK FEATURES
    # -----------------------------------------

    df["refund_to_transaction_ratio"] = (
        df["refund_count"]
        / (
            df["customer_transaction_count_before"]
            + 1
        )
    )

    df["chargeback_to_transaction_ratio"] = (
        df["chargeback_count"]
        / (
            df["customer_transaction_count_before"]
            + 1
        )
    )

    df["refund_amount_ratio"] = (
        df["refund_amount"]
        / (
            df["amount"]
            + 1
        )
    )

    df["chargeback_amount_ratio"] = (
        df["chargeback_amount"]
        / (
            df["amount"]
            + 1
        )
    )

    # -----------------------------------------
    # 9. TEMPORAL FEATURES
    # -----------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # -----------------------------------------
    # 10. TIME SINCE CUSTOMER TRANSACTION
    # -----------------------------------------

    df["previous_customer_timestamp"] = (
        df.groupby("customer_id")
        ["timestamp"]
        .shift(1)
    )

    df["seconds_since_customer_transaction"] = (
        (
            df["timestamp"]
            - df["previous_customer_timestamp"]
        )
        .dt.total_seconds()
        .fillna(999999)
    )

    # -----------------------------------------
    # 11. VELOCITY FEATURES
    # -----------------------------------------
    #
    # Approximate historical velocity based
    # on previous customer transactions.
    #

    df["high_velocity_flag"] = (
        df[
            "seconds_since_customer_transaction"
        ] < 300
    ).astype(int)

    df["very_high_velocity_flag"] = (
        df[
            "seconds_since_customer_transaction"
        ] < 60
    ).astype(int)

    # -----------------------------------------
    # 12. GEOGRAPHIC CHANGE
    # -----------------------------------------

    df["previous_customer_location"] = (
        df.groupby("customer_id")
        ["location"]
        .shift(1)
    )

    df["location_changed"] = (
        (
            df["previous_customer_location"]
            .notna()
        )
        &
        (
            df["location"]
            !=
            df["previous_customer_location"]
        )
    ).astype(int)

    # -----------------------------------------
    # 13. DEVICE CHANGE
    # -----------------------------------------

    df["previous_customer_device"] = (
        df.groupby("customer_id")
        ["device_id"]
        .shift(1)
    )

    df["device_changed"] = (
        (
            df["previous_customer_device"]
            .notna()
        )
        &
        (
            df["device_id"]
            !=
            df["previous_customer_device"]
        )
    ).astype(int)

    # -----------------------------------------
    # 14. IP CHANGE
    # -----------------------------------------

    df["previous_customer_ip"] = (
        df.groupby("customer_id")
        ["ip_id"]
        .shift(1)
    )

    df["ip_changed"] = (
        (
            df["previous_customer_ip"]
            .notna()
        )
        &
        (
            df["ip_id"]
            !=
            df["previous_customer_ip"]
        )
    ).astype(int)

    # -----------------------------------------
    # 15. RISK SIGNAL COUNT
    # -----------------------------------------

    risk_flags = [
        "is_new_account",
        "is_very_new_account",
        "high_velocity_flag",
        "very_high_velocity_flag",
        "location_changed",
        "device_changed",
        "ip_changed"
    ]

    df["behavioral_risk_signal_count"] = (
        df[risk_flags]
        .sum(axis=1)
    )

    # -----------------------------------------
    # 16. CLEAN UP
    # -----------------------------------------

    df = df.drop(
        columns=[
            "previous_customer_timestamp",
            "previous_customer_location",
            "previous_customer_device",
            "previous_customer_ip"
        ]
    )

    return df


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "Loading transaction dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input rows: {len(df):,}"
    )

    print(
        "Building point-in-time features..."
    )

    df = build_features(
        df
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Feature engineering complete."
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print()
    print(
        "Feature file saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Generated features:"
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "timestamp"
        ]
    ]

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )


if __name__ == "__main__":
    main()