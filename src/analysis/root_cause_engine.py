import os
import json
import pandas as pd

INPUT_FILE = "data/processed/fraud_features.csv"
OUTPUT_FILE = "reports/root_cause_analysis.csv"


def safe_number(value, default=0):
    if pd.isna(value):
        return default
    return float(value)


def add_reason(
    reasons,
    code,
    title,
    explanation,
    severity,
    evidence
):
    reasons.append({
        "code": code,
        "title": title,
        "explanation": explanation,
        "severity": severity,
        "evidence": evidence
    })


def analyze_transaction(row):

    reasons = []

    amount = safe_number(
        row.get("amount")
    )

    customer_average = safe_number(
        row.get(
            "customer_avg_amount_before"
        ),
        amount
    )

    merchant_average = safe_number(
        row.get(
            "merchant_avg_amount_before"
        ),
        amount
    )

    transaction_count = safe_number(
        row.get(
            "transaction_count_last_hour"
        )
    )

    device_customers = safe_number(
        row.get(
            "device_customer_count"
        )
    )

    ip_customers = safe_number(
        row.get(
            "ip_customer_count"
        )
    )

    address_customers = safe_number(
        row.get(
            "address_customer_count"
        )
    )

    account_age = safe_number(
        row.get(
            "account_age_days"
        )
    )

    customer_amount_ratio = safe_number(
        row.get(
            "amount_vs_customer_average"
        ),
        1
    )

    merchant_amount_ratio = safe_number(
        row.get(
            "amount_vs_merchant_average"
        ),
        1
    )

    refund_ratio = safe_number(
        row.get(
            "refund_to_transaction_ratio"
        )
    )

    chargeback_ratio = safe_number(
        row.get(
            "chargeback_to_transaction_ratio"
        )
    )

    velocity = safe_number(
        row.get(
            "behavioral_risk_signal_count"
        )
    )

    location_changed = int(
        safe_number(
            row.get(
                "location_changed"
            )
        )
    )

    device_changed = int(
        safe_number(
            row.get(
                "device_changed"
            )
        )
    )

    ip_changed = int(
        safe_number(
            row.get(
                "ip_changed"
            )
        )
    )

    # -----------------------------------------
    # AMOUNT ANOMALY
    # -----------------------------------------

    if (
        customer_amount_ratio >= 3
        and amount >= 10000
    ):
        add_reason(
            reasons,
            "AMOUNT_ANOMALY",
            "Unusual transaction amount",
            (
                "Transaction amount is "
                f"{customer_amount_ratio:.1f}× "
                "the customer's historical average."
            ),
            "HIGH",
            {
                "transaction_amount": round(
                    amount,
                    2
                ),
                "customer_average": round(
                    customer_average,
                    2
                ),
                "ratio": round(
                    customer_amount_ratio,
                    2
                )
            }
        )

    elif (
        merchant_amount_ratio >= 4
        and amount >= 20000
    ):
        add_reason(
            reasons,
            "MERCHANT_AMOUNT_ANOMALY",
            "Merchant-level amount anomaly",
            (
                "Transaction amount is "
                f"{merchant_amount_ratio:.1f}× "
                "the merchant's historical average."
            ),
            "MEDIUM",
            {
                "transaction_amount": round(
                    amount,
                    2
                ),
                "merchant_average": round(
                    merchant_average,
                    2
                ),
                "ratio": round(
                    merchant_amount_ratio,
                    2
                )
            }
        )

    # -----------------------------------------
    # VELOCITY
    # -----------------------------------------

    if transaction_count >= 10:

        add_reason(
            reasons,
            "HIGH_VELOCITY",
            "Unusual transaction velocity",
            (
                f"{int(transaction_count)} "
                "transactions were observed "
                "for this customer within "
                "the previous hour."
            ),
            "CRITICAL",
            {
                "transactions_last_hour": int(
                    transaction_count
                )
            }
        )

    elif transaction_count >= 5:

        add_reason(
            reasons,
            "ELEVATED_VELOCITY",
            "Elevated transaction velocity",
            (
                f"{int(transaction_count)} "
                "transactions were observed "
                "within the previous hour."
            ),
            "HIGH",
            {
                "transactions_last_hour": int(
                    transaction_count
                )
            }
        )

    # -----------------------------------------
    # NEW ACCOUNT
    # -----------------------------------------

    if account_age <= 3:

        add_reason(
            reasons,
            "VERY_NEW_ACCOUNT",
            "Very recently created account",
            (
                f"Account age is only "
                f"{int(account_age)} days."
            ),
            "HIGH",
            {
                "account_age_days": int(
                    account_age
                )
            }
        )

    elif account_age <= 14:

        add_reason(
            reasons,
            "NEW_ACCOUNT",
            "New customer account",
            (
                f"Account age is "
                f"{int(account_age)} days."
            ),
            "MEDIUM",
            {
                "account_age_days": int(
                    account_age
                )
            }
        )

    # -----------------------------------------
    # DEVICE REUSE
    # -----------------------------------------

    if device_customers >= 5:

        add_reason(
            reasons,
            "DEVICE_REUSE",
            "Device shared across customers",
            (
                f"This device is associated "
                f"with {int(device_customers)} "
                "different customers."
            ),
            "CRITICAL",
            {
                "customer_count": int(
                    device_customers
                )
            }
        )

    elif device_customers >= 3:

        add_reason(
            reasons,
            "DEVICE_REUSE",
            "Device reused across customers",
            (
                f"This device is associated "
                f"with {int(device_customers)} "
                "different customers."
            ),
            "HIGH",
            {
                "customer_count": int(
                    device_customers
                )
            }
        )

    # -----------------------------------------
    # IP REUSE
    # -----------------------------------------

    if ip_customers >= 8:

        add_reason(
            reasons,
            "IP_REUSE",
            "High IP concentration",
            (
                f"This IP is associated "
                f"with {int(ip_customers)} "
                "different customers."
            ),
            "CRITICAL",
            {
                "customer_count": int(
                    ip_customers
                )
            }
        )

    elif ip_customers >= 5:

        add_reason(
            reasons,
            "IP_REUSE",
            "IP shared across customers",
            (
                f"This IP is associated "
                f"with {int(ip_customers)} "
                "different customers."
            ),
            "HIGH",
            {
                "customer_count": int(
                    ip_customers
                )
            }
        )

    # -----------------------------------------
    # ADDRESS REUSE
    # -----------------------------------------

    if address_customers >= 5:

        add_reason(
            reasons,
            "ADDRESS_REUSE",
            "Address linked to multiple customers",
            (
                f"This address is associated "
                f"with {int(address_customers)} "
                "different customers."
            ),
            "HIGH",
            {
                "customer_count": int(
                    address_customers
                )
            }
        )

    # -----------------------------------------
    # GEOGRAPHIC CHANGE
    # -----------------------------------------

    if location_changed:

        add_reason(
            reasons,
            "LOCATION_CHANGE",
            "Unexpected geographic change",
            (
                "Customer location differs "
                "from their previous transaction."
            ),
            "MEDIUM",
            {
                "current_location":
                    row.get("location")
            }
        )

    # -----------------------------------------
    # DEVICE CHANGE
    # -----------------------------------------

    if device_changed:

        add_reason(
            reasons,
            "DEVICE_CHANGE",
            "Customer switched devices",
            (
                "Transaction occurred from "
                "a different device than "
                "the customer's previous transaction."
            ),
            "MEDIUM",
            {}
        )

    # -----------------------------------------
    # IP CHANGE
    # -----------------------------------------

    if ip_changed:

        add_reason(
            reasons,
            "IP_CHANGE",
            "Customer IP changed",
            (
                "Transaction originated from "
                "a different IP than the "
                "customer's previous transaction."
            ),
            "LOW",
            {}
        )

    # -----------------------------------------
    # REFUND ABUSE
    # -----------------------------------------

    if refund_ratio >= 0.30:

        add_reason(
            reasons,
            "REFUND_ABUSE",
            "Elevated refund behavior",
            (
                "Customer has an unusually "
                "high historical refund ratio."
            ),
            "HIGH",
            {
                "refund_ratio": round(
                    refund_ratio,
                    3
                )
            }
        )

    # -----------------------------------------
    # CHARGEBACK HISTORY
    # -----------------------------------------

    if chargeback_ratio >= 0.10:

        add_reason(
            reasons,
            "CHARGEBACK_HISTORY",
            "Previous chargeback activity",
            (
                "Customer has historical "
                "chargeback activity."
            ),
            "HIGH",
            {
                "chargeback_ratio": round(
                    chargeback_ratio,
                    3
                )
            }
        )

    # -----------------------------------------
    # RISK SCORE
    # -----------------------------------------

    severity_weights = {
        "LOW": 5,
        "MEDIUM": 10,
        "HIGH": 20,
        "CRITICAL": 30
    }

    raw_score = sum(
        severity_weights[
            reason["severity"]
        ]
        for reason in reasons
    )

    risk_score = min(
        raw_score,
        100
    )

    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reason_count": len(reasons),
        "reasons": reasons
    }


def main():

    print(
        "Loading feature dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        "Analyzing transactions..."
    )

    results = []

    for _, row in df.iterrows():

        analysis = analyze_transaction(
            row
        )

        results.append({
            "transaction_id":
                row["transaction_id"],

            "customer_id":
                row["customer_id"],

            "merchant_id":
                row["merchant_id"],

            "amount":
                row["amount"],

            "timestamp":
                row["timestamp"],

            "is_fraud":
                row["is_fraud"],

            "fraud_type":
                row["fraud_type"],

            "incident_id":
                row.get(
                    "incident_id"
                ),

            "incident_type":
                row.get(
                    "incident_type"
                ),

            "incident_severity":
                row.get(
                    "incident_severity"
                ),

            "risk_score":
                analysis["risk_score"],

            "risk_level":
                analysis["risk_level"],

            "reason_count":
                analysis["reason_count"],

            "reasons":
                json.dumps(
                    analysis["reasons"]
                )
        })

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "===== RISK DISTRIBUTION ====="
    )

    print(
        results_df[
            "risk_level"
        ].value_counts()
    )

    print()
    print(
        "===== TOP RISK TRANSACTIONS ====="
    )

    top_risk = (
        results_df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    print(
        top_risk[
            [
                "transaction_id",
                "amount",
                "risk_score",
                "risk_level",
                "fraud_type"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Root-cause analysis saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()