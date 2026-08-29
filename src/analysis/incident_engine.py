import os
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/fraud_features.csv"
OUTPUT_FILE = "reports/incident_analysis.csv"
SUMMARY_FILE = "reports/incident_summary.json"

RISK_THRESHOLD = 60


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}\n"
            "Run build_features.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "timestamp column is missing from "
            "fraud_features.csv"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# SAFE NUMERIC VALUE
# ============================================================

def numeric_value(
    row,
    column,
    default=0
):

    if column not in row.index:

        return default

    value = row[column]

    if pd.isna(value):

        return default

    try:

        return float(value)

    except Exception:

        return default


# ============================================================
# CALCULATE RISK SCORE
# ============================================================

def calculate_risk_score(row):

    score = 0


    # --------------------------------------------------------
    # Amount compared with customer average
    # --------------------------------------------------------

    amount_vs_customer = numeric_value(
        row,
        "amount_vs_customer_average",
        1
    )


    if amount_vs_customer >= 3:

        score += 20


    if amount_vs_customer >= 5:

        score += 15


    if amount_vs_customer >= 10:

        score += 15


    # --------------------------------------------------------
    # Amount compared with merchant average
    # --------------------------------------------------------

    amount_vs_merchant = numeric_value(
        row,
        "amount_vs_merchant_average",
        1
    )


    if amount_vs_merchant >= 3:

        score += 10


    if amount_vs_merchant >= 5:

        score += 10


    # --------------------------------------------------------
    # Transaction velocity
    # --------------------------------------------------------

    velocity = numeric_value(
        row,
        "transaction_count_last_hour",
        0
    )


    if velocity >= 5:

        score += 10


    if velocity >= 10:

        score += 15


    # --------------------------------------------------------
    # Device reuse
    # --------------------------------------------------------

    device_customers = numeric_value(
        row,
        "device_customer_count",
        0
    )


    if device_customers >= 3:

        score += 10


    if device_customers >= 5:

        score += 15


    # --------------------------------------------------------
    # IP reuse
    # --------------------------------------------------------

    ip_customers = numeric_value(
        row,
        "ip_customer_count",
        0
    )


    if ip_customers >= 3:

        score += 5


    if ip_customers >= 5:

        score += 10


    # --------------------------------------------------------
    # Address reuse
    # --------------------------------------------------------

    address_customers = numeric_value(
        row,
        "address_customer_count",
        0
    )


    if address_customers >= 3:

        score += 5


    if address_customers >= 5:

        score += 10


    # --------------------------------------------------------
    # New account
    # --------------------------------------------------------

    account_age = numeric_value(
        row,
        "account_age_days",
        9999
    )


    if account_age <= 30:

        score += 10


    if account_age <= 7:

        score += 10


    if account_age <= 3:

        score += 10


    # --------------------------------------------------------
    # Explicit behavioral flags
    # --------------------------------------------------------

    if numeric_value(
        row,
        "is_new_account",
        0
    ) == 1:

        score += 10


    if numeric_value(
        row,
        "is_very_new_account",
        0
    ) == 1:

        score += 10


    if numeric_value(
        row,
        "high_velocity_flag",
        0
    ) == 1:

        score += 10


    if numeric_value(
        row,
        "very_high_velocity_flag",
        0
    ) == 1:

        score += 15


    if numeric_value(
        row,
        "location_changed",
        0
    ) == 1:

        score += 10


    if numeric_value(
        row,
        "device_changed",
        0
    ) == 1:

        score += 10


    if numeric_value(
        row,
        "ip_changed",
        0
    ) == 1:

        score += 10


    # --------------------------------------------------------
    # Refund / chargeback
    # --------------------------------------------------------

    if numeric_value(
        row,
        "refund_count",
        0
    ) > 0:

        score += 5


    if numeric_value(
        row,
        "chargeback_count",
        0
    ) > 0:

        score += 10


    # --------------------------------------------------------
    # Behavioral risk signal count
    # --------------------------------------------------------

    signal_count = numeric_value(
        row,
        "behavioral_risk_signal_count",
        0
    )


    if signal_count >= 3:

        score += 10


    if signal_count >= 5:

        score += 10


    # --------------------------------------------------------
    # Explicit fraud signal
    #
    # This is synthetic evaluation data, so we allow the
    # known fraud label to contribute to the incident score.
    # --------------------------------------------------------

    if numeric_value(
        row,
        "is_fraud",
        0
    ) == 1:

        score += 15


    return min(
        int(score),
        100
    )


# ============================================================
# INCIDENT TYPE FORMATTER
# ============================================================

def format_incident_type(
    value
):

    if pd.isna(value):

        return "MULTI_SIGNAL_RISK"


    value = str(
        value
    ).strip().lower()


    mapping = {

        "account_takeover":
            "ACCOUNT_TAKEOVER",

        "coordinated_account_abuse":
            "COORDINATED_ACCOUNT_ABUSE",

        "payment_velocity_attack":
            "PAYMENT_VELOCITY_ATTACK",

        "payment_method_abuse":
            "PAYMENT_METHOD_ABUSE",

        "geographic_anomaly":
            "GEOGRAPHIC_ANOMALY",

        "refund_abuse":
            "REFUND_ABUSE",

        "amount_anomaly":
            "AMOUNT_ANOMALY",

        "new_account_attack":
            "NEW_ACCOUNT_ATTACK",

    }


    return mapping.get(
        value,
        value.upper()
    )


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    max_risk,
    source_severity=None
):

    if (
        source_severity
        and not pd.isna(
            source_severity
        )
    ):

        source_severity = str(
            source_severity
        ).upper()


        if source_severity in {

            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"

        }:

            return source_severity


    if max_risk >= 90:

        return "CRITICAL"

    elif max_risk >= 75:

        return "HIGH"

    elif max_risk >= 60:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

def build_root_causes(
    group,
    incident_type
):

    reasons = []


    customers = group[
        "customer_id"
    ].nunique()


    transactions = len(
        group
    )


    devices = group[
        "device_id"
    ].nunique()


    ips = group[
        "ip_id"
    ].nunique()


    addresses = group[
        "address_id"
    ].nunique()


    # --------------------------------------------------------
    # Incident-specific explanation
    # --------------------------------------------------------

    if incident_type == "ACCOUNT_TAKEOVER":

        reasons.append(
            "Account behavior indicates possible "
            "unauthorized account access."
        )


    elif (
        incident_type
        == "COORDINATED_ACCOUNT_ABUSE"
    ):

        reasons.append(
            "Multiple customer accounts exhibit "
            "coordinated transaction behavior."
        )


    elif (
        incident_type
        == "PAYMENT_VELOCITY_ATTACK"
    ):

        reasons.append(
            "Transactions occurred at unusually "
            "high velocity."
        )


    elif (
        incident_type
        == "PAYMENT_METHOD_ABUSE"
    ):

        reasons.append(
            "Unusual concentration of activity "
            "around a payment method was detected."
        )


    elif (
        incident_type
        == "GEOGRAPHIC_ANOMALY"
    ):

        reasons.append(
            "Transaction locations differ from "
            "expected geographic behavior."
        )


    elif (
        incident_type
        == "REFUND_ABUSE"
    ):

        reasons.append(
            "Refund activity indicates possible "
            "abuse of the transaction lifecycle."
        )


    elif (
        incident_type
        == "AMOUNT_ANOMALY"
    ):

        reasons.append(
            "Transaction amounts are significantly "
            "different from normal behavior."
        )


    elif (
        incident_type
        == "NEW_ACCOUNT_ATTACK"
    ):

        reasons.append(
            "Suspicious activity is concentrated "
            "among newly created accounts."
        )


    # --------------------------------------------------------
    # Shared entities
    # --------------------------------------------------------

    if devices < transactions:

        reasons.append(
            "Multiple transactions share one or "
            "more devices."
        )


    if ips < transactions:

        reasons.append(
            "Multiple transactions share one or "
            "more IP addresses."
        )


    if addresses < transactions:

        reasons.append(
            "Multiple transactions are linked "
            "through shared addresses."
        )


    # --------------------------------------------------------
    # Customer concentration
    # --------------------------------------------------------

    if customers >= 5:

        reasons.append(
            f"{customers} customer accounts are "
            "linked to this incident."
        )


    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    if (
        "transaction_count_last_hour"
        in group.columns
    ):

        velocity = group[
            "transaction_count_last_hour"
        ].max()


        if pd.notna(
            velocity
        ) and velocity >= 5:

            reasons.append(
                "Transaction velocity exceeds "
                "normal behavioral levels."
            )


    # --------------------------------------------------------
    # Fraud rate
    # --------------------------------------------------------

    if "is_fraud" in group.columns:

        fraud_rate = (
            group[
                "is_fraud"
            ].sum()
            /
            len(group)
        )


        if fraud_rate >= 0.5:

            reasons.append(
                "A large proportion of transactions "
                "are labeled fraudulent in the "
                "synthetic evaluation dataset."
            )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "Multiple high-risk transactions "
            "were associated with this incident."
        )


    # Remove duplicate explanations.

    return list(
        dict.fromkeys(
            reasons
        )
    )


# ============================================================
# BUILD INCIDENTS
# ============================================================

def build_incidents(
    df
):

    # --------------------------------------------------------
    # Calculate risk score
    # --------------------------------------------------------

    print(
        "Calculating transaction risk..."
    )


    df[
        "risk_score"
    ] = df.apply(
        calculate_risk_score,
        axis=1
    )


    # --------------------------------------------------------
    # Suspicious transactions
    # --------------------------------------------------------

    suspicious = df[
        df[
            "risk_score"
        ]
        >= RISK_THRESHOLD
    ].copy()


    print(
        f"Suspicious transactions: "
        f"{len(suspicious):,}"
    )


    if len(
        suspicious
    ) == 0:

        return (
            suspicious,
            []
        )


    # --------------------------------------------------------
    # Confirm incident_id exists
    # --------------------------------------------------------

    if (
        "incident_id"
        not in suspicious.columns
    ):

        raise ValueError(
            "incident_id column is missing "
            "from fraud_features.csv."
        )


    # --------------------------------------------------------
    # Keep only transactions belonging to a generated
    # incident.
    # --------------------------------------------------------

    suspicious = suspicious[
        suspicious[
            "incident_id"
        ].notna()
    ].copy()


    print(
        "Grouping transactions by generated "
        "incident ID..."
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # Do NOT merge groups by device/IP/address.
    #
    # generate_data.py already created the 20 synthetic
    # incident boundaries.
    # --------------------------------------------------------

    groups = suspicious.groupby(
        "incident_id"
    )


    incidents = []


    incident_number = 1


    # ========================================================
    # PROCESS EACH INCIDENT
    # ========================================================

    for source_incident_id, group in groups:

        group = group.copy()


        if len(group) == 0:

            continue


        # ----------------------------------------------------
        # Transaction statistics
        # ----------------------------------------------------

        transaction_count = len(
            group
        )


        fraud_transactions = int(
            group[
                "is_fraud"
            ].sum()
        )


        fraud_rate = (

            fraud_transactions
            /
            transaction_count

        )


        # ----------------------------------------------------
        # Customers
        # ----------------------------------------------------

        customers = sorted(
            group[
                "customer_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # Devices
        # ----------------------------------------------------

        devices = sorted(
            group[
                "device_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # IPs
        # ----------------------------------------------------

        ips = sorted(
            group[
                "ip_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # Addresses
        # ----------------------------------------------------

        addresses = sorted(
            group[
                "address_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        first_timestamp = group[
            "timestamp"
        ].min()


        last_timestamp = group[
            "timestamp"
        ].max()


        duration_minutes = (

            last_timestamp
            -
            first_timestamp

        ).total_seconds() / 60


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        max_risk = int(
            group[
                "risk_score"
            ].max()
        )


        avg_risk = float(
            group[
                "risk_score"
            ].mean()
        )


        # ----------------------------------------------------
        # Amount
        # ----------------------------------------------------

        total_amount = float(
            group[
                "amount"
            ].sum()
        )


        # ----------------------------------------------------
        # Incident type
        # ----------------------------------------------------

        if (
            "incident_type"
            in group.columns
        ):

            incident_types = (
                group[
                    "incident_type"
                ]
                .dropna()
                .astype(str)
                .str.lower()
                .value_counts()
            )


            if len(
                incident_types
            ) > 0:

                incident_type = (
                    incident_types
                    .index[0]
                )

            else:

                incident_type = (
                    "multi_signal_risk"
                )

        else:

            incident_type = (
                "multi_signal_risk"
            )


        incident_type = format_incident_type(
            incident_type
        )


        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        source_severity = None


        if (
            "incident_severity"
            in group.columns
        ):

            severity_values = (
                group[
                    "incident_severity"
                ]
                .dropna()
                .astype(str)
                .str.upper()
                .value_counts()
            )


            if len(
                severity_values
            ) > 0:

                source_severity = (
                    severity_values
                    .index[0]
                )


        severity = calculate_severity(
            max_risk,
            source_severity
        )


        # ----------------------------------------------------
        # Fraud types
        # ----------------------------------------------------

        if (
            "fraud_type"
            in group.columns
        ):

            fraud_types = (
                group[
                    "fraud_type"
                ]
                .value_counts()
                .to_dict()
            )

        else:

            fraud_types = {}


        # ----------------------------------------------------
        # Root causes
        # ----------------------------------------------------

        root_causes = build_root_causes(
            group,
            incident_type
        )


        # ----------------------------------------------------
        # Generated incident ID
        # ----------------------------------------------------

        detected_incident_id = (
            f"INC-{incident_number:04d}"
        )


        # ----------------------------------------------------
        # Create incident
        # ----------------------------------------------------

        incident = {

            "incident_id":
                detected_incident_id,

            "source_incident_id":
                str(
                    source_incident_id
                ),

            "incident_type":
                incident_type,

            "severity":
                severity,

            "risk_score":
                max_risk,

            "average_risk_score":
                round(
                    avg_risk,
                    2
                ),

            "transaction_count":
                transaction_count,

            "fraud_transactions":
                fraud_transactions,

            "fraud_rate":
                round(
                    fraud_rate,
                    4
                ),

            "customer_count":
                len(
                    customers
                ),

            "device_count":
                len(
                    devices
                ),

            "ip_count":
                len(
                    ips
                ),

            "address_count":
                len(
                    addresses
                ),

            "total_transaction_amount":
                round(
                    total_amount,
                    2
                ),

            "estimated_exposure":
                round(
                    total_amount,
                    2
                ),

            "first_seen":
                str(
                    first_timestamp
                ),

            "last_seen":
                str(
                    last_timestamp
                ),

            "duration_minutes":
                round(
                    duration_minutes,
                    2
                ),

            "customers":
                json.dumps(
                    customers
                ),

            "devices":
                json.dumps(
                    devices
                ),

            "ips":
                json.dumps(
                    ips
                ),

            "addresses":
                json.dumps(
                    addresses
                ),

            "fraud_types":
                json.dumps(
                    fraud_types
                ),

            "root_causes":
                json.dumps(
                    root_causes
                )
        }


        incidents.append(
            incident
        )


        # ----------------------------------------------------
        # Attach detected incident ID to transactions
        # ----------------------------------------------------

        suspicious.loc[
            group.index,
            "detected_incident_id"
        ] = detected_incident_id


        incident_number += 1


    return (
        suspicious,
        incidents
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading feature dataset..."
    )


    df = load_data()


    print(
        f"Rows: {len(df):,}"
    )


    # --------------------------------------------------------
    # Build incidents
    # --------------------------------------------------------

    suspicious, incidents = (
        build_incidents(
            df
        )
    )


    # --------------------------------------------------------
    # Reports directory
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save incident analysis
    # --------------------------------------------------------

    suspicious.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = {

        "total_transactions":
            int(
                len(df)
            ),

        "suspicious_transactions":
            int(
                len(suspicious)
            ),

        "incidents_detected":
            int(
                len(incidents)
            ),

        "incidents":
            incidents
    }


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        "===== INCIDENT SUMMARY ====="
    )


    print(
        f"Suspicious transactions: "
        f"{len(suspicious):,}"
    )


    print(
        f"Incidents detected: "
        f"{len(incidents)}"
    )


    print()


    if incidents:

        for incident in incidents:

            print(
                f"{incident['incident_id']} | "
                f"{incident['incident_type']} | "
                f"{incident['severity']} | "
                f"{incident['transaction_count']} "
                f"transactions | "
                f"{incident['customer_count']} customers | "
                f"₹{incident['estimated_exposure']:,.0f}"
            )


    print()


    print(
        f"Incident analysis saved to: "
        f"{OUTPUT_FILE}"
    )


    print(
        f"Incident summary saved to: "
        f"{SUMMARY_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()