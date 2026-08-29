"""
Fraud Relationship / Graph Analysis

Uses the existing incident_analysis.csv as the source of
incident -> transaction membership.

The raw transaction dataset is NOT modified.

Architecture:

incident_analysis.csv
        |
        | incident_id
        v
Incident transactions
        |
        +---- customer_id
        +---- device_id
        +---- ip_id
        +---- address_id
        +---- transaction_id
        |
        v
Relationship Analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INCIDENT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "incident_analysis.csv"
)


# ============================================================
# HELPERS
# ============================================================

def _unique_non_empty(
    values: Iterable[Any],
) -> list[str]:

    result = []

    seen = set()

    for value in values:

        if value is None:
            continue

        text = str(value).strip()

        if not text:
            continue

        if text.lower() in {
            "nan",
            "none",
            "null",
        }:
            continue

        if text not in seen:

            seen.add(text)
            result.append(text)

    return result


def _numeric_sum(
    df: pd.DataFrame,
    column: Optional[str],
) -> float:

    if not column:
        return 0.0

    if column not in df.columns:
        return 0.0

    return float(
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )


def _find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# LOAD INCIDENT DATA
# ============================================================

def load_incident_analysis(
    path: Optional[str | Path] = None,
) -> pd.DataFrame:

    incident_path = (
        Path(path)
        if path
        else INCIDENT_FILE
    )

    if not incident_path.exists():

        raise FileNotFoundError(
            "Incident analysis file not found: "
            f"{incident_path}"
        )

    return pd.read_csv(
        incident_path,
        low_memory=False,
    )


# ============================================================
# COLUMN RESOLUTION
# ============================================================

def resolve_columns(
    df: pd.DataFrame,
) -> Dict[str, Optional[str]]:

    return {

        "transaction_id": _find_column(
            df,
            [
                "transaction_id",
                "transactionid",
                "txn_id",
            ],
        ),

        "customer_id": _find_column(
            df,
            [
                "customer_id",
                "customerid",
            ],
        ),

        "merchant_id": _find_column(
            df,
            [
                "merchant_id",
                "merchantid",
            ],
        ),

        "device_id": _find_column(
            df,
            [
                "device_id",
                "deviceid",
            ],
        ),

        "ip_id": _find_column(
            df,
            [
                "ip_id",
                "ip_address",
                "ip",
            ],
        ),

        "address_id": _find_column(
            df,
            [
                "address_id",
                "address",
            ],
        ),

        "amount": _find_column(
            df,
            [
                "amount",
                "transaction_amount",
            ],
        ),

        "fraud_type": _find_column(
            df,
            [
                "fraud_type",
                "fraud_category",
            ],
        ),

        "incident_id": _find_column(
            df,
            [
                "incident_id",
            ],
        ),

        "detected_incident_id": _find_column(
            df,
            [
                "detected_incident_id",
            ],
        ),

        "risk_score": _find_column(
            df,
            [
                "risk_score",
                "risk",
                "fraud_score",
            ],
        ),

        "incident_type": _find_column(
            df,
            [
                "incident_type",
            ],
        ),

        "incident_severity": _find_column(
            df,
            [
                "incident_severity",
                "severity",
            ],
        ),
    }


# ============================================================
# INCIDENT TRANSACTIONS
# ============================================================

def get_incident_transactions(
    incident_id: str,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    """
    Return all transactions belonging to an incident.

    Supports both incident ID formats used by the project:

        Dashboard:
            INC-0017

        Internal incident engine:
            INC_017

    The function first checks detected_incident_id,
    then incident_id.
    """

    if df is None:
        df = load_incident_analysis()

    if df.empty:
        return df.copy()

    columns = resolve_columns(df)

    target = str(incident_id).strip()

    # --------------------------------------------------------
    # 1. Dashboard / generated incident ID
    #
    # Example:
    # INC-0017
    # --------------------------------------------------------

    detected_column = columns.get(
        "detected_incident_id"
    )

    if detected_column:

        detected_values = (
            df[detected_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        matches = df.loc[
            detected_values == target
        ]

        if not matches.empty:
            return matches.copy()

    # --------------------------------------------------------
    # 2. Internal incident ID
    #
    # Example:
    # INC_017
    # --------------------------------------------------------

    incident_column = columns.get(
        "incident_id"
    )

    if incident_column:

        incident_values = (
            df[incident_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        matches = df.loc[
            incident_values == target
        ]

        if not matches.empty:
            return matches.copy()

    # --------------------------------------------------------
    # 3. Normalize formats as a final fallback
    #
    # INC-0017
    #     ↓
    # INC_017
    # --------------------------------------------------------

    normalized_target = (
        target
        .replace("-", "_")
    )

    if normalized_target.startswith(
        "INC_"
    ):

        numeric_part = (
            normalized_target
            .replace("INC_", "", 1)
        )

        try:

            numeric_value = int(
                numeric_part
            )

            internal_target = (
                f"INC_{numeric_value:03d}"
            )

            if incident_column:

                incident_values = (
                    df[incident_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                matches = df.loc[
                    incident_values
                    == internal_target
                ]

                if not matches.empty:
                    return matches.copy()

        except ValueError:
            pass

    return df.iloc[0:0].copy()


# ============================================================
# RELATED TRANSACTIONS
# ============================================================

def find_related_transactions(
    all_transactions: pd.DataFrame,
    incident_transactions: pd.DataFrame,
) -> pd.DataFrame:

    if (
        all_transactions.empty
        or incident_transactions.empty
    ):

        return all_transactions.iloc[
            0:0
        ].copy()

    columns = resolve_columns(
        all_transactions
    )

    relationship_columns = [
        "customer_id",
        "device_id",
        "ip_id",
        "address_id",
    ]

    masks = []

    for key in relationship_columns:

        column = columns.get(key)

        if not column:
            continue

        if column not in incident_transactions.columns:
            continue

        incident_values = set(
            _unique_non_empty(
                incident_transactions[
                    column
                ].tolist()
            )
        )

        if not incident_values:
            continue

        values = (
            all_transactions[
                column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        masks.append(
            values.isin(
                incident_values
            )
        )

    if not masks:

        return all_transactions.iloc[
            0:0
        ].copy()

    combined = masks[0].copy()

    for mask in masks[1:]:

        combined = (
            combined
            | mask
        )

    return all_transactions.loc[
        combined
    ].copy()


# ============================================================
# INCIDENT RELATIONSHIP SUMMARY
# ============================================================

def build_relationship_summary(
    incident_id: str,
) -> Dict[str, Any]:

    df = load_incident_analysis()

    columns = resolve_columns(df)

    incident_transactions = (
        get_incident_transactions(
            incident_id,
            df,
        )
    )

    if incident_transactions.empty:

        return {
            "found": False,
            "incident_id": incident_id,
            "message": (
                "No transactions found "
                f"for incident {incident_id}"
            ),
        }

    related_transactions = (
        find_related_transactions(
            df,
            incident_transactions,
        )
    )

    transaction_column = (
        columns.get("transaction_id")
    )

    customer_column = (
        columns.get("customer_id")
    )

    device_column = (
        columns.get("device_id")
    )

    ip_column = (
        columns.get("ip_id")
    )

    address_column = (
        columns.get("address_id")
    )

    amount_column = (
        columns.get("amount")
    )

    fraud_type_column = (
        columns.get("fraud_type")
    )

    risk_column = (
        columns.get("risk_score")
    )

    # --------------------------------------------------------
    # ENTITY LISTS
    # --------------------------------------------------------

    related_customers = (
        _unique_non_empty(
            related_transactions[
                customer_column
            ].tolist()
        )
        if customer_column
        else []
    )

    related_devices = (
        _unique_non_empty(
            related_transactions[
                device_column
            ].tolist()
        )
        if device_column
        else []
    )

    related_ips = (
        _unique_non_empty(
            related_transactions[
                ip_column
            ].tolist()
        )
        if ip_column
        else []
    )

    related_addresses = (
        _unique_non_empty(
            related_transactions[
                address_column
            ].tolist()
        )
        if address_column
        else []
    )

    related_transaction_ids = (
        _unique_non_empty(
            related_transactions[
                transaction_column
            ].tolist()
        )
        if transaction_column
        else []
    )

    # --------------------------------------------------------
    # SUSPICIOUS TRANSACTIONS
    # --------------------------------------------------------

    suspicious_transactions = (
        related_transactions.copy()
    )

    if fraud_type_column:

        fraud_values = (
            suspicious_transactions[
                fraud_type_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        suspicious_transactions = (
            suspicious_transactions.loc[
                ~fraud_values.isin(
                    {
                        "",
                        "none",
                        "normal",
                        "legitimate",
                        "nan",
                    }
                )
            ]
        )

    suspicious_transaction_ids = (
        _unique_non_empty(
            suspicious_transactions[
                transaction_column
            ].tolist()
        )
        if transaction_column
        else []
    )

    # --------------------------------------------------------
    # EXPOSURE
    # --------------------------------------------------------

    connected_exposure = _numeric_sum(
        related_transactions,
        amount_column,
    )

    incident_exposure = _numeric_sum(
        incident_transactions,
        amount_column,
    )

    # --------------------------------------------------------
    # SHARED ENTITY COUNTS
    # --------------------------------------------------------

    def count_shared(
        column: Optional[str],
    ) -> int:

        if not column:
            return 0

        counts = (
            related_transactions[
                column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .value_counts()
        )

        return int(
            (counts > 1).sum()
        )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_values = []

    if risk_column:

        risk_values = (
            pd.to_numeric(
                incident_transactions[
                    risk_column
                ],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )

    average_risk = (
        float(
            sum(risk_values)
            / len(risk_values)
        )
        if risk_values
        else 0.0
    )

    maximum_risk = (
        float(
            max(risk_values)
        )
        if risk_values
        else 0.0
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "found": True,

        "incident_id": (
            str(incident_id)
        ),

        "incident_transaction_count": int(
            len(incident_transactions)
        ),

        "related_transaction_count": int(
            len(related_transactions)
        ),

        "related_suspicious_transaction_count": int(
            len(suspicious_transactions)
        ),

        "customer_count": int(
            len(related_customers)
        ),

        "device_count": int(
            len(related_devices)
        ),

        "ip_count": int(
            len(related_ips)
        ),

        "address_count": int(
            len(related_addresses)
        ),

        "connected_exposure": float(
            connected_exposure
        ),

        "incident_exposure": float(
            incident_exposure
        ),

        "average_risk_score": float(
            average_risk
        ),

        "maximum_risk_score": float(
            maximum_risk
        ),

        "shared_customer_entities": (
            count_shared(
                customer_column
            )
        ),

        "shared_device_entities": (
            count_shared(
                device_column
            )
        ),

        "shared_ip_entities": (
            count_shared(
                ip_column
            )
        ),

        "shared_address_entities": (
            count_shared(
                address_column
            )
        ),

        "related_customers": (
            related_customers
        ),

        "related_devices": (
            related_devices
        ),

        "related_ips": (
            related_ips
        ),

        "related_addresses": (
            related_addresses
        ),

        "related_transactions": (
            related_transaction_ids
        ),

        "related_suspicious_transactions": (
            suspicious_transaction_ids
        ),
    }


# ============================================================
# TRANSACTION RELATIONSHIPS
# ============================================================

def build_transaction_relationships(
    transaction_id: str,
) -> Dict[str, Any]:

    df = load_incident_analysis()

    # --------------------------------------------------------
    # EMPTY DATASET
    # --------------------------------------------------------

    if df.empty:

        return {
            "found": False,
            "transaction_id": transaction_id,
        }

    # --------------------------------------------------------
    # RESOLVE COLUMNS SAFELY
    # --------------------------------------------------------

    columns = resolve_columns(df)

    transaction_column = (
        columns.get("transaction_id")
    )

    # --------------------------------------------------------
    # TRANSACTION ID COLUMN NOT AVAILABLE
    # --------------------------------------------------------

    if not transaction_column:

        return {
            "found": False,
            "transaction_id": transaction_id,
        }

    # --------------------------------------------------------
    # FIND TRANSACTION
    # --------------------------------------------------------

    values = (
        df[transaction_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    transaction_rows = df.loc[
        values
        == str(transaction_id).strip()
    ]

    # --------------------------------------------------------
    # TRANSACTION NOT FOUND
    # --------------------------------------------------------

    if transaction_rows.empty:

        return {
            "found": False,
            "transaction_id": transaction_id,
        }

    # --------------------------------------------------------
    # FIND ASSOCIATED INCIDENT
    # --------------------------------------------------------

    incident_column = (
        columns.get("incident_id")
    )

    if incident_column:

        incident_values = (
            transaction_rows[
                incident_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        incident_values = [
            value
            for value in incident_values
            if value
            and value.lower()
            not in {
                "nan",
                "none",
                "null",
            }
        ]

        if incident_values:

            return build_relationship_summary(
                incident_values[0]
            )

    # --------------------------------------------------------
    # TRANSACTION-LEVEL RELATIONSHIP SUMMARY
    # --------------------------------------------------------

    customer_column = (
        columns.get("customer_id")
    )

    device_column = (
        columns.get("device_id")
    )

    ip_column = (
        columns.get("ip_id")
    )

    address_column = (
        columns.get("address_id")
    )

    transaction_customer_count = (
        len(
            _unique_non_empty(
                transaction_rows[
                    customer_column
                ].tolist()
            )
        )
        if customer_column
        else 0
    )

    transaction_device_count = (
        len(
            _unique_non_empty(
                transaction_rows[
                    device_column
                ].tolist()
            )
        )
        if device_column
        else 0
    )

    transaction_ip_count = (
        len(
            _unique_non_empty(
                transaction_rows[
                    ip_column
                ].tolist()
            )
        )
        if ip_column
        else 0
    )

    transaction_address_count = (
        len(
            _unique_non_empty(
                transaction_rows[
                    address_column
                ].tolist()
            )
        )
        if address_column
        else 0
    )

    return {

        "found": True,

        "transaction_id": (
            transaction_id
        ),

        "customer_count": (
            transaction_customer_count
        ),

        "device_count": (
            transaction_device_count
        ),

        "ip_count": (
            transaction_ip_count
        ),

        "address_count": (
            transaction_address_count
        ),
    }


# ============================================================
# HUMAN READABLE SUMMARY
# ============================================================

def format_relationship_summary(
    summary: Dict[str, Any],
) -> str:

    if not summary.get("found"):

        return (
            "No relationship evidence was "
            "found for this incident."
        )

    return (
        f"{summary['incident_id']} is connected "
        f"to {summary['customer_count']} customers, "
        f"{summary['device_count']} devices, "
        f"{summary['ip_count']} IP addresses, "
        f"{summary['address_count']} addresses, "
        f"and {summary['related_transaction_count']} "
        f"transactions. "
        f"Connected exposure is "
        f"₹{summary['connected_exposure']:,.2f}."
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "load_incident_analysis",
    "resolve_columns",
    "get_incident_transactions",
    "find_related_transactions",
    "build_relationship_summary",
    "build_transaction_relationships",
    "format_relationship_summary",
]