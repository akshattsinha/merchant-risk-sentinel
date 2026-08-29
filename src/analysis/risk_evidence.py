"""
Deterministic risk evidence generation.

This module does NOT make the fraud decision.

The existing risk engine remains the source of truth.
This module converts the engine's behavioral features
into explicit, human-readable evidence.

The local Qwen/Ollama assistant can then explain this
evidence to the investigator.
"""

from typing import Any, Dict, List


def _severity(
    level: str,
) -> str:
    """
    Normalize evidence severity.
    """

    allowed = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    level = str(
        level
    ).upper()

    if level in allowed:
        return level

    return "MEDIUM"


def build_risk_evidence(
    risk: Dict[str, Any],
    behavioral_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert the existing /predict response into
    deterministic risk evidence.

    This function does NOT predict fraud.

    It only explains the evidence already generated
    by the existing risk engine.
    """

    evidence: List[Dict[str, Any]] = []

    # ========================================================
    # AMOUNT ANOMALY
    # ========================================================

    amount_vs_customer = float(
        behavioral_features.get(
            "amount_vs_customer_average",
            0,
        )
        or 0
    )

    amount_vs_merchant = float(
        behavioral_features.get(
            "amount_to_merchant_average",
            0,
        )
        or 0
    )

    if amount_vs_customer >= 3:

        evidence.append(
            {
                "factor": "Amount anomaly",
                "value": (
                    f"{amount_vs_customer:.2f}× "
                    "customer average"
                ),
                "severity": _severity(
                    "CRITICAL"
                    if amount_vs_customer >= 5
                    else "HIGH"
                ),
                "explanation": (
                    "The transaction amount is "
                    f"{amount_vs_customer:.2f}× "
                    "the customer's historical average."
                ),
            }
        )

    elif amount_vs_customer >= 2:

        evidence.append(
            {
                "factor": "Amount anomaly",
                "value": (
                    f"{amount_vs_customer:.2f}× "
                    "customer average"
                ),
                "severity": _severity(
                    "MEDIUM"
                ),
                "explanation": (
                    "The transaction amount is "
                    f"{amount_vs_customer:.2f}× "
                    "the customer's historical average."
                ),
            }
        )

    # ========================================================
    # MERCHANT AMOUNT ANOMALY
    # ========================================================

    if amount_vs_merchant >= 3:

        evidence.append(
            {
                "factor": "Merchant amount anomaly",
                "value": (
                    f"{amount_vs_merchant:.2f}× "
                    "merchant average"
                ),
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The transaction amount is "
                    f"{amount_vs_merchant:.2f}× "
                    "the merchant's historical average."
                ),
            }
        )

    elif amount_vs_merchant >= 1.5:

        evidence.append(
            {
                "factor": "Merchant amount anomaly",
                "value": (
                    f"{amount_vs_merchant:.2f}× "
                    "merchant average"
                ),
                "severity": _severity(
                    "MEDIUM"
                ),
                "explanation": (
                    "The transaction amount is "
                    f"{amount_vs_merchant:.2f}× "
                    "the merchant's historical average."
                ),
            }
        )

    # ========================================================
    # DEVICE SHARING
    # ========================================================

    device_customer_count = int(
        behavioral_features.get(
            "device_customer_count",
            0,
        )
        or 0
    )

    if device_customer_count >= 10:

        evidence.append(
            {
                "factor": "Shared device",
                "value": (
                    f"{device_customer_count} customers"
                ),
                "severity": _severity(
                    "CRITICAL"
                ),
                "explanation": (
                    "The device is associated with "
                    f"{device_customer_count} customers."
                ),
            }
        )

    elif device_customer_count >= 3:

        evidence.append(
            {
                "factor": "Shared device",
                "value": (
                    f"{device_customer_count} customers"
                ),
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The device is associated with "
                    f"{device_customer_count} customers."
                ),
            }
        )

    # ========================================================
    # IP SHARING
    # ========================================================

    ip_customer_count = int(
        behavioral_features.get(
            "ip_customer_count",
            0,
        )
        or 0
    )

    if ip_customer_count >= 10:

        evidence.append(
            {
                "factor": "Shared IP address",
                "value": (
                    f"{ip_customer_count} customers"
                ),
                "severity": _severity(
                    "CRITICAL"
                ),
                "explanation": (
                    "The IP address is associated with "
                    f"{ip_customer_count} customers."
                ),
            }
        )

    elif ip_customer_count >= 3:

        evidence.append(
            {
                "factor": "Shared IP address",
                "value": (
                    f"{ip_customer_count} customers"
                ),
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The IP address is associated with "
                    f"{ip_customer_count} customers."
                ),
            }
        )

    # ========================================================
    # ADDRESS SHARING
    # ========================================================

    address_customer_count = int(
        behavioral_features.get(
            "address_customer_count",
            0,
        )
        or 0
    )

    if address_customer_count >= 10:

        evidence.append(
            {
                "factor": "Shared address",
                "value": (
                    f"{address_customer_count} customers"
                ),
                "severity": _severity(
                    "CRITICAL"
                ),
                "explanation": (
                    "The address is associated with "
                    f"{address_customer_count} customers."
                ),
            }
        )

    elif address_customer_count >= 3:

        evidence.append(
            {
                "factor": "Shared address",
                "value": (
                    f"{address_customer_count} customers"
                ),
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The address is associated with "
                    f"{address_customer_count} customers."
                ),
            }
        )

    # ========================================================
    # DEVICE CHANGE
    # ========================================================

    if int(
        behavioral_features.get(
            "device_changed",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "Device change",
                "value": "Detected",
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The transaction was made using "
                    "a changed device."
                ),
            }
        )

    # ========================================================
    # IP CHANGE
    # ========================================================

    if int(
        behavioral_features.get(
            "ip_changed",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "IP change",
                "value": "Detected",
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The transaction was made from "
                    "a changed IP address."
                ),
            }
        )

    # ========================================================
    # LOCATION CHANGE
    # ========================================================

    if int(
        behavioral_features.get(
            "location_changed",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "Location change",
                "value": "Detected",
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The transaction occurred after "
                    "a detected location change."
                ),
            }
        )

    # ========================================================
    # VELOCITY
    # ========================================================

    transaction_count_last_hour = int(
        behavioral_features.get(
            "transaction_count_last_hour",
            0,
        )
        or 0
    )

    if int(
        behavioral_features.get(
            "very_high_velocity_flag",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "Very high transaction velocity",
                "value": (
                    f"{transaction_count_last_hour} "
                    "transactions in recent activity"
                ),
                "severity": _severity(
                    "CRITICAL"
                ),
                "explanation": (
                    "The transaction exhibits "
                    "very high transaction velocity."
                ),
            }
        )

    elif int(
        behavioral_features.get(
            "high_velocity_flag",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "High transaction velocity",
                "value": (
                    f"{transaction_count_last_hour} "
                    "transactions in recent activity"
                ),
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The transaction exhibits "
                    "high transaction velocity."
                ),
            }
        )

    # ========================================================
    # NEW ACCOUNT
    # ========================================================

    if int(
        behavioral_features.get(
            "is_very_new_account",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "Very new account",
                "value": "Detected",
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The transaction originates from "
                    "a very new account."
                ),
            }
        )

    elif int(
        behavioral_features.get(
            "is_new_account",
            0,
        )
        or 0
    ) == 1:

        evidence.append(
            {
                "factor": "New account",
                "value": "Detected",
                "severity": _severity(
                    "MEDIUM"
                ),
                "explanation": (
                    "The transaction originates from "
                    "a new account."
                ),
            }
        )

    # ========================================================
    # REFUND / CHARGEBACK SIGNALS
    # ========================================================

    refund_ratio = float(
        behavioral_features.get(
            "refund_to_transaction_ratio",
            0,
        )
        or 0
    )

    if refund_ratio > 0.20:

        evidence.append(
            {
                "factor": "Refund behavior",
                "value": (
                    f"{refund_ratio:.2%}"
                ),
                "severity": _severity(
                    "HIGH"
                ),
                "explanation": (
                    "The account shows elevated "
                    "refund activity."
                ),
            }
        )

    chargeback_ratio = float(
        behavioral_features.get(
            "chargeback_to_transaction_ratio",
            0,
        )
        or 0
    )

    if chargeback_ratio > 0.10:

        evidence.append(
            {
                "factor": "Chargeback behavior",
                "value": (
                    f"{chargeback_ratio:.2%}"
                ),
                "severity": _severity(
                    "CRITICAL"
                ),
                "explanation": (
                    "The account shows elevated "
                    "chargeback activity."
                ),
            }
        )

    # ========================================================
    # MODEL OUTPUT
    # ========================================================

    fraud_probability = float(
        risk.get(
            "fraud_probability",
            0,
        )
        or 0
    )

    risk_score = float(
        risk.get(
            "risk_score",
            0,
        )
        or 0
    )

    risk_level = str(
        risk.get(
            "risk_level",
            "UNKNOWN",
        )
    )

    recommended_action = str(
        risk.get(
            "recommended_action",
            "UNKNOWN",
        )
    )

    # ========================================================
    # SORT EVIDENCE
    # ========================================================

    severity_order = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    evidence.sort(
        key=lambda item:
        severity_order.get(
            item["severity"],
            0,
        ),
        reverse=True,
    )

    # ========================================================
    # RETURN EVIDENCE OBJECT
    # ========================================================

    return {
        "risk_summary": {
            "fraud_probability":
                fraud_probability,

            "fraud_probability_percent":
                round(
                    fraud_probability * 100,
                    2,
                ),

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "recommended_action":
                recommended_action,
        },

        "risk_factor_count":
            len(evidence),

        "risk_factors":
            evidence,

        "relationship_evidence": {
            "device_customer_count":
                device_customer_count,

            "ip_customer_count":
                ip_customer_count,

            "address_customer_count":
                address_customer_count,
        },

        "explanation": {
            "source":
                "deterministic_risk_engine",

            "llm_role":
                "explain_existing_evidence",

            "decision_source":
                "existing_fraud_risk_engine",
        },
    }