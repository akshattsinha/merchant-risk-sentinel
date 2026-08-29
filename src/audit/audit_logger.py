from pathlib import Path
from datetime import datetime, timezone
import json
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_audit.jsonl"
)

_write_lock = threading.Lock()


def write_prediction_audit(
    transaction,
    risk,
    evidence,
    model_version="1.0.0",
    threshold=0.30,
    incident_id=None,
):
    """
    Append one prediction decision to the audit trail.

    JSONL is used so every prediction becomes an independent,
    append-only audit record.
    """

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_factors = evidence.get(
        "risk_factors",
        [],
    )

    record = {
        "audit_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "transaction_id": transaction.get(
            "transaction_id"
        ),

        "customer_id": transaction.get(
            "customer_id"
        ),

        "merchant_id": transaction.get(
            "merchant_id"
        ),

        "amount": transaction.get(
            "amount"
        ),

        "transaction_timestamp": transaction.get(
            "timestamp"
        ),

        "fraud_probability": risk.get(
            "fraud_probability"
        ),

        "risk_score": risk.get(
            "risk_score"
        ),

        "risk_level": risk.get(
            "risk_level"
        ),

        "model_version": model_version,

        "threshold": threshold,

        "risk_factors": [
            {
                "factor": item.get("factor"),
                "value": item.get("value"),
                "severity": item.get("severity"),
            }
            for item in risk_factors
        ],

        "risk_factor_count": len(
            risk_factors
        ),

        "incident_id": incident_id,

        "decision": risk.get(
            "recommended_action"
        ),
    }

    with _write_lock:

        with open(
            AUDIT_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return record