import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.risk_evidence import build_risk_evidence


def make_transaction():
    return pd.Series({
        "transaction_id": "TXN_TEST_001",
        "customer_id": "CUS_001",
        "merchant_id": "MER_001",
        "amount": 50000,
        "device_id": "DEV_001",
        "ip_id": "IP_001",
        "address_id": "ADDR_001",
    })


def make_behavioral_features():
    return {
        "amount": 50000,
        "account_age_days": 3,
        "transaction_count": 1,
        "customer_transaction_count": 1,
        "device_transaction_count": 1,
        "ip_transaction_count": 1,
    }


def get_evidence():
    transaction = make_transaction()
    behavioral_features = make_behavioral_features()

    return build_risk_evidence(
        transaction,
        behavioral_features,
    )


def test_risk_evidence_returns_result():
    result = get_evidence()

    assert result is not None


def test_risk_evidence_has_expected_structure():
    result = get_evidence()

    assert isinstance(result, dict)


def test_risk_evidence_does_not_produce_invalid_risk():
    result = get_evidence()

    for key in ("risk_score", "evidence_score", "score"):
        if key in result:
            assert 0 <= float(result[key]) <= 100