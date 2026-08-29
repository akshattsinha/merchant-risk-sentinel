import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.incident_engine import calculate_risk_score


def make_row(**values):
    return pd.Series(values)


def test_normal_transaction_has_low_score():
    row = make_row(
        amount_vs_customer_average=1,
        amount_vs_merchant_average=1,
    )

    score = calculate_risk_score(row)

    assert isinstance(score, (int, float))
    assert 0 <= score <= 100


def test_large_customer_amount_increases_risk():
    normal = make_row(
        amount_vs_customer_average=1,
        amount_vs_merchant_average=1,
    )

    suspicious = make_row(
        amount_vs_customer_average=10,
        amount_vs_merchant_average=1,
    )

    normal_score = calculate_risk_score(normal)
    suspicious_score = calculate_risk_score(suspicious)

    assert suspicious_score > normal_score


def test_large_merchant_amount_increases_risk():
    normal = make_row(
        amount_vs_customer_average=1,
        amount_vs_merchant_average=1,
    )

    suspicious = make_row(
        amount_vs_customer_average=1,
        amount_vs_merchant_average=10,
    )

    normal_score = calculate_risk_score(normal)
    suspicious_score = calculate_risk_score(suspicious)

    assert suspicious_score > normal_score


def test_risk_score_is_bounded():
    highly_suspicious = make_row(
        amount_vs_customer_average=100,
        amount_vs_merchant_average=100,
    )

    score = calculate_risk_score(highly_suspicious)

    assert 0 <= score <= 100