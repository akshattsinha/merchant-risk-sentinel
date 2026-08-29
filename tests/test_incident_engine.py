import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.incident_engine import (
    calculate_severity,
    format_incident_type,
    build_root_causes,
)


def test_critical_risk_produces_critical_severity():
    assert calculate_severity(95) == "CRITICAL"


def test_high_risk_produces_high_severity():
    assert calculate_severity(80) == "HIGH"


def test_medium_risk_produces_medium_severity():
    assert calculate_severity(65) == "MEDIUM"


def test_low_risk_produces_low_severity():
    assert calculate_severity(30) == "LOW"


def test_source_severity_takes_priority():
    assert calculate_severity(
        30,
        source_severity="HIGH",
    ) == "HIGH"


def test_incident_type_is_normalized():
    assert (
        format_incident_type("account_takeover")
        == "ACCOUNT_TAKEOVER"
    )

    assert (
        format_incident_type("refund_abuse")
        == "REFUND_ABUSE"
    )


def test_unknown_incident_type_is_uppercase():
    assert (
        format_incident_type("custom_risk")
        == "CUSTOM_RISK"
    )


def test_missing_incident_type_uses_default():
    assert (
        format_incident_type(None)
        == "MULTI_SIGNAL_RISK"
    )


def make_incident_group():
    return pd.DataFrame([
        {
            "customer_id": "CUS_001",
            "device_id": "DEV_001",
            "ip_id": "IP_001",
            "address_id": "ADDR_001",
        },
        {
            "customer_id": "CUS_002",
            "device_id": "DEV_001",
            "ip_id": "IP_002",
            "address_id": "ADDR_002",
        },
    ])


def test_root_cause_analysis_returns_reasons():
    group = make_incident_group()

    reasons = build_root_causes(
        group,
        "COORDINATED_ACCOUNT_ABUSE",
    )

    assert isinstance(reasons, list)
    assert len(reasons) > 0


def test_root_cause_analysis_contains_strings():
    group = make_incident_group()

    reasons = build_root_causes(
        group,
        "COORDINATED_ACCOUNT_ABUSE",
    )

    assert all(
        isinstance(reason, str)
        for reason in reasons
    )
