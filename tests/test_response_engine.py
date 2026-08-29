import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.response_engine import determine_action


def make_incident(
    severity="LOW",
    risk_score=20,
    transaction_count=1,
    estimated_exposure=1000,
):
    return {
        "severity": severity,
        "risk_score": risk_score,
        "transaction_count": transaction_count,
        "estimated_exposure": estimated_exposure,
    }


def test_low_risk_is_allowed():
    result = determine_action(
        make_incident(
            severity="LOW",
            risk_score=20,
        )
    )

    assert result["recommended_action"] == "ALLOW"
    assert result["priority"] == "P3"


def test_medium_risk_is_monitored():
    result = determine_action(
        make_incident(
            severity="MEDIUM",
            risk_score=60,
        )
    )

    assert result["recommended_action"] == "MONITOR"
    assert result["priority"] == "P2"


def test_high_risk_requires_step_up_verification():
    result = determine_action(
        make_incident(
            severity="HIGH",
            risk_score=80,
        )
    )

    assert (
        result["recommended_action"]
        == "STEP_UP_VERIFICATION"
    )

    assert result["priority"] == "P1"


def test_critical_risk_requires_hold_and_investigation():
    result = determine_action(
        make_incident(
            severity="CRITICAL",
            risk_score=95,
        )
    )

    assert (
        result["recommended_action"]
        == "HOLD_AND_INVESTIGATE"
    )

    assert result["priority"] == "P0"


def test_high_score_triggers_high_risk_response():
    result = determine_action(
        make_incident(
            severity="LOW",
            risk_score=90,
        )
    )

    assert (
        result["recommended_action"]
        == "HOLD_AND_INVESTIGATE"
    )


def test_large_transaction_cluster_adds_cluster_review():
    result = determine_action(
        make_incident(
            severity="MEDIUM",
            risk_score=60,
            transaction_count=20,
        )
    )

    assert (
        "Review incident-level transaction cluster"
        in result["next_steps"]
    )


def test_large_exposure_adds_financial_review():
    result = determine_action(
        make_incident(
            severity="HIGH",
            risk_score=80,
            estimated_exposure=500000,
        )
    )

    assert (
        "Prioritize financial exposure review"
        in result["next_steps"]
    )


def test_response_contains_explanation_and_next_steps():
    result = determine_action(
        make_incident(
            severity="HIGH",
            risk_score=80,
        )
    )

    assert isinstance(
        result["explanation"],
        str,
    )

    assert len(
        result["explanation"]
    ) > 0

    assert isinstance(
        result["next_steps"],
        list,
    )

    assert len(
        result["next_steps"]
    ) > 0
