import pytest

from live_predictor import LiveRiskPredictor


@pytest.fixture
def predictor():
    return LiveRiskPredictor()


@pytest.fixture
def sample_transaction():
    return {
        "customer_id": "CUS_000001",
        "merchant_id": "MER_000001",
        "amount": 85000,
        "timestamp": "2026-07-24 12:00:00",
        "payment_method": "UPI",
        "device_id": "NEW_DEVICE_001",
        "ip_id": "NEW_IP_001",
        "address_id": "NEW_ADDRESS_001",
        "account_age_days": 3,
        "location": "Delhi",
    }


def test_live_prediction_returns_result(predictor, sample_transaction):
    result = predictor.predict(sample_transaction)

    assert result is not None
    assert isinstance(result, dict)


def test_live_prediction_contains_required_fields(
    predictor, sample_transaction
):
    result = predictor.predict(sample_transaction)

    required_fields = {
        "fraud_probability",
        "risk_score",
        "risk_level",
        "recommended_action",
    }

    assert required_fields.issubset(result.keys())


def test_fraud_probability_is_valid(predictor, sample_transaction):
    result = predictor.predict(sample_transaction)

    probability = result["fraud_probability"]

    assert isinstance(probability, (int, float))
    assert 0 <= probability <= 1


def test_risk_score_is_valid(predictor, sample_transaction):
    result = predictor.predict(sample_transaction)

    risk_score = result["risk_score"]

    assert isinstance(risk_score, (int, float))
    assert 0 <= risk_score <= 100


def test_risk_level_is_present(predictor, sample_transaction):
    result = predictor.predict(sample_transaction)

    assert result["risk_level"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }


def test_recommended_action_is_present(predictor, sample_transaction):
    result = predictor.predict(sample_transaction)

    assert isinstance(result["recommended_action"], str)
    assert len(result["recommended_action"]) > 0