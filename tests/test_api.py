import sys
from pathlib import Path

from fastapi.testclient import TestClient


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# API IMPORT
# ============================================================

from src.api.app import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == (
        "Merchant Risk Sentinel"
    )

    assert data["status"] == "online"

    assert data["version"] == "1.0.0"


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert "model_loaded" in data


# ============================================================
# INVALID AMOUNT
# ============================================================

def test_predict_rejects_invalid_amount():

    transaction = {

        "customer_id": "CUS_TEST",

        "merchant_id": "MER_TEST",

        "amount": 0,

        "timestamp": (
            "2026-07-24 12:00:00"
        ),

        "payment_method": "UPI",

        "device_id": "DEV_TEST",

        "ip_id": "IP_TEST",

        "address_id": "ADDR_TEST",

        "account_age_days": 10,

        "location": "Delhi",
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


# ============================================================
# NEGATIVE ACCOUNT AGE
# ============================================================

def test_predict_rejects_negative_account_age():

    transaction = {

        "customer_id": "CUS_TEST",

        "merchant_id": "MER_TEST",

        "amount": 1000,

        "timestamp": (
            "2026-07-24 12:00:00"
        ),

        "payment_method": "UPI",

        "device_id": "DEV_TEST",

        "ip_id": "IP_TEST",

        "address_id": "ADDR_TEST",

        "account_age_days": -1,

        "location": "Delhi",
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


# ============================================================
# MISSING REQUIRED FIELD
# ============================================================

def test_predict_rejects_missing_required_field():

    transaction = {

        "customer_id": "CUS_TEST",

        "merchant_id": "MER_TEST",

        "amount": 1000,

        "timestamp": (
            "2026-07-24 12:00:00"
        ),

        "payment_method": "UPI",

        "device_id": "DEV_TEST",

        "ip_id": "IP_TEST",

        "address_id": "ADDR_TEST",

        "account_age_days": 10,

        # location intentionally missing
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


# ============================================================
# TRANSACTION RELATIONSHIP NOT FOUND
# ============================================================

def test_transaction_relationship_not_found():

    response = client.get(
        "/transactions/"
        "DOES_NOT_EXIST/"
        "relationships"
    )

    assert response.status_code == 404