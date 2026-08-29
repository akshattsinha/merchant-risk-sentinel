import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.fraud_graph import find_related_transactions


def make_transactions():
    return pd.DataFrame([
        {
            "transaction_id": "TXN_001",
            "customer_id": "CUS_001",
            "merchant_id": "MER_001",
            "device_id": "DEV_001",
            "ip_id": "IP_001",
            "address_id": "ADDR_001",
        },
        {
            "transaction_id": "TXN_002",
            "customer_id": "CUS_002",
            "merchant_id": "MER_001",
            "device_id": "DEV_001",
            "ip_id": "IP_002",
            "address_id": "ADDR_002",
        },
        {
            "transaction_id": "TXN_003",
            "customer_id": "CUS_003",
            "merchant_id": "MER_001",
            "device_id": "DEV_003",
            "ip_id": "IP_001",
            "address_id": "ADDR_003",
        },
        {
            "transaction_id": "TXN_004",
            "customer_id": "CUS_004",
            "merchant_id": "MER_001",
            "device_id": "DEV_004",
            "ip_id": "IP_004",
            "address_id": "ADDR_001",
        },
        {
            "transaction_id": "TXN_005",
            "customer_id": "CUS_005",
            "merchant_id": "MER_001",
            "device_id": "DEV_005",
            "ip_id": "IP_005",
            "address_id": "ADDR_005",
        },
    ])


def test_shared_device_finds_related_transaction():
    transactions = make_transactions()

    incident = transactions[
        transactions["transaction_id"] == "TXN_001"
    ]

    related = find_related_transactions(
        transactions,
        incident,
    )

    related_ids = set(
        related["transaction_id"]
    )

    assert "TXN_002" in related_ids


def test_shared_ip_finds_related_transaction():
    transactions = make_transactions()

    incident = transactions[
        transactions["transaction_id"] == "TXN_001"
    ]

    related = find_related_transactions(
        transactions,
        incident,
    )

    related_ids = set(
        related["transaction_id"]
    )

    assert "TXN_003" in related_ids


def test_shared_address_finds_related_transaction():
    transactions = make_transactions()

    incident = transactions[
        transactions["transaction_id"] == "TXN_001"
    ]

    related = find_related_transactions(
        transactions,
        incident,
    )

    related_ids = set(
        related["transaction_id"]
    )

    assert "TXN_004" in related_ids


def test_unrelated_transaction_is_not_linked():
    transactions = make_transactions()

    incident = transactions[
        transactions["transaction_id"] == "TXN_001"
    ]

    related = find_related_transactions(
        transactions,
        incident,
    )

    related_ids = set(
        related["transaction_id"]
    )

    assert "TXN_005" not in related_ids


def test_empty_input_returns_empty_dataframe():
    transactions = make_transactions()

    empty_incident = transactions.iloc[0:0]

    related = find_related_transactions(
        transactions,
        empty_incident,
    )

    assert isinstance(related, pd.DataFrame)
    assert related.empty
