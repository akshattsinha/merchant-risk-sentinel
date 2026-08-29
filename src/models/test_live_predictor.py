from live_predictor import LiveRiskPredictor


def main():

    predictor = LiveRiskPredictor()

    transaction = {
        "customer_id": "CUS_000001",
        "merchant_id": "MER_000001",
        "amount": 85000,
        "timestamp": "2026-07-24 12:00:00",
        "payment_method": "UPI",
        "device_id": "NEW_DEVICE_001",
        "ip_id": "NEW_IP_001",
        "address_id": "NEW_ADDRESS_001",
        "account_age_days": 3,
        "location": "Delhi"
    }

    result = predictor.predict(transaction)

    print()
    print("===== LIVE RISK PREDICTION =====")
    print(
        f"Fraud probability: "
        f"{result['fraud_probability']:.4f}"
    )
    print(
        f"Risk score: "
        f"{result['risk_score']}/100"
    )
    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )
    print(
        f"Recommended action: "
        f"{result['recommended_action']}"
    )


if __name__ == "__main__":
    main()