import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_TRANSACTIONS = 50000
NUM_CUSTOMERS = 12000
NUM_MERCHANTS = 50
NUM_INCIDENTS = 20

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transactions.csv")

PAYMENT_METHODS = [
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET"
]

LOCATIONS = [
    "Delhi",
    "Mumbai",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Lucknow"
]

INCIDENT_TYPES = [
    "coordinated_account_abuse",
    "payment_velocity_attack",
    "account_takeover",
    "payment_method_abuse",
    "geographic_anomaly",
    "amount_anomaly",
    "refund_abuse",
    "new_account_attack"
]


def random_id(prefix, number):
    return f"{prefix}_{number:06d}"


def create_merchants():
    merchants = []

    for i in range(NUM_MERCHANTS):
        merchants.append({
            "merchant_id": random_id("MER", i + 1),
            "baseline_fraud_rate": round(
                np.random.uniform(0.003, 0.012),
                4
            ),
            "average_order_value": round(
                np.random.uniform(500, 10000),
                2
            )
        })

    return merchants


def create_customers():
    customers = []

    for i in range(NUM_CUSTOMERS):
        customers.append({
            "customer_id": random_id("CUS", i + 1),
            "device_id": random_id(
                "DEV",
                random.randint(1, 15000)
            ),
            "ip_id": random_id(
                "IP",
                random.randint(1, 18000)
            ),
            "address_id": random_id(
                "ADDR",
                random.randint(1, 14000)
            ),
            "account_age_days": random.randint(
                30,
                1500
            ),
            "location": random.choice(
                LOCATIONS
            )
        })

    return customers


def generate_base_transactions(
    customers,
    merchants
):
    transactions = []

    start_time = (
        datetime.now()
        - timedelta(days=30)
    )

    for i in range(BASE_TRANSACTIONS):

        customer = random.choice(customers)
        merchant = random.choice(merchants)

        amount = np.random.lognormal(
            mean=np.log(
                merchant["average_order_value"]
            ),
            sigma=0.65
        )

        amount = round(
            min(max(amount, 100), 100000),
            2
        )

        timestamp = (
            start_time
            + timedelta(
                seconds=random.randint(
                    0,
                    30 * 24 * 60 * 60
                )
            )
        )

        transactions.append({
            "transaction_id": random_id(
                "TXN",
                i + 1
            ),
            "customer_id": customer["customer_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": amount,
            "timestamp": timestamp,
            "payment_method": random.choice(
                PAYMENT_METHODS
            ),
            "device_id": customer["device_id"],
            "ip_id": customer["ip_id"],
            "address_id": customer["address_id"],
            "account_age_days": customer[
                "account_age_days"
            ],
            "location": customer["location"],

            "refund_count": 0,
            "refund_amount": 0.0,
            "chargeback_count": 0,
            "chargeback_amount": 0.0,

            "is_refund": 0,
            "is_chargeback": 0,

            "is_fraud": 0,
            "fraud_type": "normal",

            "incident_id": None,
            "incident_type": None,
            "incident_severity": None
        })

    return transactions


def inject_legitimate_anomalies(
    transactions
):
    legitimate_indices = [
        i
        for i, transaction in enumerate(
            transactions
        )
        if transaction["is_fraud"] == 0
    ]

    high_value_count = int(
        len(legitimate_indices) * 0.012
    )

    high_value_indices = random.sample(
        legitimate_indices,
        high_value_count
    )

    for index in high_value_indices:

        transaction = transactions[index]

        transaction["amount"] = round(
            np.random.uniform(
                30000,
                90000
            ),
            2
        )

        transaction["fraud_type"] = (
            "legitimate_high_value"
        )

    remaining = [
        i
        for i in legitimate_indices
        if i not in high_value_indices
    ]

    location_count = int(
        len(remaining) * 0.006
    )

    location_indices = random.sample(
        remaining,
        location_count
    )

    for index in location_indices:

        transaction = transactions[index]

        current_location = transaction[
            "location"
        ]

        transaction["location"] = random.choice([
            location
            for location in LOCATIONS
            if location != current_location
        ])

        transaction["fraud_type"] = (
            "legitimate_geographic_change"
        )

    return transactions


def create_incidents(merchants):

    start_time = (
        datetime.now()
        - timedelta(days=30)
    )

    incidents = []

    selected_merchants = [
        random.choice(merchants)
        for _ in range(NUM_INCIDENTS)
    ]

    for number, merchant in enumerate(
        selected_merchants,
        start=1
    ):

        incident_type = random.choice(
            INCIDENT_TYPES
        )

        incident_start = (
            start_time
            + timedelta(
                hours=random.randint(
                    12,
                    29 * 24
                )
            )
        )

        duration = random.randint(
            1,
            4
        )

        severity = random.choices(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            ],
            weights=[
                15,
                30,
                40,
                15
            ],
            k=1
        )[0]

        incidents.append({
            "incident_id":
                f"INC_{number:03d}",

            "merchant_id":
                merchant["merchant_id"],

            "incident_type":
                incident_type,

            "start_time":
                incident_start,

            "end_time":
                incident_start
                + timedelta(
                    hours=duration
                ),

            "severity":
                severity
        })

    return incidents


def add_incident_transactions(
    transactions,
    customers,
    incidents
):

    next_id = len(transactions) + 1

    for incident in incidents:

        transaction_count = random.randint(
            60,
            140
        )

        affected_customers = random.sample(
            customers,
            random.randint(
                15,
                35
            )
        )

        shared_device = (
            f"INC_DEVICE_"
            f"{incident['incident_id']}"
        )

        shared_ip = (
            f"INC_IP_"
            f"{incident['incident_id']}"
        )

        shared_address = (
            f"INC_ADDR_"
            f"{incident['incident_id']}"
        )

        for _ in range(
            transaction_count
        ):

            customer = random.choice(
                affected_customers
            )

            total_seconds = int(
                (
                    incident["end_time"]
                    - incident["start_time"]
                ).total_seconds()
            )

            timestamp = (
                incident["start_time"]
                + timedelta(
                    seconds=random.randint(
                        0,
                        total_seconds
                    )
                )
            )

            incident_type = (
                incident["incident_type"]
            )

            device_id = customer[
                "device_id"
            ]

            ip_id = customer[
                "ip_id"
            ]

            address_id = customer[
                "address_id"
            ]

            account_age = customer[
                "account_age_days"
            ]

            location = customer[
                "location"
            ]

            amount = round(
                np.random.uniform(
                    1000,
                    30000
                ),
                2
            )

            payment_method = random.choice(
                PAYMENT_METHODS
            )

            refund_count = 0
            refund_amount = 0.0
            chargeback_count = 0
            chargeback_amount = 0.0

            is_refund = 0
            is_chargeback = 0

            # --------------------------------
            # COORDINATED ACCOUNT ABUSE
            # --------------------------------

            if incident_type == (
                "coordinated_account_abuse"
            ):

                device_id = shared_device
                ip_id = shared_ip
                address_id = shared_address

                account_age = random.randint(
                    1,
                    14
                )

            # --------------------------------
            # PAYMENT VELOCITY ATTACK
            # --------------------------------

            elif incident_type == (
                "payment_velocity_attack"
            ):

                amount = round(
                    np.random.uniform(
                        5000,
                        30000
                    ),
                    2
                )

            # --------------------------------
            # ACCOUNT TAKEOVER
            # --------------------------------

            elif incident_type == (
                "account_takeover"
            ):

                device_id = (
                    f"NEW_DEVICE_"
                    f"{incident['incident_id']}"
                )

                ip_id = (
                    f"NEW_IP_"
                    f"{incident['incident_id']}"
                )

                amount = round(
                    np.random.uniform(
                        15000,
                        60000
                    ),
                    2
                )

            # --------------------------------
            # PAYMENT METHOD ABUSE
            # --------------------------------

            elif incident_type == (
                "payment_method_abuse"
            ):

                payment_method = random.choices(
                    PAYMENT_METHODS,
                    weights=[
                        2,
                        12,
                        2,
                        1
                    ],
                    k=1
                )[0]

            # --------------------------------
            # GEOGRAPHIC ANOMALY
            # --------------------------------

            elif incident_type == (
                "geographic_anomaly"
            ):

                location = random.choice([
                    loc
                    for loc in LOCATIONS
                    if loc != customer[
                        "location"
                    ]
                ])

            # --------------------------------
            # AMOUNT ANOMALY
            # --------------------------------

            elif incident_type == (
                "amount_anomaly"
            ):

                amount = round(
                    np.random.uniform(
                        60000,
                        100000
                    ),
                    2
                )

            # --------------------------------
            # REFUND ABUSE
            # --------------------------------

            elif incident_type == (
                "refund_abuse"
            ):

                amount = round(
                    np.random.uniform(
                        5000,
                        25000
                    ),
                    2
                )

                is_refund = 1

                refund_count = random.randint(
                    2,
                    8
                )

                refund_amount = round(
                    amount
                    * random.uniform(
                        0.5,
                        1.0
                    ),
                    2
                )

                if random.random() < 0.25:
                    is_chargeback = 1
                    chargeback_count = 1
                    chargeback_amount = round(
                        amount
                        * random.uniform(
                            0.5,
                            1.0
                        ),
                        2
                    )

            # --------------------------------
            # NEW ACCOUNT ATTACK
            # --------------------------------

            elif incident_type == (
                "new_account_attack"
            ):

                account_age = random.randint(
                    1,
                    7
                )

                amount = round(
                    np.random.uniform(
                        3000,
                        25000
                    ),
                    2
                )

            transactions.append({

                "transaction_id":
                    random_id(
                        "TXN",
                        next_id
                    ),

                "customer_id":
                    customer[
                        "customer_id"
                    ],

                "merchant_id":
                    incident[
                        "merchant_id"
                    ],

                "amount":
                    amount,

                "timestamp":
                    timestamp,

                "payment_method":
                    payment_method,

                "device_id":
                    device_id,

                "ip_id":
                    ip_id,

                "address_id":
                    address_id,

                "account_age_days":
                    account_age,

                "location":
                    location,

                "refund_count":
                    refund_count,

                "refund_amount":
                    refund_amount,

                "chargeback_count":
                    chargeback_count,

                "chargeback_amount":
                    chargeback_amount,

                "is_refund":
                    is_refund,

                "is_chargeback":
                    is_chargeback,

                "is_fraud":
                    1,

                "fraud_type":
                    incident_type,

                "incident_id":
                    incident[
                        "incident_id"
                    ],

                "incident_type":
                    incident_type,

                "incident_severity":
                    incident[
                        "severity"
                    ]
            })

            next_id += 1

    return transactions


def inject_individual_fraud(
    transactions
):

    legitimate_indices = [
        i
        for i, transaction in enumerate(
            transactions
        )
        if (
            transaction["is_fraud"] == 0
            and transaction[
                "fraud_type"
            ] not in [
                "legitimate_high_value",
                "legitimate_geographic_change"
            ]
        )
    ]

    fraud_count = int(
        len(legitimate_indices)
        * 0.018
    )

    selected = random.sample(
        legitimate_indices,
        fraud_count
    )

    patterns = [
        "individual_velocity_abuse",
        "individual_new_account",
        "individual_amount_anomaly",
        "individual_device_reuse"
    ]

    for index in selected:

        transaction = transactions[
            index
        ]

        pattern = random.choice(
            patterns
        )

        transaction[
            "is_fraud"
        ] = 1

        transaction[
            "fraud_type"
        ] = pattern

        if pattern == (
            "individual_velocity_abuse"
        ):

            transaction[
                "amount"
            ] = round(
                np.random.uniform(
                    10000,
                    50000
                ),
                2
            )

        elif pattern == (
            "individual_new_account"
        ):

            transaction[
                "account_age_days"
            ] = random.randint(
                1,
                7
            )

        elif pattern == (
            "individual_amount_anomaly"
        ):

            transaction[
                "amount"
            ] = round(
                np.random.uniform(
                    50000,
                    100000
                ),
                2
            )

        elif pattern == (
            "individual_device_reuse"
        ):

            transaction[
                "device_id"
            ] = (
                "SHARED_DEVICE_"
                f"{random.randint(1, 50)}"
            )

    return transactions


def add_behavioral_history(
    df
):
    """
    Derive historical behavioral features
    from transaction history.

    These features are calculated without
    using fraud labels.
    """

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # ------------------------------------
    # TRANSACTION VELOCITY
    # ------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["transaction_count_last_hour"] = (
        df.groupby(
            "customer_id"
        )["transaction_id"]
        .transform(
            lambda s: s.rolling(
                window=20,
                min_periods=1
            ).count()
        )
    )

    # ------------------------------------
    # DEVICE REUSE
    # ------------------------------------

    df["device_customer_count"] = (
        df.groupby(
            "device_id"
        )["customer_id"]
        .transform("nunique")
    )

    # ------------------------------------
    # IP REUSE
    # ------------------------------------

    df["ip_customer_count"] = (
        df.groupby(
            "ip_id"
        )["customer_id"]
        .transform("nunique")
    )

    # ------------------------------------
    # ADDRESS REUSE
    # ------------------------------------

    df["address_customer_count"] = (
        df.groupby(
            "address_id"
        )["customer_id"]
        .transform("nunique")
    )

    # ------------------------------------
    # PAYMENT METHOD FREQUENCY
    # ------------------------------------

    df["payment_method_frequency"] = (
        df.groupby(
            [
                "merchant_id",
                "payment_method"
            ]
        )["transaction_id"]
        .transform("count")
    )

    # ------------------------------------
    # MERCHANT AVERAGE AMOUNT
    # ------------------------------------

    df["merchant_average_amount"] = (
        df.groupby(
            "merchant_id"
        )["amount"]
        .transform("mean")
    )

    # ------------------------------------
    # AMOUNT DEVIATION
    # ------------------------------------

    df["amount_to_merchant_average"] = (
        df["amount"]
        / df["merchant_average_amount"]
    )

    # ------------------------------------
    # REFUND RATIO
    # ------------------------------------

    merchant_transactions = (
        df.groupby(
            "merchant_id"
        )["transaction_id"]
        .transform("count")
    )

    merchant_refunds = (
        df.groupby(
            "merchant_id"
        )["is_refund"]
        .transform("sum")
    )

    df["merchant_refund_ratio"] = (
        merchant_refunds
        / merchant_transactions
    )

    # ------------------------------------
    # CHARGEBACK RATIO
    # ------------------------------------

    merchant_chargebacks = (
        df.groupby(
            "merchant_id"
        )["is_chargeback"]
        .transform("sum")
    )

    df["merchant_chargeback_ratio"] = (
        merchant_chargebacks
        / merchant_transactions
    )

    return df


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "Creating merchants..."
    )

    merchants = create_merchants()

    print(
        "Creating customers..."
    )

    customers = create_customers()

    print(
        "Generating base transactions..."
    )

    transactions = (
        generate_base_transactions(
            customers,
            merchants
        )
    )

    print(
        "Adding legitimate anomalies..."
    )

    transactions = (
        inject_legitimate_anomalies(
            transactions
        )
    )

    print(
        "Creating fraud incidents..."
    )

    incidents = create_incidents(
        merchants
    )

    print(
        f"Created {len(incidents)} fraud incidents."
    )

    print(
        "Injecting different incident types..."
    )

    transactions = (
        add_incident_transactions(
            transactions,
            customers,
            incidents
        )
    )

    print(
        "Injecting individual fraud patterns..."
    )

    transactions = (
        inject_individual_fraud(
            transactions
        )
    )

    print(
        "Calculating behavioral history..."
    )

    df = pd.DataFrame(
        transactions
    )

    df = add_behavioral_history(
        df
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Dataset created successfully."
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Fraud transactions: "
        f"{df['is_fraud'].sum():,}"
    )

    print(
        f"Fraud rate: "
        f"{df['is_fraud'].mean() * 100:.2f}%"
    )

    print()
    print(
        "Fraud type distribution:"
    )

    print(
        df[
            "fraud_type"
        ].value_counts()
    )

    print()
    print(
        "Incident type distribution:"
    )

    print(
        df[
            df["incident_id"].notna()
        ]["incident_type"]
        .value_counts()
    )

    print()
    print(
        "Number of incidents:"
    )

    print(
        df["incident_id"]
        .nunique()
    )

    print()
    print(
        "Refund transactions:"
    )

    print(
        df["is_refund"].sum()
    )

    print()
    print(
        "Chargeback transactions:"
    )

    print(
        df["is_chargeback"].sum()
    )

    print()
    print(
        "Saved to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()