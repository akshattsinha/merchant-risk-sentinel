import os
import json
import pandas as pd

INPUT_FILE = "reports/incident_summary.json"
OUTPUT_FILE = "reports/merchant_response_actions.csv"


def determine_action(incident):
    severity = incident["severity"]
    risk_score = float(
        incident["risk_score"]
    )

    transaction_count = int(
        incident["transaction_count"]
    )

    estimated_exposure = float(
        incident["estimated_exposure"]
    )

    if (
        severity == "CRITICAL"
        or risk_score >= 90
    ):
        action = "HOLD_AND_INVESTIGATE"
        priority = "P0"
        action_title = "Hold affected transactions"
        explanation = (
            "The incident has critical risk indicators "
            "and requires immediate merchant investigation."
        )
        next_steps = [
            "Hold affected transactions",
            "Review linked customer accounts",
            "Review shared device and IP evidence",
            "Create a fraud investigation case",
            "Escalate to fraud operations"
        ]

    elif (
        severity == "HIGH"
        or risk_score >= 75
    ):
        action = "STEP_UP_VERIFICATION"
        priority = "P1"
        action_title = "Require additional verification"
        explanation = (
            "The incident has strong risk indicators. "
            "Additional verification is recommended before "
            "allowing further high-risk activity."
        )
        next_steps = [
            "Require step-up verification",
            "Monitor linked transactions",
            "Review customer and device history",
            "Create investigation case if activity continues"
        ]

    elif (
        severity == "MEDIUM"
        or risk_score >= 50
    ):
        action = "MONITOR"
        priority = "P2"
        action_title = "Increase monitoring"
        explanation = (
            "The incident contains suspicious signals "
            "but does not currently justify an automatic hold."
        )
        next_steps = [
            "Continue monitoring",
            "Increase scrutiny on related transactions",
            "Review if additional suspicious activity appears"
        ]

    else:
        action = "ALLOW"
        priority = "P3"
        action_title = "Allow transaction"
        explanation = (
            "Current evidence does not justify "
            "restrictive action."
        )
        next_steps = [
            "Allow transaction",
            "Continue normal monitoring"
        ]

    if transaction_count >= 20:
        next_steps.insert(
            0,
            "Review incident-level transaction cluster"
        )

    if estimated_exposure >= 500000:
        next_steps.insert(
            0,
            "Prioritize financial exposure review"
        )

    return {
        "recommended_action": action,
        "priority": priority,
        "action_title": action_title,
        "explanation": explanation,
        "next_steps": next_steps
    }


def main():

    print(
        "Loading incident data..."
    )

    with open(
        INPUT_FILE,
        "r"
    ) as file:
        data = json.load(file)

    incidents = data.get(
        "incidents",
        []
    )

    print(
        f"Incidents loaded: {len(incidents)}"
    )

    responses = []

    for incident in incidents:

        response = determine_action(
            incident
        )

        root_causes = json.loads(
            incident.get(
                "root_causes",
                "[]"
            )
        )

        response_record = {
            "incident_id":
                incident["incident_id"],

            "incident_type":
                incident["incident_type"],

            "severity":
                incident["severity"],

            "risk_score":
                incident["risk_score"],

            "transaction_count":
                incident["transaction_count"],

            "customer_count":
                incident["customer_count"],

            "device_count":
                incident["device_count"],

            "ip_count":
                incident["ip_count"],

            "estimated_exposure":
                incident["estimated_exposure"],

            "recommended_action":
                response[
                    "recommended_action"
                ],

            "priority":
                response["priority"],

            "action_title":
                response["action_title"],

            "explanation":
                response["explanation"],

            "next_steps":
                json.dumps(
                    response["next_steps"]
                ),

            "root_causes":
                json.dumps(
                    root_causes
                )
        }

        responses.append(
            response_record
        )

    results = pd.DataFrame(
        responses
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "===== MERCHANT RESPONSE PLAN ====="
    )

    if len(results) == 0:
        print(
            "No incidents require action."
        )
    else:

        for _, row in results.iterrows():

            print()
            print(
                "----------------------------------------"
            )

            print(
                f"Incident: "
                f"{row['incident_id']}"
            )

            print(
                f"Type: "
                f"{row['incident_type']}"
            )

            print(
                f"Severity: "
                f"{row['severity']}"
            )

            print(
                f"Risk Score: "
                f"{row['risk_score']}/100"
            )

            print(
                f"Exposure: ₹"
                f"{row['estimated_exposure']:,.2f}"
            )

            print(
                f"Priority: "
                f"{row['priority']}"
            )

            print(
                f"RECOMMENDED ACTION: "
                f"{row['action_title']}"
            )

            print(
                f"WHY: "
                f"{row['explanation']}"
            )

            next_steps = json.loads(
                row["next_steps"]
            )

            print(
                "NEXT STEPS:"
            )

            for number, step in enumerate(
                next_steps,
                start=1
            ):
                print(
                    f"  {number}. {step}"
                )

    print()
    print(
        f"Response actions saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()