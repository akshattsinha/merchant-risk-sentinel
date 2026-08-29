# Merchant Risk Sentinel

Merchant Risk Sentinel is a fraud detection and investigation project built around a simple idea: detecting fraud is only the first step. A useful system should also help explain what looks suspicious, find related activity, and suggest what an analyst should do next.

The project combines machine learning with behavioral features, transaction relationship analysis, incident detection, root-cause analysis, risk evidence, and merchant response recommendations.

## What the project does

The system takes transaction data through a pipeline that looks roughly like this:

```text
Transaction Data
      |
      v
Feature Engineering
      |
      v
Fraud Prediction
      |
      +-------------------+
      |                   |
      v                   v
Risk Evidence        Fraud Graph
      |                   |
      +---------+---------+
                |
                v
        Incident Detection
                |
                v
        Root Cause Analysis
                |
                v
       Merchant Response
```

The goal is to move beyond a simple `fraud / not fraud` prediction and provide enough context for someone investigating suspicious activity.

## Main features

### Fraud prediction

The project uses a `HistGradientBoostingClassifier` inside a scikit-learn pipeline. Numerical and categorical features are handled separately, with missing-value imputation and one-hot encoding for categorical data.

The trained artifact stores the model pipeline along with the feature definitions used by the predictor.

### Behavioral features

The feature engineering pipeline creates signals from transaction history, including:

- Customer transaction history
- Customer average transaction amount
- Amount compared with customer average
- Merchant transaction history
- Merchant average transaction amount
- Amount compared with merchant average
- Payment-method history
- Device/customer relationships
- IP/customer relationships
- Address/customer relationships
- Account age
- Refund and chargeback ratios
- Transaction timing
- Customer transaction velocity
- Location changes
- Device changes
- IP changes
- A combined behavioral risk-signal count

The historical features are calculated using previous transactions so that the current transaction is not included in its own history.

## Fraud incident analysis

Suspicious transactions can be grouped into broader incident types instead of being investigated in isolation.

The project currently handles incident categories such as:

- `ACCOUNT_TAKEOVER`
- `COORDINATED_ACCOUNT_ABUSE`
- `PAYMENT_VELOCITY_ATTACK`
- `PAYMENT_METHOD_ABUSE`
- `GEOGRAPHIC_ANOMALY`
- `REFUND_ABUSE`
- `AMOUNT_ANOMALY`
- `NEW_ACCOUNT_ATTACK`

The incident engine also assigns severity and produces root-cause explanations based on the available transaction evidence.

## Transaction relationship analysis

Fraud often involves more than one transaction or account. The fraud graph component looks for relationships through shared identifiers such as:

```text
Customer
   |
   +--- Device
   |
   +--- IP
   |
   +--- Address
```

This makes it possible to find transactions connected through a shared device, IP address, or address.

For example:

```text
Transaction A ---- Device X ---- Transaction B
                         |
                         +------ Transaction C
```

These relationships can be used during incident investigation to identify clusters of suspicious activity.

## Risk evidence

The risk evidence component provides structured reasons behind a risk assessment.

Instead of only returning a score, the system can surface signals such as:

- Unusually large transaction amounts
- New accounts
- High transaction velocity
- Location changes
- Device changes
- IP changes
- Shared infrastructure
- Other behavioral anomalies

This is intended to make the model output more useful to an analyst.

## Merchant response

The response engine maps incident severity and risk score to an operational recommendation.

| Risk | Action | Priority |
|---|---|---|
| CRITICAL | `HOLD_AND_INVESTIGATE` | P0 |
| HIGH | `STEP_UP_VERIFICATION` | P1 |
| MEDIUM | `MONITOR` | P2 |
| LOW | `ALLOW` | P3 |

The response can also take transaction-cluster size and estimated exposure into account when generating next steps.

## Live prediction

The live predictor accepts transaction information and returns a result containing:

```json
{
  "fraud_probability": 0.0,
  "risk_score": 0,
  "risk_level": "LOW",
  "recommended_action": "ALLOW"
}
```

The actual values depend on the transaction and the trained model artifact.

## Model evaluation

One of the things I wanted to avoid with this project was evaluating a fraud model using only a random train/test split.

The evaluation pipeline therefore uses a chronological split:

```text
60%  Training
20%  Validation
20%  Held-out Test
```

The model is trained on earlier transactions, the threshold is selected using the validation period, and the final reported metrics come from the later held-out period.

This gives a better approximation of how the model would behave when applied to future transactions.

## Current model results

The current evaluation run produced the following results on the held-out temporal test set:

| Metric | Result |
|---|---:|
| Precision | 81.02% |
| Recall | 68.23% |
| F1 Score | 74.07% |
| PR-AUC | 83.16% |
| ROC-AUC | 97.36% |
| Fraud Rate | 4.96% |

The held-out set contains 10,346 transactions, including 513 fraudulent transactions.

At the selected operating threshold of `0.30`:

```text
True Positives:   350
False Positives:   82
True Negatives:  9751
False Negatives:  163
```

These numbers are from the current generated dataset and should be treated as experimental results rather than production performance.

## Why the threshold is 0.30

A probability threshold of `0.50` is not automatically the best choice for a fraud system.

A missed fraudulent transaction can have a much larger financial impact than the operational cost of investigating a legitimate transaction. Because of that, the evaluation script compares several thresholds using a simple configurable cost model.

For the current prototype:

```text
False-positive investigation cost = ₹500
False-negative exposure = transaction amount
```

The evaluation minimizes:

```text
Expected Loss
=
False Positive Cost
+
False Negative Exposure
```

The threshold is chosen using validation data and then applied to the untouched test set.

For the current run, validation selected:

```text
Operating threshold = 0.30
```

At that threshold on validation data:

```text
Precision = 81.58%
Recall    = 76.05%
F1        = 78.72%
```

## Business impact of threshold optimization

The evaluation also compares the selected threshold with a baseline threshold of `0.50`.

On the current held-out test set:

```text
Baseline expected loss:
₹7,179,482.48

Optimized expected loss:
₹3,978,552.17

Loss reduction:
₹3,200,930.31

Loss reduction:
44.58%
```

The optimized threshold detected 68 additional fraudulent transactions compared with the 0.50 baseline, while producing 63 additional false positives.

The cost model is intentionally simple and configurable. In a real payment system, these costs would need to be estimated from actual investigation costs, fraud losses, recovery rates, and business policies.

## Project structure

```text
merchant-risk-sentinel/
|
├── .github/
│   └── workflows/
│       └── tests.yml
|
├── data/
│   ├── raw/
│   └── processed/
|
├── reports/
│   ├── baseline_metrics.json
│   ├── fraud_model_metadata.json
│   ├── optimized_metrics.json
│   ├── strong_model_metrics.json
│   ├── strong_optimized_metrics.json
│   ├── strong_threshold_analysis.csv
│   └── threshold_analysis.csv
|
├── scripts/
│   └── evaluate_model.py
|
├── src/
│   ├── analysis/
│   │   ├── fraud_graph.py
│   │   ├── incident_engine.py
│   │   ├── response_engine.py
│   │   ├── risk_evidence.py
│   │   └── root_cause_engine.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   │
│   ├── audit/
│   │   ├── __init__.py
│   │   └── audit_logger.py
│   │
│   ├── chatbot/
│   │   ├── __init__.py
│   │   └── fraud_assistant.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── build_model_artifact.py
│   │   ├── live_predictor.py
│   │   ├── optimize_strong_threshold.py
│   │   ├── optimize_threshold.py
│   │   ├── test_live_predictor.py
│   │   ├── train_model.py
│   │   └── train_strong_model.py
│   │
│   ├── dashboard.py
│   ├── generate_data.py
│   └── inspect_data.py
|
├── tests/
│   ├── test_fraud_graph.py
│   ├── test_incident_engine.py
│   ├── test_model_configuration.py
│   ├── test_response_engine.py
│   ├── test_risk_evidence.py
│   └── test_risk_score.py
|
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/akshattsinha/merchant-risk-sentinel.git
cd merchant-risk-sentinel
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Generate the transaction data

The project uses a generated transaction dataset for development and experimentation.

```bash
python src/generate_data.py
```

This creates:

```text
data/raw/transactions.csv
```

## Build the feature dataset

```bash
python src/features/build_features.py
```

This creates:

```text
data/processed/fraud_features.csv
```

## Build the model artifact

```bash
python src/models/build_model_artifact.py
```

The trained artifact is saved as:

```text
reports/fraud_model.joblib
```

and the associated metadata is saved as:

```text
reports/fraud_model_metadata.json
```

## Evaluate the model

Run:

```bash
python scripts/evaluate_model.py
```

This performs the temporal evaluation, threshold comparison, cost optimization, and held-out test evaluation.

The detailed output is saved to:

```text
reports/strong_optimized_metrics.json
```

## Run the tests

Run the complete test suite with:

```bash
pytest -v
```

The current suite contains 38 tests covering:

- Live prediction
- Fraud graph relationships
- Incident severity
- Incident type normalization
- Root-cause analysis
- Merchant response decisions
- Risk evidence
- Risk-score behavior
- Model configuration

Current local result:

```text
38 passed
```

## Continuous integration

The repository includes a GitHub Actions workflow that runs the project checks automatically.

The workflow:

```text
Checkout repository
        |
        v
Set up Python 3.13
        |
        v
Install dependencies
        |
        v
Generate transaction dataset
        |
        v
Build fraud features
        |
        v
Build model artifact
        |
        v
Run pytest
```

This also means the generated dataset and model artifact do not need to be committed to the repository just to run the test suite.

## Docker

Build the containers with:

```bash
docker compose build
```

Start the application with:

```bash
docker compose up
```

## Technology stack

- Python
- Pandas
- NumPy
- scikit-learn
- Joblib
- Flask
- Streamlit
- pytest
- Docker
- Docker Compose
- GitHub Actions

## Testing approach

The test suite is intentionally split across the main parts of the system rather than testing only the ML model.

```text
Model
  |
  +-- Live prediction tests
  |
  +-- Configuration tests
  |
  +-- Risk score tests
  |
  +-- Risk evidence tests
  |
  +-- Fraud graph tests
  |
  +-- Incident tests
  |
  +-- Response tests
```

This helps catch problems in the investigation and decision-making layers as well as the prediction layer.

## Limitations

This is a prototype built with a synthetic transaction dataset.

The reported metrics are therefore useful for demonstrating the engineering and modeling approach, but they should not be interpreted as real-world fraud detection performance.

The financial cost model is also an assumption for experimentation. The ₹500 investigation cost and transaction-amount exposure model would need to be replaced or calibrated using real business data before being used for production decisions.

The system should also have appropriate monitoring, model governance, security controls, human review, and regulatory processes before being deployed in a real payment environment.

## Possible next steps

Some areas I would explore next are:

- Real-time transaction streaming
- A feature store for online features
- Model and data drift monitoring
- Probability calibration
- SHAP-based explanations
- More advanced graph analysis
- Real-time alerting
- Case management for fraud analysts
- Model versioning and experiment tracking
- Feedback from analyst decisions into model retraining
- Production database integration

## Author

**Akshat Sinha**

Computer Science & Engineering

GitHub:  
https://github.com/akshattsinha

## Disclaimer

Merchant Risk Sentinel is an educational and experimental fraud-risk detection project. It is not intended to make autonomous financial decisions in a production payment environment without appropriate validation, monitoring, governance, security controls, regulatory review, and human oversight.
