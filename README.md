# Merchant Risk Sentinel

### Fraud Detection & Incident Response Platform for Merchants

Merchant Risk Sentinel is an end-to-end fraud risk platform designed to help merchants **detect suspicious transactions, understand why they are risky, investigate connected activity, estimate financial exposure, and make human-reviewed decisions**.

Unlike a transaction-only fraud classifier, the system combines machine-learning probability with deterministic behavioral evidence and relationship analysis, then closes the loop through analyst feedback and governed continual learning.

---

## Why This Project

Fraud detection is not only a classification problem.

A useful merchant risk system must answer four questions:

1. **Is this transaction suspicious?**
2. **Why is it suspicious?**
3. **What other entities or transactions are connected to the risk?**
4. **What should the merchant do next?**

Merchant Risk Sentinel addresses these questions through a layered risk architecture:

```text
Transaction
     ↓
Point-in-Time Behavioral Features
     ↓
ML Fraud Probability
     ↓
Deterministic Evidence / Risk Engine
     ↓
Relationship Analysis
     ↓
Risk Score + Risk Level + Recommendation
     ↓
Incident / Financial Exposure
     ↓
Human Analyst Decision
     ↓
Confirmed Feedback
     ↓
SQLite Feedback Store
     ↓
Governed Continual Learning
     ↓
Candidate Model Evaluation
     ↓
Promotion Gate
     ↓
Versioned Active Model
```

---

## Core Capabilities

### 1. ML Fraud Detection

- HistGradientBoosting fraud classifier
- 43 model features
- Fraud probability scoring
- Operating threshold: `0.30`
- Categorical handling for features such as payment method and location
- Exact artifact feature schema enforced by the live predictor

### 2. Behavioral Risk Analysis

The system derives point-in-time behavioral signals including:

- Transaction amount
- Customer historical transaction behavior
- Merchant historical transaction behavior
- Amount-to-customer and amount-to-merchant ratios
- Transaction velocity
- Device/customer relationships
- IP/customer relationships
- Address/customer relationships
- Payment-method frequency
- Refund behavior
- Chargeback behavior
- Account age
- Location changes
- Device changes
- IP changes
- Time-of-day and day-of-week behavior
- High and very-high velocity flags
- Behavioral risk signal count

Historical values are calculated using information available **before the transaction being evaluated**, preventing future transaction information from being used as a prediction feature.

### 3. Deterministic Evidence Engine

ML probability is supplemented by bounded, explainable evidence signals.

Current evidence weights include:

| Evidence | Weight |
|---|---:|
| Device change | 3 |
| IP change | 4 |
| Location change | 4 |
| Amount anomaly | 8 |
| Merchant amount anomaly | 4 |
| High velocity | 8 |
| Very high velocity | 12 |
| Shared device | 8 |
| Shared IP | 8 |
| Shared address | 6 |

The dashboard risk fusion keeps the ML model primary while allowing deterministic evidence to make a bounded adjustment.

### 4. Relationship Evidence

The platform investigates connections across:

```text
Customer ↔ Device
Customer ↔ IP
Customer ↔ Address
Merchant ↔ Transaction behavior
```

This allows an analyst to distinguish a suspicious transaction from a potentially suspicious **network of related activity**.

### 5. Human-in-the-Loop Investigation

The AI does not make the final operational decision automatically.

Analysts can review a transaction and select:

- `ALLOW`
- `REVIEW`
- `HOLD`

Investigation outcomes can be:

- `CONFIRMED_FRAUD`
- `CONFIRMED_LEGITIMATE`
- `INCONCLUSIVE`

Feedback is persisted to SQLite and becomes eligible for future governed retraining.

### 6. Continual Learning with Promotion Governance

The system does **not** retrain after every transaction.

The current workflow requires a minimum of **10 confirmed feedback labels** before retraining.

```text
Analyst Feedback
      ↓
SQLite
      ↓
10-label threshold
      ↓
Training dataset
      ↓
Chronological 80/20 holdout
      ↓
Candidate model
      ↓
Candidate metrics
      ↓
Active-model metrics
      ↓
Promotion gate
      ↓
Promote / Reject
```

The promotion gate is designed to reject a candidate only when **both PR-AUC and F1 materially degrade beyond the configured tolerance**.

### 7. Model Versioning & Atomic Promotion

Promoted models are versioned using identifiers such as:

```text
v20260903164428
```

The active model is recorded in:

```text
reports/active_model.json
```

Versioned artifacts are stored under:

```text
reports/model_versions/<version>/
```

Promotion uses atomic file replacement so the live artifact is not left partially updated.

### 8. Maintenance Worker

Docker Compose includes a dedicated maintenance service that executes the continual-learning worker every six hours.

It can skip retraining safely when the feedback threshold has not been reached.

### 9. API + Dashboard

- FastAPI backend
- Streamlit analyst dashboard
- Live prediction endpoint
- Feedback endpoint
- Learning/retraining endpoint
- Learning status endpoint
- Model metadata endpoint
- Dockerized deployment

---

## Risk Decision Model

The live API exposes four risk levels:

| Fraud Probability | Risk Level |
|---:|---|
| `< 0.10` | LOW |
| `0.10 – 0.39` | MEDIUM |
| `0.40 – 0.74` | HIGH |
| `>= 0.75` | CRITICAL |

The dashboard additionally combines ML probability with bounded deterministic evidence to produce the final operational risk decision.

Dashboard action thresholds are:

| Final Score | Level | Action |
|---:|---|---|
| `>= 80` | CRITICAL | HOLD |
| `>= 60` | HIGH | HOLD |
| `>= 40` | MEDIUM | REVIEW |
| `>= 20` | LOW | MONITOR |
| `< 20` | LOW | ALLOW |

The distinction between **model probability** and **operational risk decision** is intentional: the ML model estimates fraud likelihood, while the risk engine incorporates explainable behavioral evidence for merchant operations.

---

## Model Evaluation

The initial model was evaluated before continual-learning validation and achieved approximately:

```text
ROC-AUC: 0.9716
PR-AUC:  0.8081
Threshold: 0.30
Features: 43
```

The continual-learning pipeline was subsequently exercised using controlled analyst-feedback test cases.

During the promotion test:

| Metric | Active Model | Candidate Model |
|---|---:|---:|
| ROC-AUC | 0.9731 | 0.9628 |
| PR-AUC | 0.8197 | 0.8456 |
| F1 | 0.7014 | 0.7723 |
|

The candidate was promoted because PR-AUC and F1 improved and the configured promotion gate was not triggered.

The promoted version was:

```text
v20260903164428
```

### Important validation note

The continual-learning promotion test used controlled `CL_TEST_*` feedback records to exercise the retraining pipeline. Those synthetic feedback records were subsequently removed from the feedback database. The promoted artifact remains available as the validated continual-learning model version.

These test labels should therefore be described as **controlled validation data**, not production merchant feedback.

---

## Example Live Prediction

A controlled existing-customer test was used to validate the behavioral pipeline.

The transaction used a new device while keeping the customer's historical context, merchant, amount, IP, address, location, and payment method.

The live API successfully returned:

```text
HTTP 200
Model: v20260903164428
Features: 43
Risk level: LOW
Recommendation: MONITOR
Behavioral signal: Device change
```

The response contained explainable evidence including:

```text
Device change → HIGH severity evidence
```

This demonstrates the separation between:

- ML fraud probability
- behavioral evidence
- operational risk decision
- human analyst action

---

## API Endpoints

### Health

```http
GET /health
```

Returns service and model health.

### Root

```http
GET /
```

Returns API information.

### Model Information

```http
GET /model-info
```

Returns the active model version, feature count, threshold, learning mode, and risk levels.

### Fraud Prediction

```http
POST /predict
```

Accepts a transaction and returns:

- fraud probability
- risk score
- risk level
- recommendation
- behavioral features
- evidence
- relationship evidence
- model version
- human-in-the-loop options

### Analyst Feedback

```http
POST /feedback
```

Stores analyst-confirmed outcomes and associated transaction/features for future learning.

### Retraining

```http
POST /learning/retrain
```

Runs the governed retraining workflow when the feedback threshold is met.

### Learning Status

```http
GET /learning/status
```

Returns:

- feedback count
- confirmed fraud count
- confirmed legitimate count
- AI acceptance/override statistics
- agreement rate
- active model version
- retraining threshold
- learning readiness

---

## Project Structure

```text
merchant-risk-sentinel/
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── models/
│   │   ├── build_model_artifact.py
│   │   ├── live_predictor.py
│   │   └── test_live_predictor.py
│   │
│   ├── learning/
│   │   └── continual_learner.py
│   │
│   └── dashboard/
│       └── ...
│
├── data/
│   ├── processed/
│   │   └── fraud_features.csv
│   └── feedback/
│       └── feedback.db
│
├── reports/
│   ├── fraud_model.joblib
│   ├── fraud_model_metadata.json
│   ├── active_model.json
│   └── model_versions/
│       └── <version>/
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML | scikit-learn |
| Model | HistGradientBoostingClassifier |
| Data Processing | Pandas / NumPy |
| API | FastAPI |
| Dashboard | Streamlit |
| Persistence | SQLite |
| Model Serialization | Joblib |
| Testing | Pytest |
| Deployment | Docker / Docker Compose |
| Model Governance | Custom promotion/versioning pipeline |

---

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd merchant-risk-sentinel
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 5. Run the dashboard

```bash
streamlit run src/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

The default local interfaces are:

```text
API:       http://localhost:8000
Dashboard: http://localhost:8501
```

---

## Running with Docker Compose

Build and start the complete stack:

```bash
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

The stack contains:

```text
merchant-risk-api
merchant-risk-dashboard
merchant-risk-maintenance
```

Stop the stack:

```bash
docker compose down
```

---

## Testing

Run the complete test suite:

```bash
pytest -q
```

Current validated result:

```text
44 passed
```

The project currently emits NumPy/joblib deprecation warnings during some live-predictor tests. These warnings do not cause test failures.

---

## Continual Learning Workflow

The continual-learning implementation is intentionally conservative.

### Feedback collection

Each analyst decision records:

- transaction ID
- label
- ground truth
- AI recommendation
- human decision
- final decision
- reason
- investigation notes
- transaction context
- behavioral features
- model version
- prediction information when available

### Retraining threshold

The default threshold is:

```text
10 confirmed labels
```

Below the threshold, retraining is skipped.

Example:

```json
{
  "status": "skipped",
  "reason": "Need 10 feedback labels; have 2."
}
```

### Chronological validation

Training rows are ordered using their transaction timestamps before the 80/20 holdout is created.

The timestamp is used for **ordering only** and is not passed to the model as one of the 43 model features.

This is important for fraud systems because a random split can allow temporal leakage and produce an unrealistically optimistic estimate of future performance.

### Candidate evaluation

A candidate model is compared against the currently active model on the same chronological holdout.

The candidate is promoted only if it satisfies the configured promotion rule.

### Versioned promotion

A successful promotion creates:

```text
reports/model_versions/<version>/fraud_model.joblib
reports/model_versions/<version>/metadata.json
```

and updates:

```text
reports/fraud_model.joblib
reports/fraud_model_metadata.json
reports/active_model.json
```

The active artifact and promoted registry artifact were independently verified to have matching SHA-256 hashes for the validated version.

---

## Model Governance Principles

Merchant Risk Sentinel treats model deployment as a governed process rather than simply overwriting a model file.

Key controls include:

- Human-confirmed labels
- Minimum feedback threshold
- Chronological validation
- Candidate-vs-active evaluation
- Promotion tolerance
- Model versioning
- Active model metadata
- Atomic promotion
- File-based retraining lock
- Registry retention
- Live model reload support
- Maintenance status reporting

This architecture is intended to reduce the risk of silently deploying a degraded model.

---

## Fraud Risk → Merchant Action

The platform translates model output into an operational workflow:

```text
LOW
  ↓
MONITOR / ALLOW

MEDIUM
  ↓
REVIEW

HIGH
  ↓
HOLD

CRITICAL
  ↓
HOLD
```

The final action remains reviewable by a human analyst.

---

## Key Design Decisions

### ML + deterministic evidence

Pure ML probability is useful but not sufficient for investigation. The deterministic layer provides transparent reasons for risk signals.

### Point-in-time features

Historical behavioral features are calculated from information available before the transaction being scored.

### Bounded evidence adjustment

Deterministic evidence can influence the operational score without completely replacing the ML model.

### Human-in-the-loop

Analysts can override AI recommendations and provide structured outcomes that feed the learning pipeline.

### Governed continual learning

The model does not automatically retrain after every transaction. Feedback must reach a threshold, a candidate must be evaluated, and the promotion gate must pass.

### Versioned models

Every promoted continual-learning model receives a unique version so model evolution can be audited.

---

## Limitations & Future Improvements

This is a prototype/portfolio-grade risk platform rather than a production payment processor.

Potential next improvements include:

- Larger and more representative labeled fraud datasets
- Better calibration of predicted probabilities
- Cost-sensitive evaluation based on merchant loss
- Precision/recall monitoring by merchant and fraud type
- Model drift detection
- Feature drift monitoring
- Automated rollback to a previous model version
- Stronger immutable audit logs
- Role-based analyst access
- Authentication and authorization
- Distributed feedback storage
- Production-grade job scheduling
- Real payment-gateway integrations
- Alerting and notification workflows
- More extensive temporal backtesting
- Shadow evaluation before promotion

---

## What This Project Demonstrates

Merchant Risk Sentinel demonstrates practical ML engineering beyond model training:

**Fraud Detection**

Builds a supervised fraud model using behavioral transaction features.

**Risk Engineering**

Combines probabilistic ML output with deterministic evidence.

**Explainability**

Surfaces concrete risk factors and relationship evidence to analysts.

**Human-in-the-Loop AI**

Allows analysts to accept, override, and classify AI decisions.

**Continual Learning**

Uses confirmed feedback to train candidate models after a defined threshold.

**Model Governance**

Evaluates candidates against the active model before promotion.

**Temporal Validation**

Uses chronological holdout evaluation instead of relying only on random splitting.

**MLOps**

Uses versioned artifacts, atomic promotion, locks, metadata, and maintenance workers.

**Production-Oriented Architecture**

Runs API, dashboard, and maintenance services through Docker Compose.

---

## Validation Snapshot

```text
──────────────────────────────────────────────
 MERCHANT RISK SENTINEL — VALIDATION SNAPSHOT
──────────────────────────────────────────────

Automated tests             44 passed
Model features              43
Model type                  HistGradientBoosting
Operating threshold         0.30

Initial model ROC-AUC       0.9716
Initial model PR-AUC        0.8081

Candidate ROC-AUC           0.9628
Candidate PR-AUC            0.8456
Candidate F1                0.7723

Promotion result             PROMOTED
Active version               v20260903164428

API health                  HEALTHY
Model loaded                YES
Human-in-the-loop            YES
Continual learning           ENABLED
Docker services              3 / 3 UP

──────────────────────────────────────────────
```

---

## Author

**Akshat Sinha**

B.Tech — Computer Science & Engineering

Built as an AI risk engineering project focused on fraud detection, explainability, human-in-the-loop decision making, and governed continual learning.

---

## License

Add the project's chosen license here before publishing the repository.
