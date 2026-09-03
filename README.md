# Merchant Risk Sentinel

### Fraud detection is only useful when it leads to a better decision.

Merchant Risk Sentinel is an end-to-end **fraud detection and investigation platform** built around a simple idea:

> A fraud model should not just say *“this looks suspicious.”*  
> It should help an analyst understand **why**, see **what is connected**, decide **what to do**, and feed that decision back into a controlled learning loop.

The project combines machine-learning risk scoring, point-in-time behavioral features, deterministic evidence, relationship analysis, human review, financial-risk context, and governed continual learning into one workflow.

It is a portfolio/research-grade system, not a production payment processor. The goal is to demonstrate the engineering thinking required to take an ML fraud model beyond a notebook and turn it into a **decision-support system**.

---

## Why I Built This

Fraud detection is often presented as a binary classification problem:

```text
Transaction → Model → Fraud / Not Fraud
```

That is useful for a benchmark. It is not enough for an analyst sitting in a risk queue.

In a real investigation, the next questions are usually:

- Why did the model flag this transaction?
- Is the customer's behavior unusual?
- Has this device, IP, or address appeared elsewhere?
- Is this an isolated event or part of a larger pattern?
- Should the merchant allow, review, or hold the transaction?
- If an analyst disagrees with the model, can that feedback improve the system safely?
- How do we prevent a newly trained model from silently becoming worse?

Merchant Risk Sentinel was designed around those questions.

The resulting flow is:

```text
                         ┌──────────────────────┐
                         │     Transaction      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Point-in-Time Feature Engine │
                    │  "What did we know then?"   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       ML Fraud Model         │
                    │    Fraud Probability P(f)    │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ Deterministic      │      │ Relationship       │
          │ Evidence Engine    │      │ Analysis           │
          └─────────┬──────────┘      └─────────┬──────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │ Risk Score + Explanation│
                     │ + Recommended Action    │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Human Analyst / HITL   │
                     │ ALLOW / REVIEW / HOLD  │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Confirmed Feedback     │
                     │ SQLite Feedback Store  │
                     └────────────┬───────────┘
                                  │
                           threshold reached
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Candidate Retraining   │
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │ Chronological Holdout  │
                     │ Candidate vs Active    │
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │ Promotion Gate         │
                     │ Promote / Reject       │
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │ Versioned Active Model │
                     └────────────────────────┘
```

---

## What Makes the Project Different

The interesting part of this project is not the choice of classifier.

It is the **system around the classifier**.

### 1. Point-in-time behavioral features

For fraud, leakage is especially dangerous.

A transaction should be scored using information that would actually have been available **before that transaction happened**.

The feature pipeline therefore derives historical signals such as:

- customer transaction count before the event
- customer historical average amount
- merchant historical average amount
- transaction velocity
- device/customer relationships
- IP/customer relationships
- address/customer relationships
- payment-method behavior
- refund and chargeback history
- account age
- location changes
- device changes
- IP changes
- time-of-day and day-of-week behavior
- amount anomalies
- behavioral risk signal count

The timestamp is used to establish temporal context, not as a shortcut for the model.

This matters because a random train/test split can make a fraud model look excellent while allowing future information to influence the past.

---

### 2. ML probability is not the final decision

The model produces a probability of fraud.

That probability is only one input into the operational risk decision.

The platform adds a deterministic evidence layer containing interpretable signals such as:

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

The evidence adjustment is deliberately **bounded**:

```text
ML score = fraud_probability × 100

Evidence adjustment = min(evidence_score × 0.15, 15)

Final score = min(100, ML score + evidence adjustment)
```

This keeps the ML model primary while allowing transparent evidence to influence the operational score.

That separation is intentional:

```text
Model:
"What is the estimated probability of fraud?"

Risk engine:
"What evidence supports or challenges that assessment?"

Decision layer:
"What should the merchant do?"
```

---

## Risk Decisioning

The API exposes model-level risk bands:

| Fraud Probability | Risk Level |
|---:|---|
| `< 0.10` | LOW |
| `0.10 – 0.39` | MEDIUM |
| `0.40 – 0.74` | HIGH |
| `>= 0.75` | CRITICAL |

The dashboard converts the fused risk score into an operational action:

| Final Score | Risk Level | Action |
|---:|---|---|
| `>= 80` | CRITICAL | HOLD |
| `>= 60` | HIGH | HOLD |
| `>= 40` | MEDIUM | REVIEW |
| `>= 20` | LOW | MONITOR |
| `< 20` | LOW | ALLOW |

The distinction between **prediction** and **decisioning** is important.

A model can be statistically strong while a business policy built around it is poorly calibrated. Keeping the two layers separate makes the system easier to reason about, test, and change.

---

## Relationship Analysis

Fraud is not always a property of one transaction.

Sometimes the useful signal is the relationship between entities.

Merchant Risk Sentinel looks for connections such as:

```text
Customer ───── Device
    │
    ├────────── IP
    │
    └────────── Address
```

This lets an analyst investigate questions such as:

- Has another customer used this device?
- Is the IP shared across multiple customers?
- Is the address associated with multiple accounts?
- Does the current transaction form part of a suspicious cluster?

The objective is to move from:

```text
"This transaction looks risky."
```

toward:

```text
"This transaction looks risky, and here is the surrounding evidence."
```

---

# Human-in-the-Loop

I intentionally did **not** make the AI the final authority.

An analyst can choose:

```text
ALLOW
REVIEW
HOLD
```

and classify the investigation as:

```text
CONFIRMED_FRAUD
CONFIRMED_LEGITIMATE
INCONCLUSIVE
```

The system records the decision alongside the original AI recommendation.

That makes it possible to measure:

```text
AI recommendation
        ↓
Human decision
        ↓
Agreement / Override
        ↓
Confirmed outcome
```

This is more useful than simply storing a `0/1` label because it preserves the context of the human-machine interaction.

---

# Continual Learning — With a Promotion Gate

A common mistake in "continual learning" demos is:

```text
New feedback → retrain → overwrite production model
```

Merchant Risk Sentinel deliberately does not do that.

The current workflow requires a minimum of **10 confirmed labels** before retraining.

```text
Analyst Feedback
       │
       ▼
SQLite Feedback Store
       │
       ├── insufficient labels → SKIP
       │
       ▼
Candidate Training
       │
       ▼
Chronological 80/20 Holdout
       │
       ├───────────────┐
       ▼               ▼
Candidate Model    Active Model
       │               │
       └───────┬───────┘
               ▼
        Same Holdout Set
               │
               ▼
        Promotion Gate
          │         │
       PASS        FAIL
          │         │
          ▼         ▼
      Promote      Reject
```

### Why the gate exists

A candidate can improve one metric and damage another.

For example, a model might achieve better recall by producing many more false positives.

The promotion logic therefore compares the candidate against the currently active model using:

- ROC-AUC
- PR-AUC
- F1

The configured promotion tolerance is `2%`.

The candidate is rejected when both PR-AUC and F1 materially degrade beyond that tolerance.

The active model is never replaced simply because retraining completed successfully.

---

# Temporal Validation

The continual-learning pipeline uses a chronological holdout rather than a random split.

Conceptually:

```text
Older transactions ────────────────► Newer transactions
|                                   |
|         Training data             | Holdout
|                                   |
└───────────────────────────────────┘
```

The timestamp is used for ordering only.

It is **not** passed to the model as one of the 43 model features.

This is a small implementation detail with a large consequence: it makes the evaluation closer to the real question the system eventually has to answer:

> "How well will this model perform on transactions it has not seen yet?"

---

# Model Versioning and Safe Promotion

Every promoted continual-learning model gets a unique version.

Example:

```text
v20260903164428
```

The system maintains:

```text
reports/
├── fraud_model.joblib
├── fraud_model_metadata.json
├── active_model.json
└── model_versions/
    └── v20260903164428/
        ├── fraud_model.joblib
        └── metadata.json
```

Promotion uses atomic file replacement so the active artifact is not left in a partially updated state.

The validated active artifact and its versioned registry copy were also checked using SHA-256 and produced the same hash:

```text
b60b67329a40e880f5f541ab962ee1e7b4ca81080496d720cc498252506f00ee
```

This gives the model lifecycle a basic chain of custody:

```text
Train
  ↓
Evaluate
  ↓
Version
  ↓
Promote atomically
  ↓
Record active version
  ↓
Reload live predictor
```

---

# Validation Results

The initial fraud model achieved:

```text
ROC-AUC       0.9716
PR-AUC        0.8081
Threshold     0.30
Features      43
```

The continual-learning pipeline was then exercised using controlled validation feedback.

### Candidate vs Active

| Metric | Active | Candidate |
|---|---:|---:|
| ROC-AUC | 0.9731 | 0.9628 |
| PR-AUC | 0.8197 | **0.8456** |
| F1 | 0.7014 | **0.7723** |

The candidate was promoted because the metrics relevant to the configured promotion gate improved rather than materially degrading.

The resulting active model version is:

```text
v20260903164428
```

### Important note about the validation data

The continual-learning test used controlled `CL_TEST_*` records to exercise the feedback → retraining → evaluation → promotion path.

Those synthetic validation records were subsequently removed from the feedback database.

The promoted model remains as the validated artifact.

They should therefore **not** be interpreted as production merchant feedback.

---

# End-to-End Live Test

The live API was tested with an existing customer while introducing a changed device.

The test kept the customer's historical context and changed only the relevant behavioral condition.

The API returned:

```text
HTTP 200
Model version:       v20260903164428
Feature count:       43
Risk level:          LOW
Recommendation:      MONITOR
Behavioral signal:   Device change
```

The response also contained explainable evidence:

```text
Device change
    ↓
HIGH severity evidence
```

This test is useful because it demonstrates that the deployed system can:

1. construct behavioral features,
2. load the active model,
3. score a live transaction,
4. generate deterministic evidence,
5. expose relationship context,
6. return an operational recommendation,
7. preserve the model version used for the decision.

---

# Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │   Analyst Dashboard  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │  Prediction / HITL   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             ┌────────────┐ ┌──────────────┐ ┌──────────────┐
             │ ML Model   │ │ Evidence     │ │ Relationship │
             │ Predictor  │ │ Risk Engine  │ │ Analysis     │
             └─────┬──────┘ └──────┬───────┘ └──────┬───────┘
                   │               │                │
                   └───────────────┼────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │  Risk Decision +     │
                         │  Explanation         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Analyst Feedback     │
                         │ SQLite               │
                         └──────────┬───────────┘
                                    │
                          every 6h maintenance
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Continual Learner    │
                         │ Candidate Training   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Promotion Gate       │
                         │ + Model Registry     │
                         └──────────────────────┘
```

---

# Project Structure

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
│   └── dashboard.py
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
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Service/model health |
| `GET` | `/model-info` | Active model metadata |
| `POST` | `/predict` | Score a transaction |
| `POST` | `/feedback` | Store analyst feedback |
| `POST` | `/learning/retrain` | Run governed retraining |
| `GET` | `/learning/status` | Inspect learning state |

A prediction can include:

- fraud probability
- risk score
- risk level
- recommendation
- behavioral features
- evidence
- relationship evidence
- model version
- human-in-the-loop options

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML | scikit-learn |
| Model | HistGradientBoostingClassifier |
| Data | Pandas / NumPy |
| API | FastAPI |
| Dashboard | Streamlit |
| Feedback Store | SQLite |
| Serialization | Joblib |
| Testing | Pytest |
| Deployment | Docker / Docker Compose |

---

# Running the Project

## Local

### 1. Clone

```bash
git clone <your-repository-url>
cd merchant-risk-sentinel
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 5. Start the dashboard

```bash
streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
API:       http://localhost:8000
Dashboard: http://localhost:8501
```

---

# Docker Compose

Run the full stack:

```bash
docker compose up --build -d
```

Check the services:

```bash
docker compose ps
```

The deployment contains:

```text
merchant-risk-api
merchant-risk-dashboard
merchant-risk-maintenance
```

The maintenance worker runs the continual-learning process every six hours.

Stop everything:

```bash
docker compose down
```

---

# Testing

Run the full test suite:

```bash
pytest -q
```

Current validated result:

```text
44 passed
```

The current suite also produces NumPy/joblib deprecation warnings in some live-predictor tests. They do not cause test failures.

---

# Operational Safety and Governance

The project intentionally includes controls that are easy to miss in a basic ML demo:

- **Point-in-time feature construction** to reduce temporal leakage
- **Human-confirmed labels** rather than blindly learning from model predictions
- **Minimum feedback threshold** before retraining
- **Chronological holdout evaluation**
- **Candidate vs active comparison**
- **Promotion tolerance**
- **Versioned model artifacts**
- **Atomic model promotion**
- **Retraining lock**
- **Model registry retention**
- **Active model metadata**
- **Maintenance status reporting**
- **Human override of AI recommendations**

The principle is simple:

> **Training a new model is not the same as earning the right to deploy it.**

---

# What I Would Do Next for Production

This project deliberately stops short of pretending to be a production payment-risk engine.

If I were taking it further, the next engineering priorities would be:

### Risk / ML

- probability calibration
- cost-sensitive threshold optimization
- merchant-specific thresholds
- precision/recall by fraud type
- fraud-loss-weighted evaluation
- temporal backtesting across multiple windows
- model drift and feature drift detection
- shadow evaluation before promotion
- automated rollback

### Data

- larger and more representative fraud labels
- label-delay handling
- stronger feature-store semantics
- entity-level graph features
- robust handling of missing and delayed signals

### Platform

- distributed feedback storage
- authenticated APIs
- role-based analyst access
- immutable audit logs
- production job orchestration
- monitoring and alerting
- model registry integration
- observability for latency, errors, drift, and decision outcomes

### Decisioning

- configurable merchant policies
- reason codes with stable taxonomy
- review queues
- case management
- SLA-aware prioritization
- expected-loss based decisioning

That is the direction I would take the prototype if the objective were to move from **“fraud model”** to a real **risk decisioning platform**.

---

# Key Engineering Takeaways

### Fraud detection
A supervised ML model can estimate fraud probability from transaction and behavioral data.

### Risk engineering
The probability becomes more useful when paired with deterministic, inspectable evidence.

### Explainability
An analyst should be able to understand the signals behind a decision instead of receiving a black-box score alone.

### Human-in-the-loop
Human overrides are not treated as noise; they are structured feedback.

### Continual learning
Feedback can improve the model, but retraining and deployment are separate controlled steps.

### MLOps
Model versioning, evaluation, promotion, locking, metadata, and atomic replacement make model lifecycle management explicit.

### Temporal correctness
Fraud evaluation must respect time. Preventing leakage is more important than producing an impressive random-split metric.

---

# Validation Snapshot

```text
┌─────────────────────────────────────────────────────┐
│              MERCHANT RISK SENTINEL                 │
├─────────────────────────────────────────────────────┤
│ Automated tests              44 passed              │
│ Model                         HistGradientBoosting   │
│ Model features               43                     │
│ Operating threshold           0.30                  │
│                                                     │
│ Initial ROC-AUC               0.9716                │
│ Initial PR-AUC                0.8081                │
│                                                     │
│ Candidate ROC-AUC             0.9628                │
│ Candidate PR-AUC              0.8456                │
│ Candidate F1                  0.7723                │
│                                                     │
│ Promotion result              PROMOTED              │
│ Active version                v20260903164428       │
│                                                     │
│ API health                    HEALTHY               │
│ Model loaded                  YES                   │
│ Human-in-the-loop             YES                   │
│ Continual learning            ENABLED               │
│ Docker services               3 / 3 UP              │
└─────────────────────────────────────────────────────┘
```

---

# A Note on Scope

Merchant Risk Sentinel uses a controlled fraud dataset and validation scenarios. It is **not** claiming production-level fraud performance, real merchant loss reduction, or deployment at payment-network scale.

The purpose of the project is to demonstrate the engineering path from:

```text
ML model
   ↓
behavioral intelligence
   ↓
risk evidence
   ↓
human decisioning
   ↓
feedback
   ↓
controlled model improvement
   ↓
governed deployment
```

That distinction matters.

A strong fraud system is not just a model with a high ROC-AUC. It is a system that can make decisions, explain them, learn from the right feedback, and fail safely when the model changes.

---

# Author

**Akshat Sinha**  
B.Tech — Computer Science & Engineering

Built as an AI risk engineering project focused on fraud detection, decisioning, explainability, human-in-the-loop systems, continual learning, and model governance.

---

## License

Add the project's chosen license before publishing the repository.
