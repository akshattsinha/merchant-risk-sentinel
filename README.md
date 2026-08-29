# 🛡️ Merchant Risk Sentinel

## Explainable AI Fraud-Risk Detection & Incident Response Platform

Merchant Risk Sentinel is a defensive AI-powered fraud-risk management platform designed to help merchants detect suspicious transactions, investigate connected abuse patterns, quantify financial risk, and make informed human-in-the-loop decisions.

The system combines machine-learning fraud prediction with deterministic evidence analysis, relationship-based abuse detection, risk scoring, incident escalation, model evaluation, and model monitoring.

---

## 🎯 AI Risk Manager Track

This project focuses on **payment fraud** as the primary class of financial loss.

The system implements the complete risk-management workflow:

```text
Transaction
     ↓
Feature Engineering
     ↓
ML Fraud Prediction
     ↓
Evidence Analysis
     ↓
Abuse-Ring Analysis
     ↓
Risk Decision Engine
     ↓
Model Recommendation
     ↓
Merchant / Analyst Decision
     ↓
Incident Escalation
     ↓
Investigation & Audit
```

The model provides recommendations only. Final decisions remain with the merchant or risk analyst.

---

## 🚨 Problem

Fraud detection cannot be reduced to a simple:

> Fraud / Not Fraud

decision.

A useful fraud-risk system should answer:

- Is this transaction suspicious?
- How likely is it to be fraudulent?
- What evidence caused the risk to increase?
- Is this transaction connected to other suspicious transactions?
- What is the potential financial exposure?
- Should the transaction be allowed, reviewed, or escalated?
- What is the cost of false positives and false negatives?
- Is model performance changing over time?

Merchant Risk Sentinel is designed around these questions.

---

# 🤖 Key Features

## 1. ML Fraud Detection

Machine-learning based fraud prediction using engineered transaction features.

The system produces a fraud probability that is evaluated independently from the final decision-support risk score.

---

## 2. Held-Out Model Evaluation

The fraud model is evaluated using a **temporal holdout methodology**.

### Current Results

| Metric | Score |
|---|---:|
| Precision | **83.14%** |
| Recall | **69.20%** |
| F1 Score | **75.53%** |
| PR-AUC | **84.51%** |

The final metrics are calculated on an untouched held-out test set.

---

## 3. Threshold Optimization

The operating threshold is selected using validation data rather than tuning directly on the final test set.

The system evaluates different thresholds using:

- Precision
- Recall
- F1 Score
- False-positive cost
- False-negative cost
- Expected financial loss

This allows the operating point to be selected based on the merchant's risk objective rather than accuracy alone.

---

## 4. Explainable Risk Scoring

The platform separates different sources of risk:

```text
Fraud Probability
       +
ML Risk
       +
Behavioral Evidence
       +
Relationship / Abuse Risk
       ↓
Final Decision Risk
```

The final risk score is a **decision-support score**, not a direct probability of fraud.

Risk evidence can include:

- Amount anomalies
- Transaction velocity
- Device reuse
- IP reuse
- Address reuse
- Customer relationships
- Connected transactions

---

## 5. 🔗 Abuse Risk Sentinel

Relationship-based abuse analysis connects:

```text
Customers
   ↕
Devices
   ↕
IPs
   ↕
Addresses
   ↕
Merchants
   ↕
Transactions
```

The system identifies reused entities and connected transaction activity to detect potential coordinated abuse.

Analysts can inspect the relationship graph and related transactions behind the abuse-risk signal.

---

## 6. ⚡ Live Transaction Risk Assessment

New transactions can be submitted for risk assessment.

The system:

1. Validates the transaction
2. Generates required features
3. Runs the fraud model
4. Calculates deterministic evidence
5. Evaluates relationship signals
6. Calculates final risk
7. Generates a recommendation
8. Persists the transaction and risk information
9. Escalates significant cases into the Incident Centre

New transactions do not automatically retrain the model.

---

## 7. 👤 Human-in-the-Loop Decisioning

The model does not independently execute payment actions.

Instead:

```text
Model
  ↓
Recommendation
  ↓
Merchant / Analyst
  ↓
Final Decision
```

Possible defensive decisions include:

- Allow
- Review
- Hold
- Escalate

The analyst can provide a decision reason for auditability.

---

## 8. 🚨 Incident Centre

High-risk activity can be escalated into an incident workflow.

Incidents contain investigation context such as:

- Incident ID
- Severity
- Risk score
- Affected transactions
- Customers
- Potential exposure
- Fraud signals
- Root-cause indicators
- Recommended response
- Investigation information

This turns individual predictions into an operational fraud-investigation workflow.

---

## 9. 📊 Model Performance

The platform provides model evaluation including:

- Confusion Matrix
- Precision
- Recall
- F1 Score
- PR-AUC
- Threshold Analysis
- Financial Cost Analysis
- Held-Out Evaluation

The goal is to evaluate the model based on both statistical performance and business impact.

---

## 10. 📈 Model Monitoring

The platform provides production-style monitoring for risk behavior, including:

- Fraud Rate
- Fraud-rate changes
- Fraud spikes
- Risk Distribution
- Model Drift indicators

Monitoring is read-only and does not automatically retrain or replace the production model.

---

## 11. 🔍 Transaction Explorer

Transactions can be inspected individually and in relation to other activity.

The explorer can surface:

- Transaction attributes
- Risk information
- Related customers
- Shared devices
- Shared IPs
- Shared addresses
- Related transactions
- Fraud evidence

This allows an analyst to move from an alert to the underlying evidence.

---

## 12. 🤖 AI Fraud Investigation Assistant

A local AI assistant helps analysts investigate fraud-related questions using existing risk evidence and incident context.

The AI is used for:

- Investigation assistance
- Evidence explanation
- Incident summarization
- Analyst support

It does not independently execute payment actions.

---

## 13. 📝 Audit Trail

Risk assessments and decisions can be persisted for traceability.

The audit workflow is designed around:

```text
Transaction
    ↓
Prediction
    ↓
Risk Evidence
    ↓
Recommendation
    ↓
Analyst Decision
    ↓
Incident / Outcome
```

This provides an investigation history rather than treating predictions as isolated events.

---

# 🏗️ Architecture

```text
                         MERCHANT RISK SENTINEL
                                  │
                           Transaction API
                                  │
                                  ▼
                         Feature Engineering
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
             ML Model       Evidence Engine    Abuse Analysis
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
                           Risk Decision Engine
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               ML Risk       Evidence Risk   Abuse Risk
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                             Final Risk
                                  │
                                  ▼
                           Recommendation
                                  │
                                  ▼
                         Human Decision
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
                     Allow               Escalate
                                             │
                                             ▼
                                      Incident Centre
                                             │
                                             ▼
                                        Investigation
                                             │
                                             ▼
                                          Audit Trail
```

---

# 📁 Project Structure

```text
merchant-risk-sentinel/
│
├── src/
│   ├── analysis/
│   │   ├── fraud_graph.py
│   │   ├── incident_engine.py
│   │   ├── response_engine.py
│   │   ├── risk_evidence.py
│   │   └── root_cause_engine.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   └── __init__.py
│   │
│   ├── audit/
│   │   └── audit_logger.py
│   │
│   ├── chatbot/
│   │   └── fraud_assistant.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   ├── train_strong_model.py
│   │   ├── live_predictor.py
│   │   ├── optimize_threshold.py
│   │   ├── optimize_strong_threshold.py
│   │   └── build_model_artifact.py
│   │
│   ├── dashboard.py
│   ├── generate_data.py
│   └── inspect_data.py
│
├── scripts/
│   └── evaluate_model.py
│
├── reports/
│   ├── fraud_model_metadata.json
│   ├── strong_model_metrics.json
│   ├── strong_optimized_metrics.json
│   ├── strong_threshold_analysis.csv
│   └── threshold_analysis.csv
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧪 Model Evaluation Methodology

The project uses a temporal holdout evaluation strategy.

```text
Historical Data
      │
      ├── Training Set
      │
      ├── Validation Set
      │       ↓
      │   Threshold Selection
      │
      └── Held-Out Test Set
              ↓
        Final Evaluation
```

The test set remains untouched during threshold selection.

This helps prevent test-set leakage when reporting final model performance.

---

# 📈 Understanding the Metrics

### Precision

Measures how many transactions predicted as fraudulent were actually fraudulent.

### Recall

Measures how much of the actual fraud the model successfully detects.

### F1 Score

Provides a balance between precision and recall.

### PR-AUC

Measures the overall precision-recall trade-off across different classification thresholds and is particularly useful when fraud is relatively rare compared with legitimate transactions.

---

# 💰 Business Risk Perspective

Fraud detection is not simply an accuracy problem.

A false positive can result in:

- Lost legitimate revenue
- Customer friction
- Manual investigation cost

A false negative can result in:

- Fraud losses
- Merchant exposure
- Chargeback-related costs

Therefore, Merchant Risk Sentinel considers financial cost when evaluating operating thresholds rather than optimizing solely for statistical performance.

---

# 🔐 Defensive Design

Merchant Risk Sentinel is designed strictly for:

- Fraud prevention
- Fraud detection
- Risk analysis
- Fraud investigation
- Defensive incident response

The system does not provide offensive capabilities.

Model recommendations are advisory and human-controlled.

---

# 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- Docker
- REST APIs
- Local LLM / Ollama
- CSV-based persistence
- JSON evaluation artifacts

---

# 🚀 Running Locally

## 1. Clone the Repository

```bash
git clone https://github.com/akshattsinha/merchant-risk-sentinel.git
cd merchant-risk-sentinel
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the Dashboard

```bash
python -m streamlit run src/dashboard.py --server.port 8503
```

Open:

```text
http://localhost:8503
```

## 5. Start the Risk API

If the API is configured separately:

```bash
python src/api/app.py
```

---

# 🐳 Docker

The project includes:

```text
Dockerfile
docker-compose.yml
```

These provide containerized development and deployment configuration.

---

# ⚠️ Data & Privacy

The repository does not include private or production transaction data.

Development datasets containing transaction-level information should remain local and should not be committed to a public repository.

Synthetic data can be used for demonstrations and testing.

---

# 📌 Current Limitations

Merchant Risk Sentinel is a fraud-risk management prototype and is not presented as a production payment-processing system.

Current limitations include:

- Evaluation data is synthetic/development data.
- Production fraud distributions may differ.
- Model performance can change under distribution shift.
- Financial cost assumptions depend on merchant-specific economics.
- Analyst outcomes are required for true production feedback-loop evaluation.
- Model monitoring does not automatically retrain the model.

These limitations are intentionally documented rather than hidden.

---

# 🎯 Future Improvements

Planned improvements include:

- Probability calibration
- Stronger business-cost optimization
- Analyst outcome feedback loops
- Feature-level drift monitoring
- Prediction drift monitoring
- Automated model-quality alerts
- Model version registry
- PostgreSQL persistence
- Expanded automated testing
- CI/CD pipeline
- Production API deployment

---

# 🎓 AI Risk Manager Alignment

Merchant Risk Sentinel addresses the core requirements of the AI Risk Manager track:

| Track Requirement | Implementation |
|---|---|
| Class of loss | Payment fraud |
| Working detector | ML fraud-risk model |
| Held-out evaluation | Temporal holdout |
| Precision | 83.14% |
| Recall | 69.20% |
| F1 | 75.53% |
| PR-AUC | 84.51% |
| Business cost | Threshold / financial-cost analysis |
| Fraud investigation | Evidence + Transaction Explorer |
| Abuse detection | Relationship-based Abuse Risk Sentinel |
| Response | Defensive recommendations |
| Human control | Merchant / Analyst final decision |
| Escalation | Incident Centre |
| Monitoring | Fraud rate, risk distribution, drift indicators |

---

# 👨‍💻 Project

## Merchant Risk Sentinel

**Explainable AI Fraud-Risk Detection & Incident Response Platform**

Built with a focus on defensive fraud detection, explainability, abuse-network analysis, business-risk evaluation, and human-in-the-loop decisioning.
