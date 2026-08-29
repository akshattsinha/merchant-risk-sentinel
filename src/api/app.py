from pathlib import Path
import sys
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXISTING PROJECT IMPORTS
# ============================================================

from src.models.live_predictor import LiveRiskPredictor

from src.analysis.fraud_graph import (
    build_relationship_summary,
    build_transaction_relationships,
)

from src.analysis.risk_evidence import (
    build_risk_evidence,
)

from src.audit.audit_logger import (
    write_prediction_audit,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Merchant Risk Sentinel API",
    description="Real-time merchant fraud risk prediction API",
    version="1.0.0",
)


# ============================================================
# GLOBAL MODEL
# ============================================================

predictor = None


# ============================================================
# TRANSACTION REQUEST
# ============================================================

class TransactionRequest(BaseModel):

    customer_id: str = Field(
        min_length=1
    )

    merchant_id: str = Field(
        min_length=1
    )

    amount: float = Field(
        gt=0,
        finite=True
    )

    timestamp: str = Field(
        min_length=1
    )

    payment_method: str = Field(
        min_length=1
    )

    device_id: str = Field(
        min_length=1
    )

    ip_id: str = Field(
        min_length=1
    )

    address_id: str = Field(
        min_length=1
    )

    account_age_days: int = Field(
        ge=1
    )

    location: str = Field(
        min_length=1
    )


# ============================================================
# MODEL STARTUP
# ============================================================

@app.on_event("startup")
def load_model():

    global predictor

    predictor = LiveRiskPredictor()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Merchant Risk Sentinel",
        "status": "online",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    return {
        "model": "HistGradientBoosting",

        "artifact": (
            "reports/fraud_model.joblib"
        ),

        "operating_threshold": 0.30,

        "risk_levels": {

            "LOW": "< 0.10",

            "MEDIUM": "0.10 - 0.39",

            "HIGH": "0.40 - 0.74",

            "CRITICAL": ">= 0.75",
        },
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    transaction: TransactionRequest,
):

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    try:

        # ----------------------------------------------------
        # CONVERT REQUEST TO DICTIONARY
        # ----------------------------------------------------

        transaction_data = (
            transaction.model_dump()
        )


        # ----------------------------------------------------
        # GENERATE ID FOR LIVE TRANSACTION
        # ----------------------------------------------------
        #
        # The incoming API request does not contain a
        # transaction_id, so we generate one for the
        # audit trail.
        #
        # Existing transaction fields are not modified.
        # ----------------------------------------------------

        transaction_data[
            "transaction_id"
        ] = (
            "TXN_LIVE_"
            + uuid.uuid4()
            .hex[:12]
            .upper()
        )


        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        result = predictor.predict(
            transaction_data
        )


        # ----------------------------------------------------
        # RISK INFORMATION
        # ----------------------------------------------------

        risk = {

            "fraud_probability":
                result[
                    "fraud_probability"
                ],

            "fraud_probability_percent":
                round(
                    result[
                        "fraud_probability"
                    ] * 100,
                    2,
                ),

            "risk_score":
                result[
                    "risk_score"
                ],

            "risk_level":
                result[
                    "risk_level"
                ],

            "recommended_action":
                result[
                    "recommended_action"
                ],
        }


        # ----------------------------------------------------
        # BEHAVIORAL FEATURES
        # ----------------------------------------------------

        behavioral_features = (
            result[
                "features"
            ]
        )


        # ----------------------------------------------------
        # RISK EVIDENCE
        # ----------------------------------------------------
        #
        # This layer converts the model and behavioral
        # signals into structured evidence.
        #
        # It does not replace the ML prediction.
        # ----------------------------------------------------

        evidence = build_risk_evidence(
            risk=risk,
            behavioral_features=(
                behavioral_features
            ),
        )


        # ----------------------------------------------------
        # AUDIT TRAIL
        # ----------------------------------------------------

        write_prediction_audit(
            transaction=transaction_data,
            risk=risk,
            evidence=evidence,
            model_version="1.0.0",
            threshold=0.30,
            incident_id=None,
        )


        # ----------------------------------------------------
        # API RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "transaction":
                transaction_data,

            "risk":
                risk,

            "behavioral_features":
                behavioral_features,

            "evidence":
                evidence,
        }


    except HTTPException:

        raise


    except Exception:

        # Do not expose internal Python exceptions,
        # model paths, stack traces, or implementation
        # details to API clients.

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )


# ============================================================
# FRAUD GRAPH / INCIDENT RELATIONSHIP ANALYSIS
# ============================================================

@app.get(
    "/incidents/{incident_id}/relationships"
)
def get_incident_relationships(
    incident_id: str,
):

    """
    Return relationship evidence for an incident.

    This endpoint is read-only.

    It does not modify:

    - ML predictions
    - risk scores
    - incident generation
    - transaction data
    """

    try:

        result = build_relationship_summary(
            incident_id
        )


        if not result.get(
            "found",
            False,
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relationship data found "
                    f"for incident {incident_id}"
                ),
            )


        return result


    except HTTPException:

        raise


    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "Relationship analysis failed."
            ),
        )


# ============================================================
# TRANSACTION RELATIONSHIP ANALYSIS
# ============================================================

@app.get(
    "/transactions/{transaction_id}/relationships"
)
def get_transaction_relationships(
    transaction_id: str,
):

    """
    Return relationship evidence around a transaction.

    A missing transaction is treated as a normal
    not-found condition and returns HTTP 404.
    """

    try:

        result = build_transaction_relationships(
            transaction_id
        )


    except HTTPException:

        raise


    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "Transaction relationship "
                "analysis failed."
            ),
        )


    # --------------------------------------------------------
    # TRANSACTION NOT FOUND
    # --------------------------------------------------------

    if not result.get(
        "found",
        False,
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "No relationship data found "
                f"for transaction {transaction_id}"
            ),
        )


    return result