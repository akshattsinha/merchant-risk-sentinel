from contextlib import asynccontextmanager
from pathlib import Path
import json
import sys
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat


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
# CONTINUAL LEARNING IMPORTS
# ============================================================

from src.learning.continual_learner import (
    record_feedback,
    learning_status,
    retrain_if_needed,
)


# ============================================================
# GLOBAL MODEL
# ============================================================

predictor = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor

    try:
        predictor = LiveRiskPredictor()

    except Exception:
        predictor = None
        raise

    yield

    predictor = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Merchant Risk Sentinel API",
    description=(
        "Real-time merchant fraud risk prediction, "
        "human-in-the-loop investigation, and "
        "feedback-driven continual learning API"
    ),
    version="1.2.0",
    lifespan=lifespan,
)


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

    amount: FiniteFloat = Field(
        gt=0
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
# CONTINUAL LEARNING FEEDBACK REQUEST
# ============================================================

class FeedbackRequest(BaseModel):

    # --------------------------------------------------------
    # Transaction ID
    # --------------------------------------------------------

    transaction_id: str = Field(
        min_length=1
    )

    # --------------------------------------------------------
    # Machine-learning label
    #
    # 1 = confirmed fraud
    # 0 = confirmed legitimate
    # --------------------------------------------------------

    label: int = Field(
        ge=0,
        le=1
    )

    # --------------------------------------------------------
    # Ground-truth investigation outcome
    # --------------------------------------------------------

    ground_truth: str = Field(
        min_length=1
    )

    # --------------------------------------------------------
    # What the AI recommended
    # --------------------------------------------------------

    ai_recommendation: str = Field(
        min_length=1
    )

    # --------------------------------------------------------
    # What the analyst actually decided
    # --------------------------------------------------------

    human_decision: str = Field(
        min_length=1
    )

    # --------------------------------------------------------
    # Existing final decision field
    #
    # Kept for compatibility.
    # --------------------------------------------------------

    final_decision: str = Field(
        min_length=1
    )

    # --------------------------------------------------------
    # Reason for decision
    # --------------------------------------------------------

    reason: str = Field(
        min_length=3
    )

    # --------------------------------------------------------
    # Additional investigation notes
    # --------------------------------------------------------

    investigation_notes: str = ""

    # --------------------------------------------------------
    # Original transaction
    # --------------------------------------------------------

    transaction: dict = Field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Point-in-time behavioral features
    # --------------------------------------------------------

    features: dict = Field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    model_version: str | None = None

    fraud_probability: FiniteFloat | None = None

    risk_score: FiniteFloat | None = None

    risk_level: str | None = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Merchant Risk Sentinel",
        "status": "online",
        "version": "1.2.0",
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

    metadata = {}

    metadata_file = (
        PROJECT_ROOT
        / "reports"
        / "fraud_model_metadata.json"
    )

    try:

        if metadata_file.exists():

            metadata = json.loads(
                metadata_file.read_text(
                    encoding="utf-8"
                )
            )

    except Exception:

        metadata = {}

    model_version = (
        metadata.get("model_version")
        or metadata.get("version")
        or "initial-model"
    )

    return {

        "model": metadata.get(
            "model",
            "HistGradientBoosting",
        ),

        "artifact": (
            "reports/fraud_model.joblib"
        ),

        "model_version": model_version,

        "feature_count": metadata.get(
            "feature_count",
            43,
        ),

        "operating_threshold": metadata.get(
            "threshold",
            0.30,
        ),

        "learning_mode": (
            "feedback_driven_continual_learning"
        ),

        "human_in_the_loop": True,

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
        # CHECK FOR NEWLY PROMOTED MODEL
        # ----------------------------------------------------

        if hasattr(
            predictor,
            "reload_if_changed",
        ):

            predictor.reload_if_changed()


        # ----------------------------------------------------
        # CONVERT REQUEST TO DICTIONARY
        # ----------------------------------------------------

        transaction_data = (
            transaction.model_dump()
        )


        # ----------------------------------------------------
        # GENERATE LIVE TRANSACTION ID
        # ----------------------------------------------------

        transaction_data[
            "transaction_id"
        ] = (
            "TXN_LIVE_"
            + uuid.uuid4().hex[:12].upper()
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
            result["features"]
        )


        # ----------------------------------------------------
        # RISK EVIDENCE
        # ----------------------------------------------------

        evidence = build_risk_evidence(
            risk=risk,
            behavioral_features=(
                behavioral_features
            ),
        )


        # ----------------------------------------------------
        # GET MODEL VERSION
        # ----------------------------------------------------

        model_version = "initial-model"

        try:

            metadata_file = (
                PROJECT_ROOT
                / "reports"
                / "fraud_model_metadata.json"
            )

            if metadata_file.exists():

                metadata = json.loads(
                    metadata_file.read_text(
                        encoding="utf-8"
                    )
                )

                model_version = (
                    metadata.get(
                        "model_version"
                    )
                    or metadata.get(
                        "version"
                    )
                    or model_version
                )

        except Exception:

            # Metadata problems must not
            # break prediction.

            pass


        # ----------------------------------------------------
        # AUDIT TRAIL
        # ----------------------------------------------------

        write_prediction_audit(
            transaction=transaction_data,
            risk=risk,
            evidence=evidence,
            model_version=model_version,
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

            "model": {

                "version":
                    model_version,

                "learning_mode": (
                    "feedback_driven_"
                    "continual_learning"
                ),

            },

            "human_feedback": {

                "required": True,

                "ai_recommendation":
                    risk[
                        "recommended_action"
                    ],

                "available_decisions": [

                    "ALLOW",
                    "REVIEW",
                    "HOLD",

                ],

                "investigation_outcomes": [

                    "CONFIRMED_FRAUD",
                    "CONFIRMED_LEGITIMATE",
                    "INCONCLUSIVE",

                ],

            },

        }


    except HTTPException:

        raise


    except Exception:

        # Do not expose internal exceptions.

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )


# ============================================================
# CONTINUAL LEARNING — SUBMIT FEEDBACK
# ============================================================

@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
):

    try:

        # ----------------------------------------------------
        # NORMALIZE VALUES
        # ----------------------------------------------------

        ground_truth = (
            request.ground_truth
            .strip()
            .upper()
        )

        ai_recommendation = (
            request.ai_recommendation
            .strip()
            .upper()
        )

        human_decision = (
            request.human_decision
            .strip()
            .upper()
        )

        final_decision = (
            request.final_decision
            .strip()
            .upper()
        )


        # ----------------------------------------------------
        # VALID GROUND TRUTH VALUES
        # ----------------------------------------------------

        allowed_ground_truth = {

            "CONFIRMED_FRAUD",

            "CONFIRMED_LEGITIMATE",

            "INCONCLUSIVE",

        }

        if ground_truth not in (
            allowed_ground_truth
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "ground_truth must be "
                    "CONFIRMED_FRAUD, "
                    "CONFIRMED_LEGITIMATE, "
                    "or INCONCLUSIVE."
                ),
            )


        # ----------------------------------------------------
        # VALID OPERATIONAL DECISIONS
        # ----------------------------------------------------

        allowed_decisions = {

            "ALLOW",
            "REVIEW",
            "HOLD",

        }


        if human_decision not in (
            allowed_decisions
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "human_decision must be "
                    "ALLOW, REVIEW, or HOLD."
                ),
            )


        if ai_recommendation not in (
            allowed_decisions
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "ai_recommendation must be "
                    "ALLOW, REVIEW, or HOLD."
                ),
            )


        if final_decision not in (
            allowed_decisions
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "final_decision must be "
                    "ALLOW, REVIEW, or HOLD."
                ),
            )


        # ----------------------------------------------------
        # INCONCLUSIVE CASE
        #
        # IMPORTANT:
        #
        # Do NOT train the model on uncertain feedback.
        # ----------------------------------------------------

        if ground_truth == "INCONCLUSIVE":

            return {

                "status": "success",

                "feedback": {

                    "status":
                        "not_used_for_training",

                    "transaction_id":
                        request.transaction_id,

                    "ground_truth":
                        ground_truth,

                    "ai_recommendation":
                        ai_recommendation,

                    "human_decision":
                        human_decision,

                    "final_decision":
                        final_decision,

                    "reason":
                        request.reason,

                    "investigation_notes":
                        request.investigation_notes,

                },

                "learning":
                    learning_status(),

            }


        # ----------------------------------------------------
        # DERIVE LABEL FROM CONFIRMED GROUND TRUTH
        # ----------------------------------------------------

        if (
            ground_truth
            == "CONFIRMED_FRAUD"
        ):

            expected_label = 1

        else:

            expected_label = 0


        # ----------------------------------------------------
        # VERIFY CLIENT LABEL
        # ----------------------------------------------------

        if request.label != expected_label:

            raise HTTPException(
                status_code=400,
                detail=(
                    "label does not match "
                    "ground_truth."
                ),
            )


        # ----------------------------------------------------
        # STORE CONFIRMED FEEDBACK
        # ----------------------------------------------------

        result = record_feedback(

            transaction_id=(
                request.transaction_id
            ),

            label=expected_label,

            ground_truth=(
                ground_truth
            ),

            ai_recommendation=(
                ai_recommendation
            ),

            human_decision=(
                human_decision
            ),

            final_decision=(
                final_decision
            ),

            reason=(
                request.reason
            ),

            investigation_notes=(
                request.investigation_notes
            ),

            transaction=(
                request.transaction
            ),

            features=(
                request.features
            ),

            model_version=(
                request.model_version
            ),

            fraud_probability=(
                request.fraud_probability
            ),

            risk_score=(
                request.risk_score
            ),

            risk_level=(
                request.risk_level
            ),

        )


        # ----------------------------------------------------
        # CURRENT LEARNING STATUS
        # ----------------------------------------------------

        current_learning_status = (
            learning_status()
        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "feedback":
                result,

            "learning":
                current_learning_status,

        }


    except HTTPException:

        raise


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Unable to store feedback.",
        )


# ============================================================
# CONTINUAL LEARNING — STATUS
# ============================================================

@app.get("/learning/status")
def get_learning_status():

    try:

        return {

            "status":
                "success",

            "learning":
                learning_status(),

        }

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "learning status."
            ),
        )


# ============================================================
# CONTINUAL LEARNING — RETRAIN
# ============================================================

@app.post("/learning/retrain")
def retrain_model():

    global predictor

    try:

        result = retrain_if_needed(
            min_feedback=10,
            force=False,
        )


        # ----------------------------------------------------
        # RELOAD MODEL AFTER PROMOTION
        # ----------------------------------------------------

        if (
            result.get("status")
            == "promoted"
        ):

            if predictor is not None:

                if hasattr(
                    predictor,
                    "reload_if_changed",
                ):

                    predictor.reload_if_changed(
                        force=True
                    )

                else:

                    predictor = (
                        LiveRiskPredictor()
                    )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "result":
                result,

            "learning":
                learning_status(),

        }


    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Model retraining failed.",
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

        result = (
            build_relationship_summary(
                incident_id
            )
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

        result = (
            build_transaction_relationships(
                transaction_id
            )
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