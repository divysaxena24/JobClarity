import logging
import time
from fastapi import APIRouter, HTTPException

from src.api import predictor
from src.api.predictor import predict_job
from src.api.schemas import (
    JDGenerationRequest,
    JDGenerationResponse,
    PredictionRequest,
    PredictionResponse,
    StagingPredictionResponse,
)
from src.models.staging_predictor import get_staging_predictor
from src.services.jd_generator import generate_job_description
from src.services.prediction_logger import log_prediction

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
model = predictor.model
vectorizer = predictor.vectorizer

router = APIRouter()


@router.get("/")
def home():
    return {
        "project": "JobClarity",
        "version": "1.0.0",
        "description": "AI-powered Fake Job Detection API",
        "docs": "/docs",
        "health": "/health",
        "generate_jd": "/generate-jd",
    }


@router.get("/health")
def health():
    is_model_loaded = predictor.model is not None
    is_vectorizer_loaded = predictor.vectorizer is not None
    is_healthy = is_model_loaded and is_vectorizer_loaded
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_model_loaded,
        "vectorizer_loaded": is_vectorizer_loaded,
        "version": "1.0.0"
    }


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    try:
        result = predict_job(request.job_description)
    except HTTPException:
        raise
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model service is currently unavailable."
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction request."
        )

    # Structured prediction logging: strictly decoupled from inference lifecycle
    try:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_prediction(
            endpoint="/predict",
            model_name="JobClarity-Baseline",
            model_version=str(result.get("model_version", "1.0.0")),
            probability=result.get("fraud_probability"),
            decision_threshold=0.50,
            risk_tier=result.get("risk_level"),
            latency_ms=latency_ms,
            job_description=request.job_description,
            status="success",
        )
    except Exception as log_exc:
        logger.warning("Prediction logging failed safely: %s", str(log_exc))

    return result


@router.post(
    "/predict/staging",
    response_model=StagingPredictionResponse
)
def predict_staging(request: PredictionRequest):
    """
    Non-breaking staging endpoint for multi-tier decision classification.
    Evaluates candidate model with calibrated thresholds:
      - LOW: p < 0.14 -> PASS
      - REVIEW: 0.14 <= p < 0.33 -> FLAG_FOR_REVIEW
      - HIGH_RISK: p >= 0.33 -> BLOCK
    """
    staging_pred = get_staging_predictor()
    if not staging_pred.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Staging model service is currently unavailable or uninitialized."
        )
    start_time = time.perf_counter()
    try:
        result = staging_pred.predict(request.job_description)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the staging prediction request."
        )

    # Structured prediction logging: strictly decoupled from inference lifecycle
    try:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_prediction(
            endpoint="/predict/staging",
            model_name="JobClarity-Candidate-D",
            model_version=str(result.get("staging_version", "staging-candidate-d-1.0.0")),
            probability=result.get("fraud_probability"),
            decision_threshold=result.get("calibrated_thresholds", {"review": 0.14, "high_risk": 0.33}),
            risk_tier=result.get("risk_tier"),
            action=result.get("action"),
            latency_ms=latency_ms,
            job_description=request.job_description,
            status="success",
        )
    except Exception as log_exc:
        logger.warning("Staging prediction logging failed safely: %s", str(log_exc))

    return result


@router.post(
    "/generate-jd",
    response_model=JDGenerationResponse
)
def generate_jd_endpoint(request: JDGenerationRequest):
    """
    Generate an editable synthetic job description from structured parameters.
    Clearly marks content as AI-generated and notes risk estimation disclaimers.
    """
    try:
        result = generate_job_description(
            job_role=request.job_role,
            skills=request.skills,
            experience_level=request.experience_level,
            location=request.location,
            company_name=request.company_name,
            salary_range=request.salary_range,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the job description: {str(exc)}",
        )
