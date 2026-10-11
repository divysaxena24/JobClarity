from fastapi import APIRouter, HTTPException

from src.api import predictor
from src.api.predictor import predict_job
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

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
        "health": "/health"
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
    try:
        return predict_job(request.job_description)
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