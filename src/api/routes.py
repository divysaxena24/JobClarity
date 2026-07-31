from fastapi import APIRouter

from src.api.predictor import (
    predict_job,
    model,
    vectorizer
)

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "version": "1.0.0"
    }


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    return predict_job(
        request.job_description
    )