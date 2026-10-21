from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PredictionRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=20,
        description="Complete job description"
    )


class PredictionResponse(BaseModel):

    prediction: str
    fraud_probability: float
    fraud_risk_score: int
    risk_level: str

    top_reasons: List[Dict]

    model_version: str


class StagingPredictionResponse(BaseModel):
    """Non-breaking staging response schema for multi-tier decision classification."""

    job_description_snippet: str
    fraud_probability: float
    risk_tier: str
    action: str
    calibrated_thresholds: Dict[str, float]
    explanation_summary: Optional[Dict[str, Any]] = None
    staging_version: str