from pydantic import BaseModel, Field
from typing import List, Dict


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