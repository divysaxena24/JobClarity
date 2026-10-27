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


class JDGenerationRequest(BaseModel):
    """Request schema for synthetic job description generation."""

    job_role: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Target job title or role (e.g. Senior Backend Engineer)",
    )
    skills: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Key skills and competencies (comma-separated or list)",
    )
    experience_level: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Experience level (e.g. Entry-level, Mid-level, Senior)",
    )
    location: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Job location or mode (e.g. Remote, San Francisco, CA)",
    )
    company_name: Optional[str] = Field(
        None,
        max_length=120,
        description="Hiring company name (optional)",
    )
    salary_range: Optional[str] = Field(
        None,
        max_length=100,
        description="Salary range or compensation details (optional)",
    )


class JDGenerationResponse(BaseModel):
    """Response schema for synthetic job description generation."""

    generated_job_description: str
    is_ai_generated: bool = True
    provider: str
    disclaimer: str
    input_parameters: Dict[str, Any]
