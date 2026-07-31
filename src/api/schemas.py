from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=20,
        description="Complete job description"
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(description="Predicted class: Real or Fake")
    fraud_probability: float = Field(description="Probability that the job is fraudulent")
    fraud_risk_score: int = Field(description="Fraud score from 0 to 100")
    risk_level: str = Field(description="Low, Medium, High or Critical")
    model_version: str = Field(description="Current deployed model version")