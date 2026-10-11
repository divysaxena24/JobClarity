"""
Prediction utilities for JobClarity.
"""
from src.explainability.shap_explainer import explain_prediction
from src.models.model_loader import load_model, load_vectorizer
from src.preprocessing.text_cleaner import clean_text

# Load once when API starts
try:
    model = load_model()
    vectorizer = load_vectorizer()
except Exception:
    model = None
    vectorizer = None


def predict_job(job_description: str):
    """
    Predict whether a job posting is Real or Fake.
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model or vectorizer artifact is not loaded.")

    # Clean text
    cleaned_text = clean_text(job_description)

    # TF-IDF transformation
    X = vectorizer.transform([cleaned_text])

    # Prediction
    prediction = int(model.predict(X)[0])

    # Probability of Fake class
    probability = float(model.predict_proba(X)[0][1])

    # Fraud Risk Score
    fraud_risk_score = int(round(probability * 100))
    top_reasons = explain_prediction(job_description)

    # Risk Level
    if fraud_risk_score <= 25:
        risk = "Low"
    elif fraud_risk_score <= 50:
        risk = "Medium"
    elif fraud_risk_score <= 75:
        risk = "High"
    else:
        risk = "Critical"

    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "fraud_probability": round(probability, 4),
        "fraud_risk_score": fraud_risk_score,
        "risk_level": risk,
        "top_reasons": top_reasons,
        "model_version": "1.0.0",
    }