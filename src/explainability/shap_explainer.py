"""
SHAP Explainability utilities for JobClarity.
"""

import numpy as np
import shap

from src.models.model_loader import (
    load_model,
    load_vectorizer
)

from src.preprocessing.text_cleaner import clean_text


# Load model and vectorizer once
model = load_model()
vectorizer = load_vectorizer()

# Create SHAP explainer once
explainer = shap.TreeExplainer(model)


def explain_prediction(job_description: str, top_k: int = 10):
    """
    Return the top SHAP features for a prediction.
    """

    cleaned_text = clean_text(job_description)

    X = vectorizer.transform([cleaned_text])

    shap_values = explainer.shap_values(X)

    feature_names = vectorizer.get_feature_names_out()

    values = shap_values[0]

    top_idx = np.argsort(np.abs(values))[::-1][:top_k]

    explanations = []

    for idx in top_idx:
        explanations.append({
            "feature": feature_names[idx],
            "impact": round(float(values[idx]), 4),
            "effect": (
                "Increases fraud risk"
                if values[idx] > 0
                else "Decreases fraud risk"
            )
        })

    return explanations