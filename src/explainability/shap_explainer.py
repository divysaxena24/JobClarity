"""
SHAP Explainability utilities for JobClarity.

Uses native XGBoost TreeSHAP (booster.predict(dmatrix, pred_contribs=True))
to compute exact feature attributions efficiently and reliably.
"""

import logging
from typing import Any
import numpy as np
import xgboost as xgb

from src.models.model_loader import (
    load_model,
    load_vectorizer,
)
from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)

# Module-level singletons for model, vectorizer, and booster
model: Any = None
vectorizer: Any = None
booster: Any = None
explainer: Any = None


def init_explainer() -> None:
    """Initialize or reload model, vectorizer, and XGBoost booster."""
    global model, vectorizer, booster, explainer
    try:
        model = load_model()
        vectorizer = load_vectorizer()
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        else:
            booster = model
        # Keep explainer pointing to booster for backward compatibility
        explainer = booster
        logger.info("Successfully initialized XGBoost native TreeSHAP explainer.")
    except Exception as exc:
        logger.warning(
            "Failed to initialize explainability model or vectorizer: %s",
            str(exc),
        )
        model = None
        vectorizer = None
        booster = None
        explainer = None


# Load once when module is imported
init_explainer()


def explain_prediction(job_description: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Return the top SHAP features for a prediction using native XGBoost TreeSHAP.

    Args:
        job_description: Raw job description text.
        top_k: Maximum number of top features to return (default: 10).

    Returns:
        List of dicts with keys:
          - 'feature': Feature name string from vectorizer vocabulary
          - 'impact': Float rounded to 4 decimals (positive = increases fraud risk)
          - 'effect': Human-readable direction ("Increases fraud risk" / "Decreases fraud risk")
        Returns an empty list gracefully on invalid input or failure.
    """
    if vectorizer is None:
        logger.debug("Explainability skipped: vectorizer artifact is not loaded.")
        return []

    active_model = explainer if explainer is not None else (booster if booster is not None else model)
    if active_model is None:
        logger.debug("Explainability skipped: model or booster artifact is not loaded.")
        return []

    if not isinstance(job_description, str) or not job_description.strip():
        return []

    try:
        cleaned_text = clean_text(job_description)
        X = vectorizer.transform([cleaned_text])

        # Support mock explainer with shap_values() in legacy unit tests
        if hasattr(active_model, "shap_values"):
            raw_shap = active_model.shap_values(X)
            values = np.asarray(raw_shap[0] if isinstance(raw_shap, list) else raw_shap).ravel()
        else:
            # Native XGBoost C++ TreeSHAP calculation
            raw_booster = getattr(active_model, "get_booster", lambda: active_model)()
            dmat = xgb.DMatrix(X)
            contribs = raw_booster.predict(dmat, pred_contribs=True)
            # Row 0: shape (num_features + 1,) where the last element is the bias term
            row_contribs = np.asarray(contribs[0]).ravel()
            values = row_contribs[:-1]

        feature_names = vectorizer.get_feature_names_out()
        if len(values) != len(feature_names):
            logger.warning(
                "SHAP values dimension (%d) does not match vectorizer vocabulary length (%d).",
                len(values),
                len(feature_names),
            )
            return []

        # Filter strictly non-zero features (do not invent explanations for zero impact)
        non_zero_indices = np.flatnonzero(np.abs(values) > 1e-6)
        if len(non_zero_indices) == 0:
            return []

        # Sort non-zero features by absolute impact descending
        sorted_order = np.argsort(np.abs(values[non_zero_indices]))[::-1]
        top_indices = non_zero_indices[sorted_order][:top_k]

        explanations: list[dict[str, Any]] = []
        for idx in top_indices:
            val = float(values[idx])
            explanations.append({
                "feature": str(feature_names[idx]),
                "impact": round(val, 4),
                "effect": "Increases fraud risk" if val > 0 else "Decreases fraud risk",
            })

        return explanations
    except Exception as exc:
        logger.warning(
            "Explainability calculation failed safely for input length %d: %s",
            len(job_description) if isinstance(job_description, str) else 0,
            str(exc),
        )
        return []
