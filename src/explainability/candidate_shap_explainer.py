"""
Candidate SHAP explainability and error analysis module for JobClarity (Day 5).

Provides:
  1. Exact TreeSHAP computation via XGBoost native pred_contribs algorithm (Lundberg et al., 2018).
  2. Feature name alignment combining word TF-IDF n-grams and 21 deterministic description indicators.
  3. Distinction between model contributions and causal explanations.
  4. Global feature importance calculation (mean absolute SHAP values).
  5. Individual instance attribution (top-k positive and negative risk contributors).
  6. Sliced error analysis across text length cohorts, missingness, and suspicious language markers.
  7. Representative cohort extraction for True Positives, False Positives, True Negatives, False Negatives.

Governance & Safety:
  - Production model artifacts in models/ are NEVER modified or overwritten.
  - Zero data leakage: fits preprocessors only on training data.
  - Causal disclaimers attached to all explanations to prevent false claims of fraud proof.
"""

from __future__ import annotations

import logging
from math import isinf, isnan
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
import xgboost as xgb

from src.features.feature_pipeline import CompositeFeaturePipeline
from src.features.suspicious_language import SuspiciousLanguageExtractor
from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)

CAUSAL_DISCLAIMER: str = (
    "Model attribution reflects statistical tree split contribution along XGBoost decision paths. "
    "Features indicate correlated patterns in training data and do not constitute causal proof of fraudulent intent."
)


def get_feature_type(feature_name: str) -> str:
    """
    Categorize a feature as a deterministic domain indicator or a TF-IDF lexical n-gram.
    """
    if feature_name in SuspiciousLanguageExtractor.FEATURE_NAMES:
        return "deterministic_indicator"
    return "tfidf_ngram"


def compute_tree_shap_contributions(
    model: Any,
    X: Any,
) -> tuple[np.ndarray, float]:
    """
    Compute exact TreeSHAP attribution values using XGBoost's native C++ pred_contribs algorithm.

    Args:
        model: Trained XGBClassifier or XGBoost Booster.
        X: Feature matrix (CSR sparse matrix or 2D numpy array) of shape (n_samples, n_features).

    Returns:
        shap_values: np.ndarray of shape (n_samples, n_features) with margin contributions.
        base_score: float bias term (expected log-odds margin).

    Raises:
        RuntimeError: If model is not an XGBoost model or does not expose a Booster.
    """
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise RuntimeError(
            f"Expected XGBClassifier or Booster with get_booster(), got {type(model).__name__}"
        )

    # Convert sparse matrix or array to DMatrix
    dmat = xgb.DMatrix(X)

    # Native TreeSHAP computation: returns shape (n_samples, n_features + 1)
    # The last column is the expected value (bias/margin offset).
    contribs = booster.predict(dmat, pred_contribs=True)

    if contribs.ndim == 1:
        contribs = contribs.reshape(1, -1)

    shap_values = contribs[:, :-1]
    base_score = float(contribs[0, -1])

    return shap_values, base_score


def explain_candidate_prediction(
    job_description: str,
    model: Any,
    pipeline: CompositeFeaturePipeline,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Generate local SHAP explanations for an individual job description using Candidate D.

    Args:
        job_description: Raw job description text.
        model: Trained XGBoost model.
        pipeline: Fitted CompositeFeaturePipeline.
        top_k: Number of highest-magnitude contributing features to return.

    Returns:
        Structured explanation dictionary with feature attributions, model margin,
        predicted probability, and causal disclaimer.
    """
    if not isinstance(job_description, str):
        raise TypeError(f"job_description must be a string, got {type(job_description).__name__}")

    raw_stripped = job_description.strip()
    if len(raw_stripped) < 20:
        return {
            "job_description_snippet": raw_stripped,
            "predicted_probability": 0.0,
            "margin_sum": 0.0,
            "base_margin": 0.0,
            "top_features": [],
            "status": "INPUT_TOO_SHORT",
            "message": "Input contains fewer than 20 characters; explanation skipped.",
            "causal_disclaimer": CAUSAL_DISCLAIMER,
        }

    # Transform input through candidate feature pipeline
    X = pipeline.transform([job_description])
    shap_matrix, base_score = compute_tree_shap_contributions(model, X)
    shap_vals = shap_matrix[0]

    feature_names = (
        pipeline.get_feature_names_out()
        if hasattr(pipeline, "get_feature_names_out")
        else np.array(
            list(pipeline.tfidf_model.get_feature_names_out())
            + list(pipeline.suspicious_extractor.FEATURE_NAMES),
            dtype=object,
        )
    )

    if len(feature_names) != len(shap_vals):
        raise ValueError(
            f"Feature count mismatch: {len(feature_names)} names vs {len(shap_vals)} SHAP values."
        )

    # Sort by absolute SHAP magnitude descending
    abs_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_vals)[::-1][:top_k]

    # Convert sparse X row to dense array for feature value inspection
    X_dense = X.toarray()[0] if issparse(X) else np.asarray(X)[0]

    explanations: list[dict[str, Any]] = []
    for idx in top_indices:
        val = float(shap_vals[idx])
        if abs(val) < 1e-6:
            continue
        feat_name = str(feature_names[idx])
        raw_val = float(X_dense[idx])
        explanations.append({
            "feature": feat_name,
            "feature_type": get_feature_type(feat_name),
            "feature_value": round(raw_val, 4),
            "impact": round(val, 4),
            "effect": "Increases fraud risk" if val > 0 else "Decreases fraud risk",
            "attribution_type": "TreeSHAP Margin Contribution",
        })

    margin_sum = float(shap_vals.sum() + base_score)
    prob_from_shap = float(1.0 / (1.0 + np.exp(-margin_sum)))

    # Direct model probability
    prob_direct = float(model.predict_proba(X)[0][1])

    snippet = (
        raw_stripped[:97] + "..."
        if len(raw_stripped) > 100
        else raw_stripped
    )

    return {
        "job_description_snippet": snippet,
        "predicted_probability": round(prob_direct, 4),
        "probability_from_shap": round(prob_from_shap, 4),
        "base_margin": round(base_score, 4),
        "margin_sum": round(margin_sum, 4),
        "top_features": explanations,
        "total_active_features": int((abs_vals > 1e-6).sum()),
        "total_evaluated_features": len(feature_names),
        "causal_disclaimer": CAUSAL_DISCLAIMER,
    }


def compute_global_feature_importance(
    model: Any,
    pipeline: CompositeFeaturePipeline,
    texts: list[str],
    sample_size: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute global feature importance by averaging absolute SHAP values across a sample of postings.

    Args:
        model: Trained XGBoost model.
        pipeline: Fitted CompositeFeaturePipeline.
        texts: List of raw job description texts.
        sample_size: Number of postings to sample for efficiency.
        random_state: Random seed for sampling.

    Returns:
        DataFrame sorted by mean absolute SHAP contribution descending.
    """
    rng = np.random.default_rng(random_state)
    n = len(texts)
    if n > sample_size:
        indices = rng.choice(n, size=sample_size, replace=False)
        sampled_texts = [texts[i] for i in indices]
    else:
        sampled_texts = texts

    logger.info("Computing global TreeSHAP across %d sampled postings...", len(sampled_texts))
    X_sample = pipeline.transform(sampled_texts)
    shap_matrix, _ = compute_tree_shap_contributions(model, X_sample)

    feature_names = (
        pipeline.get_feature_names_out()
        if hasattr(pipeline, "get_feature_names_out")
        else np.array(
            list(pipeline.tfidf_model.get_feature_names_out())
            + list(pipeline.suspicious_extractor.FEATURE_NAMES),
            dtype=object,
        )
    )

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_dir_shap = np.mean(shap_matrix, axis=0)
    activation_freq = np.mean(np.abs(shap_matrix) > 1e-6, axis=0)

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "feature_type": [get_feature_type(f) for f in feature_names],
        "mean_abs_shap": mean_abs_shap,
        "mean_directional_shap": mean_dir_shap,
        "activation_frequency": activation_freq,
    })

    return df_importance.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)


def analyze_prediction_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.14,
) -> dict[str, Any]:
    """
    Perform granular error analysis by slicing the dataset across description length,
    missing/empty descriptions, and suspicious language markers.
    """
    y_pred = (y_prob >= threshold).astype(int)
    descriptions = df["description"].fillna("").astype(str).tolist()

    # Word count per text
    word_counts = np.array([len(t.split()) for t in descriptions])

    # Slices by text length
    slices = {
        "Very Short (< 50 words)": word_counts < 50,
        "Short (50-150 words)": (word_counts >= 50) & (word_counts < 150),
        "Medium (150-300 words)": (word_counts >= 150) & (word_counts < 300),
        "Long (>= 300 words)": word_counts >= 300,
        "Empty/Whitespace Only": np.array([len(t.strip()) == 0 for t in descriptions]),
    }

    # Slices by domain signals (using extractor regex patterns)
    extractor = SuspiciousLanguageExtractor()
    extracted_features = np.array([extractor._extract_from_single_text(t) for t in descriptions])
    feat_names = extractor.FEATURE_NAMES

    # Map signal indices
    idx_has_email = feat_names.index("has_email")
    idx_has_phone = feat_names.index("has_phone")
    idx_has_url = feat_names.index("has_url")
    idx_urgency = feat_names.index("urgency_score")
    idx_evasion = feat_names.index("contact_evasion_score")

    slices["Has Email Marker"] = extracted_features[:, idx_has_email] > 0
    slices["Has Phone Marker"] = extracted_features[:, idx_has_phone] > 0
    slices["Has URL Marker"] = extracted_features[:, idx_has_url] > 0
    slices["Has Urgency Signals"] = extracted_features[:, idx_urgency] > 0
    slices["Has Contact Evasion Signals"] = extracted_features[:, idx_evasion] > 0

    results: dict[str, dict[str, Any]] = {}

    for slice_name, mask in slices.items():
        count = int(mask.sum())
        if count == 0:
            results[slice_name] = {
                "sample_count": 0,
                "pct_of_total": 0.0,
                "fraud_count": 0,
                "legit_count": 0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        precision = round(float(tp / (tp + fp) * 100), 2) if (tp + fp) > 0 else None
        recall = round(float(tp / (tp + fn) * 100), 2) if (tp + fn) > 0 else None
        f1 = (
            round(float(2 * (precision / 100) * (recall / 100) / ((precision / 100) + (recall / 100))), 4)
            if precision and recall and (precision + recall) > 0
            else None
        )

        results[slice_name] = {
            "sample_count": count,
            "pct_of_total": round(float(count / len(df) * 100), 2),
            "fraud_count": int(yt.sum()),
            "legit_count": int((1 - yt).sum()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision_pct": precision,
            "recall_pct": recall,
            "f1_score": f1,
        }

    return results


def extract_representative_cohorts(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model: Any,
    pipeline: CompositeFeaturePipeline,
    threshold: float = 0.14,
    n_per_cohort: int = 2,
    random_state: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract representative True Positives, False Positives, True Negatives,
    and False Negatives with detailed individual SHAP explanations.
    """
    rng = np.random.default_rng(random_state)
    y_pred = (y_prob >= threshold).astype(int)

    tp_indices = np.where((y_pred == 1) & (y_true == 1))[0]
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    tn_indices = np.where((y_pred == 0) & (y_true == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]

    cohort_indices = {
        "true_positives": tp_indices,
        "false_positives": fp_indices,
        "true_negatives": tn_indices,
        "false_negatives": fn_indices,
    }

    cohort_explanations: dict[str, list[dict[str, Any]]] = {}

    for cohort_name, indices in cohort_indices.items():
        selected = []
        if len(indices) > 0:
            chosen = (
                rng.choice(indices, size=min(n_per_cohort, len(indices)), replace=False)
                if len(indices) >= n_per_cohort
                else indices
            )
            for idx in chosen:
                row_text = str(df.iloc[idx]["description"]) if "description" in df.columns else ""
                prob = float(y_prob[idx])
                label = int(y_true[idx])
                explanation = explain_candidate_prediction(
                    job_description=row_text,
                    model=model,
                    pipeline=pipeline,
                    top_k=5,
                )
                explanation["cohort"] = cohort_name
                explanation["true_label"] = "Fraud" if label == 1 else "Legitimate"
                explanation["predicted_class"] = "Fraud" if y_pred[idx] == 1 else "Legitimate"
                explanation["index"] = int(idx)
                selected.append(explanation)

        cohort_explanations[cohort_name] = selected

    return cohort_explanations
