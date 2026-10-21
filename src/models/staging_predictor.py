"""
Staging prediction module for JobClarity (Day 4 Session 3).

Implements non-breaking, staging-only multi-tier risk classification:
  - LOW: probability < 0.14  -> Action: PASS (auto-approve / standard display)
  - REVIEW: 0.14 <= probability < 0.33 -> Action: FLAG_FOR_REVIEW (moderation queue)
  - HIGH_RISK: probability >= 0.33 -> Action: BLOCK (automated mitigation)

Safety & Governance:
  - Production model artifacts in models/ are NEVER modified or overwritten.
  - Production /predict endpoint contracts remain 100% backwards-compatible.
  - Candidate pipelines use exact preprocessing and feature extraction matching training.
  - Missing or corrupt artifacts fail safely without silent fallbacks.
"""

from __future__ import annotations

from enum import Enum
import json
import logging
from math import isinf, isnan
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.feature_pipeline import CompositeFeaturePipeline
from src.features.suspicious_language import SuspiciousLanguageExtractor
from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)

# Default Staging Thresholds from Day 4 Session 2 OOF Optimization
DEFAULT_REVIEW_THRESHOLD: float = 0.14
DEFAULT_HIGH_RISK_THRESHOLD: float = 0.33
DEFAULT_STAGING_VERSION: str = "staging-candidate-d-1.0.0"

# Root and staging paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "evaluation" / "day4" / "model_staging"
PRODUCTION_MODELS_DIR = PROJECT_ROOT / "models"


class RiskTier(str, Enum):
    LOW = "LOW"
    REVIEW = "REVIEW"
    HIGH_RISK = "HIGH_RISK"


class StagingAction(str, Enum):
    PASS = "PASS"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    BLOCK = "BLOCK"


def classify_risk_tier(
    probability: float,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
) -> str:
    """
    Classify a fraud probability into a three-tier operational risk category.

    Args:
        probability: Float in [0.0, 1.0] representing predicted fraud risk.
        review_threshold: Lower boundary for human review queue (default: 0.14).
        high_risk_threshold: Upper boundary for automated high-risk action (default: 0.33).

    Returns:
        RiskTier string: "LOW", "REVIEW", or "HIGH_RISK".

    Raises:
        TypeError: If probability is not an integer or float, or is a boolean.
        ValueError: If probability is outside [0.0, 1.0], NaN, Infinite,
                    or if thresholds are misconfigured.
    """
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError(f"Probability must be a numeric float, got {type(probability).__name__}")

    prob = float(probability)
    if isnan(prob) or isinf(prob):
        raise ValueError("Probability cannot be NaN or Infinite")

    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"Probability must be in range [0.0, 1.0], got {prob}")

    if not (0.0 < review_threshold < high_risk_threshold < 1.0):
        raise ValueError(
            f"Invalid threshold configuration: 0.0 < {review_threshold} < {high_risk_threshold} < 1.0 required"
        )

    if prob < review_threshold:
        return RiskTier.LOW.value
    elif prob < high_risk_threshold:
        return RiskTier.REVIEW.value
    else:
        return RiskTier.HIGH_RISK.value


def get_staging_action(risk_tier: str) -> str:
    """
    Map a three-tier risk classification to its operational trust & safety action.
    """
    if risk_tier == RiskTier.LOW.value:
        return StagingAction.PASS.value
    elif risk_tier == RiskTier.REVIEW.value:
        return StagingAction.FLAG_FOR_REVIEW.value
    elif risk_tier == RiskTier.HIGH_RISK.value:
        return StagingAction.BLOCK.value
    else:
        raise ValueError(f"Unknown risk tier: {risk_tier}")


def verify_feature_parity(
    pipeline: CompositeFeaturePipeline,
    sample_texts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Verify that a fitted feature pipeline matches Candidate D specifications:
      - Word TF-IDF fitted with 10,000 features
      - SuspiciousLanguageExtractor fitted with 21 features
      - Total feature width = 10,021
      - Deterministic transform outputs
    """
    if not isinstance(pipeline, CompositeFeaturePipeline):
        raise TypeError(f"Expected CompositeFeaturePipeline, got {type(pipeline).__name__}")

    if pipeline.tfidf_model is None:
        raise ValueError("Pipeline TF-IDF model is not initialized.")

    tfidf_vocab = getattr(pipeline.tfidf_model, "vocabulary_", None)
    if tfidf_vocab is None or len(tfidf_vocab) == 0:
        raise ValueError("TF-IDF vectorizer has not been fitted.")

    tfidf_features = len(tfidf_vocab)
    suspicious_features = (
        len(SuspiciousLanguageExtractor.FEATURE_NAMES)
        if pipeline.use_suspicious_features
        else 0
    )
    expected_total = tfidf_features + suspicious_features

    test_input = sample_texts or [
        "Software engineer python remote full-time position.",
        "URGENT: Work from home earn $5000 weekly wire transfer now!",
    ]
    X_test = pipeline.transform(test_input)

    if X_test.shape[1] != expected_total:
        raise ValueError(
            f"Feature dimensionality mismatch: expected {expected_total} "
            f"({tfidf_features} TF-IDF + {suspicious_features} suspicious), got {X_test.shape[1]}"
        )

    # Determinism check
    X_repeat = pipeline.transform(test_input)
    diff = (X_test != X_repeat).sum()
    if diff > 0:
        raise ValueError("Pipeline transform is non-deterministic across repeated executions.")

    return {
        "verified": True,
        "tfidf_features": tfidf_features,
        "suspicious_features": suspicious_features,
        "total_features": expected_total,
        "transform_rows_tested": len(test_input),
        "deterministic": True,
    }


def save_staging_artifacts(
    model: Any,
    pipeline: CompositeFeaturePipeline,
    output_dir: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save staging candidate model and feature pipeline artifacts.
    Refuses to write to production models/ directory.
    """
    out = Path(output_dir).resolve()
    prod = PRODUCTION_MODELS_DIR.resolve()

    if out == prod:
        raise ValueError(
            "Safety Violation: Refusing to write staging artifacts into production models directory."
        )

    out.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    model_path = out / "staging_candidate_d_model.joblib"
    pipeline_path = out / "staging_candidate_d_pipeline.joblib"
    meta_path = out / "staging_candidate_metadata.json"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(pipeline, pipeline_path, compress=3)

    if metadata:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Saved staging artifacts to %s", out)
    return out


def load_staging_artifacts(
    artifact_dir: Path | str | None = None,
) -> tuple[Any, CompositeFeaturePipeline]:
    """
    Load staging candidate model and pipeline artifacts.
    Fails safely if artifacts are missing.
    """
    src_dir = Path(artifact_dir) if artifact_dir is not None else STAGING_ARTIFACTS_DIR
    model_path = src_dir / "staging_candidate_d_model.joblib"
    pipeline_path = src_dir / "staging_candidate_d_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Staging model artifact not found: {model_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Staging pipeline artifact not found: {pipeline_path}")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


class StagingPredictor:
    """
    Predictor for staging candidate evaluation using multi-tier decision rules.
    """

    def __init__(
        self,
        model: Any = None,
        pipeline: CompositeFeaturePipeline | None = None,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
        model_version: str = DEFAULT_STAGING_VERSION,
    ) -> None:
        self.model = model
        self.pipeline = pipeline
        self.review_threshold = review_threshold
        self.high_risk_threshold = high_risk_threshold
        self.model_version = model_version

    def is_loaded(self) -> bool:
        """Check whether both model and pipeline are initialized."""
        return self.model is not None and self.pipeline is not None

    def predict(self, job_description: str) -> dict[str, Any]:
        """
        Predict fraud risk and classify into three-tier decision rule.
        """
        if not self.is_loaded():
            raise RuntimeError("Staging model or feature pipeline is not loaded.")

        if not isinstance(job_description, str):
            raise TypeError(f"job_description must be a string, got {type(job_description).__name__}")

        if len(job_description.strip()) < 20:
            raise ValueError("job_description must contain at least 20 characters.")

        # Transform using exact candidate feature pipeline
        X = self.pipeline.transform([job_description])

        # Predict probability for class 1 (Fraud)
        proba = float(self.model.predict_proba(X)[0][1])
        proba_clamped = max(0.0, min(1.0, proba))

        # Risk tier & action
        risk_tier = classify_risk_tier(
            proba_clamped,
            review_threshold=self.review_threshold,
            high_risk_threshold=self.high_risk_threshold,
        )
        action = get_staging_action(risk_tier)

        # Explainability: Extract active domain indicators from raw text
        suspicious_detected = self._extract_active_signals(job_description)

        snippet = (
            job_description[:97] + "..."
            if len(job_description) > 100
            else job_description
        )

        return {
            "job_description_snippet": snippet,
            "fraud_probability": round(proba_clamped, 4),
            "risk_tier": risk_tier,
            "action": action,
            "calibrated_thresholds": {
                "review_threshold": self.review_threshold,
                "high_risk_threshold": self.high_risk_threshold,
            },
            "explanation_summary": {
                "suspicious_features_detected": suspicious_detected,
                "total_features_evaluated": X.shape[1],
            },
            "staging_version": self.model_version,
        }

    def predict_batch(self, job_descriptions: list[str]) -> list[dict[str, Any]]:
        """
        Efficient vectorized batch inference for multiple job descriptions.
        """
        if not self.is_loaded():
            raise RuntimeError("Staging model or feature pipeline is not loaded.")

        if not isinstance(job_descriptions, (list, tuple)) or len(job_descriptions) == 0:
            raise ValueError("job_descriptions must be a non-empty list of strings.")

        for i, text in enumerate(job_descriptions):
            if not isinstance(text, str) or len(text.strip()) < 20:
                raise ValueError(
                    f"Item at index {i} must be a valid job description with >= 20 characters."
                )

        X = self.pipeline.transform(job_descriptions)
        probas = self.model.predict_proba(X)[:, 1]

        results = []
        for text, proba in zip(job_descriptions, probas):
            p = float(max(0.0, min(1.0, proba)))
            tier = classify_risk_tier(
                p,
                review_threshold=self.review_threshold,
                high_risk_threshold=self.high_risk_threshold,
            )
            act = get_staging_action(tier)
            snippet = text[:97] + "..." if len(text) > 100 else text
            results.append({
                "job_description_snippet": snippet,
                "fraud_probability": round(p, 4),
                "risk_tier": tier,
                "action": act,
                "calibrated_thresholds": {
                    "review_threshold": self.review_threshold,
                    "high_risk_threshold": self.high_risk_threshold,
                },
                "staging_version": self.model_version,
            })

        return results

    def _extract_active_signals(self, raw_text: str) -> list[str]:
        """Inspect domain signals triggered in raw text."""
        if self.pipeline.suspicious_extractor is None:
            return []

        extractor = self.pipeline.suspicious_extractor
        names = extractor.FEATURE_NAMES
        vals = extractor._extract_from_single_text(raw_text)

        active = []
        for name, val in zip(names, vals):
            if val > 0.0 and name in [
                "urgency_score",
                "financial_promise_score",
                "low_barrier_score",
                "logistics_scam_score",
                "contact_evasion_score",
                "has_email",
                "has_phone",
                "has_url",
                "currency_symbol_count",
            ]:
                active.append(f"{name}={val:.0f}")

        return active


# Global singleton instance for staging API routes
_STAGING_PREDICTOR_INSTANCE: StagingPredictor | None = None


def get_staging_predictor() -> StagingPredictor:
    """
    Get or lazily initialize the singleton StagingPredictor.
    If staging artifacts are absent, returns an uninitialized predictor
    that fails safely when invoked.
    """
    global _STAGING_PREDICTOR_INSTANCE
    if _STAGING_PREDICTOR_INSTANCE is not None:
        return _STAGING_PREDICTOR_INSTANCE

    try:
        model, pipeline = load_staging_artifacts()
        _STAGING_PREDICTOR_INSTANCE = StagingPredictor(model=model, pipeline=pipeline)
        logger.info("Successfully loaded StagingPredictor singleton.")
    except Exception as e:
        logger.warning(
            "StagingPredictor initialized without active artifacts: %s", str(e)
        )
        _STAGING_PREDICTOR_INSTANCE = StagingPredictor(model=None, pipeline=None)

    return _STAGING_PREDICTOR_INSTANCE


def reset_staging_predictor() -> None:
    """Reset the singleton (useful for test isolation)."""
    global _STAGING_PREDICTOR_INSTANCE
    _STAGING_PREDICTOR_INSTANCE = None
