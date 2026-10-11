"""
Unit tests for inference logic in src/api/predictor.py.
"""

import numpy as np
import pytest

from src.api import predictor
from src.api.predictor import predict_job


@pytest.mark.unit
class TestPredictor:
    """Test suite for predictor inference orchestration."""

    def test_predict_job_legitimate(
        self, monkeypatch, fitted_tiny_vectorizer, dummy_legitimate_model, sample_legitimate_text
    ):
        """A legitimate prediction (class 0) should return Real with appropriate low risk."""
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(
            "src.api.predictor.explain_prediction",
            lambda text, top_k=10: [
                {"feature": "engineer", "impact": -0.85, "effect": "Decreases fraud risk"}
            ],
        )

        result = predict_job(sample_legitimate_text)

        assert isinstance(result, dict)
        assert result["prediction"] == "Real"
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["fraud_probability"] == 0.08
        assert result["fraud_risk_score"] == 8
        assert result["risk_level"] == "Low"
        assert result["model_version"] == "1.0.0"
        assert len(result["top_reasons"]) == 1
        assert result["top_reasons"][0]["feature"] == "engineer"

    def test_predict_job_fraudulent(
        self, monkeypatch, fitted_tiny_vectorizer, dummy_fraudulent_model, sample_fraudulent_text
    ):
        """A fraudulent prediction (class 1) should return Fake with Critical/High risk."""
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(predictor, "model", dummy_fraudulent_model)
        monkeypatch.setattr(
            "src.api.predictor.explain_prediction",
            lambda text, top_k=10: [
                {"feature": "wire", "impact": 1.25, "effect": "Increases fraud risk"},
                {"feature": "cash", "impact": 0.95, "effect": "Increases fraud risk"},
            ],
        )

        result = predict_job(sample_fraudulent_text)

        assert isinstance(result, dict)
        assert result["prediction"] == "Fake"
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["fraud_probability"] == 0.92
        assert result["fraud_risk_score"] == 92
        assert result["risk_level"] == "Critical"
        assert result["model_version"] == "1.0.0"
        assert len(result["top_reasons"]) == 2

    @pytest.mark.parametrize(
        ("prob", "expected_score", "expected_risk"),
        [
            (0.00, 0, "Low"),
            (0.25, 25, "Low"),
            (0.26, 26, "Medium"),
            (0.50, 50, "Medium"),
            (0.51, 51, "High"),
            (0.75, 75, "High"),
            (0.76, 76, "Critical"),
            (1.00, 100, "Critical"),
        ],
    )
    def test_risk_level_threshold_boundaries(
        self, monkeypatch, fitted_tiny_vectorizer, prob, expected_score, expected_risk
    ):
        """All risk level thresholds (Low, Medium, High, Critical) should be correctly categorized."""
        from tests.conftest import DummyModel

        pred_class = 1 if prob >= 0.5 else 0
        custom_model = DummyModel(pred_class=pred_class, fraud_prob=prob)

        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(predictor, "model", custom_model)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        result = predict_job("Testing risk thresholds with sufficient length string")
        assert result["fraud_risk_score"] == expected_score
        assert result["risk_level"] == expected_risk

    def test_predict_job_deterministic(
        self, monkeypatch, fitted_tiny_vectorizer, dummy_legitimate_model, sample_legitimate_text
    ):
        """Consecutive calls with deterministic models must produce identical results."""
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        res1 = predict_job(sample_legitimate_text)
        res2 = predict_job(sample_legitimate_text)

        assert res1 == res2

    def test_predict_job_unloaded_model_raises_runtime_error(self, monkeypatch):
        """Calling predict_job when model or vectorizer is None should raise RuntimeError."""
        monkeypatch.setattr(predictor, "model", None)
        with pytest.raises(RuntimeError, match="Model or vectorizer artifact is not loaded"):
            predict_job("Valid job description that meets character count requirements")

    def test_predict_job_unloaded_vectorizer_raises_runtime_error(self, monkeypatch, dummy_legitimate_model):
        """Calling predict_job when vectorizer is None should raise RuntimeError."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", None)
        with pytest.raises(RuntimeError, match="Model or vectorizer artifact is not loaded"):
            predict_job("Valid job description that meets character count requirements")

    def test_predictor_module_import_handles_loading_exception(self, monkeypatch):
        """Module import should catch loading exceptions and set model/vectorizer to None."""
        import importlib
        from unittest.mock import MagicMock
        import src.api.predictor as p

        monkeypatch.setattr("src.models.model_loader.load_model", MagicMock(side_effect=Exception("Load error")))
        importlib.reload(p)
        assert p.model is None
        assert p.vectorizer is None

        # Restore
        monkeypatch.undo()
        importlib.reload(p)
