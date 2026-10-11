"""
Unit tests for SHAP explainability utilities in src/explainability/shap_explainer.py.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.explainability import shap_explainer
from src.explainability.shap_explainer import explain_prediction


@pytest.mark.unit
class TestExplainability:
    """Test suite for SHAP feature attribution and formatting."""

    def test_explain_prediction_formatting_and_direction(self, monkeypatch, fitted_tiny_vectorizer):
        """Top features should be correctly ranked, rounded, and annotated with effect direction."""
        # Mock tree explainer returning 1 sample with known shap values
        mock_tree_explainer = MagicMock()
        # Create dummy shap values for the vocabulary size
        vocab_size = len(fitted_tiny_vectorizer.get_feature_names_out())
        dummy_shap = np.zeros(vocab_size)
        dummy_shap[0] = 1.45678  # Increases fraud risk
        dummy_shap[1] = -0.89234  # Decreases fraud risk
        mock_tree_explainer.shap_values.return_value = [dummy_shap]

        monkeypatch.setattr(shap_explainer, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(shap_explainer, "explainer", mock_tree_explainer)

        reasons = explain_prediction("Work from home python developer", top_k=2)

        assert isinstance(reasons, list)
        assert len(reasons) == 2

        # Check structure
        for item in reasons:
            assert "feature" in item
            assert "impact" in item
            assert "effect" in item
            assert isinstance(item["impact"], float)
            assert item["effect"] in ["Increases fraud risk", "Decreases fraud risk"]

        # Check ranking: highest abs value first
        assert reasons[0]["impact"] == 1.4568
        assert reasons[0]["effect"] == "Increases fraud risk"
        assert reasons[1]["impact"] == -0.8923
        assert reasons[1]["effect"] == "Decreases fraud risk"

    def test_explain_prediction_top_k_limit(self, monkeypatch, fitted_tiny_vectorizer):
        """top_k argument should constrain the number of returned feature reasons."""
        mock_tree_explainer = MagicMock()
        vocab_size = len(fitted_tiny_vectorizer.get_feature_names_out())
        dummy_shap = np.random.default_rng(42).standard_normal(vocab_size)
        mock_tree_explainer.shap_values.return_value = [dummy_shap]

        monkeypatch.setattr(shap_explainer, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(shap_explainer, "explainer", mock_tree_explainer)

        reasons_top1 = explain_prediction("Job description text here", top_k=1)
        assert len(reasons_top1) == 1

        reasons_top5 = explain_prediction("Job description text here", top_k=5)
        assert len(reasons_top5) == min(5, vocab_size)

    def test_explain_prediction_handles_none_explainer_safely(self, monkeypatch, fitted_tiny_vectorizer):
        """When explainer is None, explain_prediction should safely return an empty list."""
        monkeypatch.setattr(shap_explainer, "explainer", None)
        monkeypatch.setattr(shap_explainer, "vectorizer", fitted_tiny_vectorizer)

        assert explain_prediction("Job description") == []

    def test_explain_prediction_handles_none_vectorizer_safely(self, monkeypatch):
        """When vectorizer is None, explain_prediction should safely return an empty list."""
        monkeypatch.setattr(shap_explainer, "vectorizer", None)
        assert explain_prediction("Job description") == []

    def test_explain_prediction_handles_calculation_exception_safely(self, monkeypatch, fitted_tiny_vectorizer):
        """If SHAP tree explainer throws an exception, explain_prediction should catch it and return []."""
        mock_tree_explainer = MagicMock()
        mock_tree_explainer.shap_values.side_effect = RuntimeError("SHAP computation failed")

        monkeypatch.setattr(shap_explainer, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(shap_explainer, "explainer", mock_tree_explainer)

        reasons = explain_prediction("Some job posting")
        assert reasons == []

    def test_shap_explainer_module_import_handles_loading_exception(self, monkeypatch):
        """Module import should catch loading exceptions and set explainer to None."""
        import importlib
        import src.explainability.shap_explainer as s

        monkeypatch.setattr("src.models.model_loader.load_model", MagicMock(side_effect=Exception("Load error")))
        importlib.reload(s)
        assert s.model is None
        assert s.explainer is None

        # Restore
        monkeypatch.undo()
        importlib.reload(s)
