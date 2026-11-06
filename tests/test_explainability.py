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


@pytest.mark.unit
class TestNativeXGBoostExplainability:
    """Tests verifying native XGBoost TreeSHAP integration and mathematical validity."""

    def test_legitimate_jd_returns_non_empty_explanations(self, sample_legitimate_text):
        """A realistic legitimate job description returns populated top_reasons."""
        from src.explainability.shap_explainer import explain_prediction, init_explainer

        init_explainer()
        reasons = explain_prediction(sample_legitimate_text, top_k=10)
        assert isinstance(reasons, list)
        assert len(reasons) > 0
        assert len(reasons) <= 10
        assert all("feature" in r and "impact" in r and "effect" in r for r in reasons)

    def test_suspicious_jd_returns_non_empty_explanations(self, sample_fraudulent_text):
        """A realistic suspicious job description returns populated high-risk reasons."""
        from src.explainability.shap_explainer import explain_prediction, init_explainer

        init_explainer()
        reasons = explain_prediction(sample_fraudulent_text, top_k=10)
        assert isinstance(reasons, list)
        assert len(reasons) > 0
        # Suspicious posting should contain fraud-increasing signals
        assert any(r["effect"] == "Increases fraud risk" for r in reasons)

    def test_positive_and_negative_shap_impact_effect_labels(self, sample_fraudulent_text):
        """Positive SHAP impacts must map to 'Increases fraud risk' and negative to 'Decreases fraud risk'."""
        from src.explainability.shap_explainer import explain_prediction

        reasons = explain_prediction(sample_fraudulent_text, top_k=10)
        for r in reasons:
            if r["impact"] > 0:
                assert r["effect"] == "Increases fraud risk"
            elif r["impact"] < 0:
                assert r["effect"] == "Decreases fraud risk"

    def test_features_map_to_correct_vectorizer_vocabulary_indices(self, sample_legitimate_text):
        """All returned feature names must belong to the fitted vectorizer vocabulary."""
        from src.explainability.shap_explainer import explain_prediction, vectorizer

        reasons = explain_prediction(sample_legitimate_text, top_k=10)
        vocab_features = set(vectorizer.get_feature_names_out())
        assert len(vocab_features) == 10000

        for r in reasons:
            assert r["feature"] in vocab_features

    def test_contributions_reconstruct_model_margin_and_probability(self, sample_fraudulent_text):
        """Mathematical check: sum of TreeSHAP contributions must reconstruct the raw model margin."""
        import scipy.special
        import xgboost as xgb
        from src.explainability.shap_explainer import booster, model, vectorizer
        from src.preprocessing.text_cleaner import clean_text

        cleaned = clean_text(sample_fraudulent_text)
        X = vectorizer.transform([cleaned])
        raw_booster = getattr(model, "get_booster", lambda: booster)()
        dmat = xgb.DMatrix(X)

        # 1. Native TreeSHAP contribs (features + bias)
        contribs = raw_booster.predict(dmat, pred_contribs=True)[0]
        sum_contribs = float(contribs.sum())

        # 2. Raw margin from booster
        raw_margin = float(raw_booster.predict(dmat, output_margin=True)[0])

        # Sum of SHAP must equal raw logit margin within tight tolerance
        assert np.isclose(sum_contribs, raw_margin, atol=1e-4)

        # 3. Logistic transform of reconstructed margin must match predict_proba
        predicted_prob = float(model.predict_proba(X)[0][1])
        reconstructed_prob = float(scipy.special.expit(sum_contribs))
        assert np.isclose(reconstructed_prob, predicted_prob, atol=1e-4)

    def test_empty_or_invalid_input_returns_empty_list(self):
        """Empty, whitespace-only, or non-string input safely yields an empty list."""
        from src.explainability.shap_explainer import explain_prediction

        assert explain_prediction("") == []
        assert explain_prediction("   \n\t  ") == []
        assert explain_prediction(None) == []  # type: ignore[arg-type]
        assert explain_prediction(12345) == []  # type: ignore[arg-type]

    def test_explanation_failure_does_not_break_prediction(self, monkeypatch, sample_legitimate_text):
        """Prediction pipeline must remain 100% operational even if explain_prediction fails."""
        from src.api.predictor import predict_job

        monkeypatch.setattr(
            "src.api.predictor.explain_prediction",
            MagicMock(side_effect=RuntimeError("Simulated explainer crash")),
        )
        # Should not raise exception
        try:
            result = predict_job(sample_legitimate_text)
            assert result["prediction"] in ("Real", "Fake")
            assert "fraud_probability" in result
        except Exception:
            pytest.fail("predict_job should never crash when explain_prediction raises an exception.")

    def test_api_predict_endpoint_response_schema_populated(self, client, sample_legitimate_text, sample_fraudulent_text):
        """End-to-end /predict endpoint returns fully compliant schema with populated top_reasons."""
        from src.api.schemas import PredictionResponse

        # Legitimate test
        resp_legit = client.post("/predict", json={"job_description": sample_legitimate_text})
        assert resp_legit.status_code == 200
        data_legit = resp_legit.json()
        validated_legit = PredictionResponse(**data_legit)
        assert validated_legit.prediction == "Real"
        assert len(validated_legit.top_reasons) > 0
        assert all(isinstance(r["impact"], float) for r in validated_legit.top_reasons)

        # Suspicious test
        resp_fraud = client.post("/predict", json={"job_description": sample_fraudulent_text})
        assert resp_fraud.status_code == 200
        data_fraud = resp_fraud.json()
        validated_fraud = PredictionResponse(**data_fraud)
        assert validated_fraud.prediction == "Fake"
        assert len(validated_fraud.top_reasons) > 0
        assert any(r["effect"] == "Increases fraud risk" for r in validated_fraud.top_reasons)
