"""
Regression and contract compatibility tests for JobClarity.
"""

from pathlib import Path
import pytest

from src.api import predictor
from src.api.predictor import predict_job
from src.api.schemas import PredictionRequest, PredictionResponse
from src.preprocessing.text_cleaner import clean_text


@pytest.mark.regression
class TestRegression:
    """Regression suite guarding against previously identified failure modes."""

    def test_frontend_contract_field_compatibility(
        self, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        """API prediction output must match the exact TypeScript types expected by JobClarity frontend."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(
            "src.api.predictor.explain_prediction",
            lambda text, top_k=10: [
                {"feature": "developer", "impact": -0.65, "effect": "Decreases fraud risk"}
            ],
        )

        res = predict_job(sample_legitimate_text)

        # Expected frontend fields
        expected_fields = {
            "prediction",
            "fraud_probability",
            "fraud_risk_score",
            "risk_level",
            "top_reasons",
            "model_version",
        }
        assert set(res.keys()) == expected_fields

        # Type checks
        assert isinstance(res["prediction"], str)
        assert res["prediction"] in ["Real", "Fake"]
        assert isinstance(res["fraud_probability"], float)
        assert isinstance(res["fraud_risk_score"], int)
        assert res["risk_level"] in ["Low", "Medium", "High", "Critical"]
        assert isinstance(res["top_reasons"], list)
        assert isinstance(res["model_version"], str)

        # Reason schema
        for item in res["top_reasons"]:
            assert "feature" in item
            assert "impact" in item
            assert "effect" in item
            assert item["effect"] in ["Increases fraud risk", "Decreases fraud risk"]

    def test_pydantic_schema_validation(self, sample_legitimate_text):
        """Pydantic schemas should enforce minimum length constraints and type guarantees."""
        # Valid request
        valid_req = PredictionRequest(job_description=sample_legitimate_text)
        assert valid_req.job_description == sample_legitimate_text

        # Invalid short description
        with pytest.raises(Exception):
            PredictionRequest(job_description="Too short")

    def test_model_inference_failure_cannot_produce_fake_success(
        self, monkeypatch, fitted_tiny_vectorizer
    ):
        """A failure in underlying model inference must not silently fallback to a default successful prediction."""
        class CrashModel:
            def predict(self, X):
                raise ValueError("Model weight corruption error")

            def predict_proba(self, X):
                raise ValueError("Model probability error")

        monkeypatch.setattr(predictor, "model", CrashModel())
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)

        # Calling predict_job must raise the error rather than returning a default {"prediction": "Real"}
        with pytest.raises(ValueError, match="Model weight corruption error"):
            predict_job("Software engineer backend position open at company")

    def test_production_artifacts_unmodified_by_suite(self, production_artifacts_integrity):
        """Explicitly assert that no test execution touches or modifies production model artifacts in models/."""
        # The production_artifacts_integrity fixture records timestamps before and verifies them after
        pass

    @pytest.mark.parametrize(
        ("raw_input", "expected_substring"),
        [
            ("Contact HR: hr@fake.com", "contact hr"),
            ("Visit https://apply.fake.org", "visit"),
            ("Call 123-456-7890 for quick cash!", "call for quick cash"),
            ("<b>Senior</b> Developer &amp; Architect", "senior developer architect"),
        ],
    )
    def test_preprocessing_sanitizes_common_scam_signatures(self, raw_input, expected_substring):
        """Verify that common scam signatures (contact URLs, phone numbers, emails, HTML formatting) are sanitized."""
        cleaned = clean_text(raw_input)
        assert expected_substring in cleaned
        assert "@" not in cleaned
        assert "http" not in cleaned
        assert "<" not in cleaned
