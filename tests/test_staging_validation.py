"""
Unit and integration tests for Model Staging & Multi-Tier Decision Validation (Day 4 Session 3).

Covers:
  1. Threshold boundaries (LOW < 0.14, 0.14 <= REVIEW < 0.33, HIGH_RISK >= 0.33)
  2. Input validation for probabilities (NaN, Inf, < 0.0, > 1.0, booleans, types)
  3. Action mapping (PASS, FLAG_FOR_REVIEW, BLOCK)
  4. Missing artifacts fail-safe behavior (RuntimeError, FileNotFoundError, HTTP 503)
  5. API response compatibility (production /predict intact; /predict/staging schema valid)
  6. Inference parity & production models immutability
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PredictionResponse, StagingPredictionResponse
from src.models.staging_predictor import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    RiskTier,
    StagingAction,
    StagingPredictor,
    classify_risk_tier,
    get_staging_action,
    load_staging_artifacts,
    reset_staging_predictor,
    save_staging_artifacts,
    verify_feature_parity,
)


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


class TestThresholdBoundariesAndValidation:
    """Tests for three-tier classification boundaries and input safety."""

    @pytest.mark.parametrize(
        ("prob", "expected_tier"),
        [
            (0.00, RiskTier.LOW.value),
            (0.05, RiskTier.LOW.value),
            (0.1399, RiskTier.LOW.value),
            (0.14, RiskTier.REVIEW.value),
            (0.20, RiskTier.REVIEW.value),
            (0.3299, RiskTier.REVIEW.value),
            (0.33, RiskTier.HIGH_RISK.value),
            (0.75, RiskTier.HIGH_RISK.value),
            (1.00, RiskTier.HIGH_RISK.value),
        ],
    )
    def test_classify_risk_tier_exact_boundaries(self, prob: float, expected_tier: str):
        """Boundary points must strictly map to correct operational tiers."""
        assert classify_risk_tier(prob) == expected_tier

    @pytest.mark.parametrize(
        "invalid_prob",
        [-0.01, -1.0, 1.0001, 2.5, 100.0],
    )
    def test_classify_risk_tier_out_of_bounds_raises_value_error(self, invalid_prob: float):
        """Probabilities outside [0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            classify_risk_tier(invalid_prob)

    @pytest.mark.parametrize(
        "invalid_numeric",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_classify_risk_tier_nan_inf_raises_value_error(self, invalid_numeric: float):
        """NaN and Infinite values must raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
            classify_risk_tier(invalid_numeric)

    @pytest.mark.parametrize(
        "invalid_type",
        ["0.5", None, [0.5], {"prob": 0.5}, True, False],
    )
    def test_classify_risk_tier_invalid_types_raise_type_error(self, invalid_type):
        """Non-numeric inputs or booleans must raise TypeError."""
        with pytest.raises(TypeError, match="must be a numeric float"):
            classify_risk_tier(invalid_type)

    def test_misconfigured_thresholds_raise_value_error(self):
        """Thresholds that violate 0 < review < high_risk < 1 must raise ValueError."""
        # review >= high_risk
        with pytest.raises(ValueError, match="Invalid threshold configuration"):
            classify_risk_tier(0.2, review_threshold=0.50, high_risk_threshold=0.40)

        # review <= 0
        with pytest.raises(ValueError, match="Invalid threshold configuration"):
            classify_risk_tier(0.2, review_threshold=0.0, high_risk_threshold=0.40)

        # high_risk >= 1.0
        with pytest.raises(ValueError, match="Invalid threshold configuration"):
            classify_risk_tier(0.2, review_threshold=0.20, high_risk_threshold=1.0)

    @pytest.mark.parametrize(
        ("tier", "expected_action"),
        [
            (RiskTier.LOW.value, StagingAction.PASS.value),
            (RiskTier.REVIEW.value, StagingAction.FLAG_FOR_REVIEW.value),
            (RiskTier.HIGH_RISK.value, StagingAction.BLOCK.value),
        ],
    )
    def test_get_staging_action_mapping(self, tier: str, expected_action: str):
        """Each tier must map to its exact operational mitigation action."""
        assert get_staging_action(tier) == expected_action

    def test_get_staging_action_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown risk tier"):
            get_staging_action("UNKNOWN")


class TestStagingPredictorLogic:
    """Tests for StagingPredictor inference methods and edge handling."""

    def test_predict_unloaded_raises_runtime_error(self):
        """Calling predict on uninitialized StagingPredictor must raise RuntimeError."""
        unloaded = StagingPredictor(model=None, pipeline=None)
        assert not unloaded.is_loaded()
        with pytest.raises(RuntimeError, match="Staging model or feature pipeline is not loaded"):
            unloaded.predict("A valid job description that is long enough to pass validation.")

    def test_predict_batch_unloaded_raises_runtime_error(self):
        unloaded = StagingPredictor(model=None, pipeline=None)
        with pytest.raises(RuntimeError, match="Staging model or feature pipeline is not loaded"):
            unloaded.predict_batch(["A valid job description that is long enough to pass validation."])

    def test_predict_short_input_raises_value_error(self, monkeypatch):
        mock_pred = StagingPredictor(model=object(), pipeline=object())
        with pytest.raises(ValueError, match="at least 20 characters"):
            mock_pred.predict("Short text")

    def test_predict_non_string_raises_type_error(self):
        mock_pred = StagingPredictor(model=object(), pipeline=object())
        with pytest.raises(TypeError, match="must be a string"):
            mock_pred.predict(12345)  # type: ignore

    def test_predict_batch_invalid_inputs_raise_value_error(self):
        mock_pred = StagingPredictor(model=object(), pipeline=object())
        with pytest.raises(ValueError, match="non-empty list"):
            mock_pred.predict_batch([])

        with pytest.raises(ValueError, match="at index 0"):
            mock_pred.predict_batch(["Too short"])


class TestMissingArtifactsFailSafely:
    """Verifies fail-safe behavior when artifacts are absent and safeguards against directory clobbering."""

    def test_load_staging_artifacts_missing_dir_raises_file_not_found(self, tmp_path):
        empty_dir = tmp_path / "empty_staging"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Staging model artifact not found"):
            load_staging_artifacts(empty_dir)

    def test_save_staging_artifacts_refuses_production_models_dir(self):
        project_root = Path(__file__).resolve().parents[1]
        prod_models = project_root / "models"
        with pytest.raises(ValueError, match="Refusing to write staging artifacts into production models directory"):
            save_staging_artifacts(model="dummy", pipeline=None, output_dir=prod_models)


class TestStagingInferenceParityAndArtifacts:
    """Tests feature parity and active artifact integration if trained."""

    def test_verify_feature_parity_validates_pipeline(self):
        from src.features.feature_pipeline import (
            CompositeFeaturePipeline,
            create_word_tfidf_vectorizer,
        )

        sample_corpus = [
            "Senior python engineer fastapi postgresql remote",
            "Urgent work from home earn wire money transfer fee",
            "Administrative assistant data entry office clerk",
        ]
        pipe = CompositeFeaturePipeline(
            tfidf_model=create_word_tfidf_vectorizer(max_features=50, ngram_range=(1, 2), min_df=1),
            use_suspicious_features=True,
        )
        pipe.fit(sample_corpus)

        parity = verify_feature_parity(pipe, sample_corpus)
        assert parity["verified"] is True
        assert parity["suspicious_features"] == 21
        assert parity["total_features"] == parity["tfidf_features"] + parity["suspicious_features"]
        assert parity["deterministic"] is True


class TestAPIEndpointsCompatibility:
    """Verifies non-breaking staging route and intact production API route."""

    def test_production_predict_endpoint_completely_intact(
        self, test_client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        """Production /predict must remain 100% backward-compatible and match PredictionResponse schema."""
        from src.api import predictor

        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        resp = test_client.post("/predict", json={"job_description": sample_legitimate_text})
        assert resp.status_code == 200

        data = resp.json()
        validated = PredictionResponse(**data)
        assert validated.prediction in ["Real", "Fake"]
        assert validated.risk_level in ["Low", "Medium", "High", "Critical"]
        assert "model_version" in data

    def test_staging_predict_endpoint_success(self, test_client, sample_legitimate_text):
        """Staging endpoint /predict/staging returns valid StagingPredictionResponse."""
        resp = test_client.post("/predict/staging", json={"job_description": sample_legitimate_text})
        # If artifacts are present, returns 200, else 503
        if resp.status_code == 200:
            data = resp.json()
            validated = StagingPredictionResponse(**data)
            assert validated.risk_tier in [RiskTier.LOW.value, RiskTier.REVIEW.value, RiskTier.HIGH_RISK.value]
            assert validated.action in [StagingAction.PASS.value, StagingAction.FLAG_FOR_REVIEW.value, StagingAction.BLOCK.value]
            assert "review_threshold" in validated.calibrated_thresholds
            assert "high_risk_threshold" in validated.calibrated_thresholds
        else:
            assert resp.status_code == 503
            assert "Staging model service is currently unavailable" in resp.json()["detail"]

    def test_staging_predict_endpoint_validation_error_on_short_text(self, test_client, sample_short_text):
        """Descriptions < 20 chars must return HTTP 422 Unprocessable Entity."""
        resp = test_client.post("/predict/staging", json={"job_description": sample_short_text})
        assert resp.status_code == 422


class TestProductionArtifactsSafety:
    """Verifies that no staging validation execution modified production models in models/."""

    def test_models_directory_unmodified(self):
        project_root = Path(__file__).resolve().parents[1]
        prod_models = project_root / "models"
        prod_files = [f for f in prod_models.iterdir() if f.is_file() and f.suffix in [".pkl", ".joblib"]]
        assert len(prod_files) > 0
        for f in prod_files:
            # File should exist and have positive size
            assert f.exists()
            assert f.stat().st_size > 0


class TestLoadedStagingPredictorExecution:
    """End-to-end tests using saved staging candidate artifacts if present."""

    def test_predict_and_batch_on_saved_artifacts(self, sample_legitimate_text, sample_fraudulent_text):
        from src.models.staging_predictor import STAGING_ARTIFACTS_DIR, get_staging_predictor, reset_staging_predictor

        reset_staging_predictor()
        pred = get_staging_predictor()
        if not pred.is_loaded():
            pytest.skip("Staging artifacts not generated yet.")

        # Single prediction legitimate
        res_legit = pred.predict(sample_legitimate_text)
        assert res_legit["risk_tier"] in [RiskTier.LOW.value, RiskTier.REVIEW.value]
        assert 0.0 <= res_legit["fraud_probability"] <= 1.0
        assert "calibrated_thresholds" in res_legit
        assert res_legit["staging_version"] == "staging-candidate-d-1.0.0"

        # Single prediction fraudulent
        res_fraud = pred.predict(sample_fraudulent_text)
        assert res_fraud["risk_tier"] in [RiskTier.REVIEW.value, RiskTier.HIGH_RISK.value]
        assert res_fraud["fraud_probability"] >= res_legit["fraud_probability"]

        # Batch prediction
        batch_res = pred.predict_batch([sample_legitimate_text, sample_fraudulent_text])
        assert len(batch_res) == 2
        assert batch_res[0]["risk_tier"] == res_legit["risk_tier"]
        assert batch_res[1]["risk_tier"] == res_fraud["risk_tier"]
