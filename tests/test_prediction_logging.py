"""
Tests for structured prediction logging (JobClarity Day 7 - Phase B).
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from src.api import predictor
from src.api.schemas import PredictionResponse, StagingPredictionResponse
from src.services.prediction_logger import (
    clear_prediction_logs,
    get_prediction_log_path,
    get_recent_prediction_logs,
    hash_text,
    log_prediction,
)


@pytest.fixture
def temp_log_file(tmp_path):
    """Provides a fresh isolated log file path for testing."""
    log_file = tmp_path / "test_predictions.jsonl"
    return log_file


class TestStructuredPredictionLogging:
    """Unit tests for the prediction logger service."""

    def test_log_prediction_success(self, temp_log_file):
        """Verify successful prediction logging with all required fields."""
        record = log_prediction(
            endpoint="/predict",
            model_name="JobClarity-Baseline",
            model_version="1.0.0",
            probability=0.1234,
            decision_threshold=0.50,
            risk_tier="Low",
            latency_ms=18.45,
            job_description="Senior Python backend engineer with 5 years experience.",
            status="success",
            log_file_path=temp_log_file,
        )

        assert record is not None
        assert "prediction_id" in record
        assert "timestamp_utc" in record
        assert record["endpoint"] == "/predict"
        assert record["model_name"] == "JobClarity-Baseline"
        assert record["model_version"] == "1.0.0"
        assert record["probability"] == 0.1234
        assert record["decision_threshold"] == 0.50
        assert record["risk_tier"] == "Low"
        assert record["latency_ms"] == 18.45
        assert record["status"] == "success"
        assert record["input_char_length"] > 20
        assert record["input_word_count"] > 5
        assert isinstance(record["input_sha256"], str)
        assert len(record["input_sha256"]) == 64

        # Verify file content
        assert temp_log_file.exists()
        lines = temp_log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        stored = json.loads(lines[0])
        assert stored["prediction_id"] == record["prediction_id"]

    def test_privacy_guarantee_no_raw_text_stored(self, temp_log_file):
        """STRICT PRIVACY: Raw job description text and PII must never be stored."""
        sensitive_text = "Contact hiring manager John Doe at secret.email@acme.com with SSN 000-11-2222."
        log_prediction(
            endpoint="/predict",
            model_name="JobClarity-Baseline",
            model_version="1.0.0",
            probability=0.88,
            decision_threshold=0.5,
            risk_tier="High",
            job_description=sensitive_text,
            extra_metadata={"raw_text": "SHOULD_BE_BLOCKED", "job_description": "SHOULD_BE_BLOCKED"},
            log_file_path=temp_log_file,
        )

        content = temp_log_file.read_text(encoding="utf-8")
        assert "secret.email@acme.com" not in content
        assert "John Doe" not in content
        assert "000-11-2222" not in content
        assert "SHOULD_BE_BLOCKED" not in content
        assert sensitive_text not in content

    def test_missing_optional_metadata_handled_gracefully(self, temp_log_file):
        """Missing or None metadata values must not crash logging."""
        record = log_prediction(
            endpoint="/predict",
            model_name="JobClarity-Baseline",
            model_version="1.0.0",
            probability=None,
            decision_threshold=None,
            risk_tier=None,
            latency_ms=None,
            job_description=None,
            log_file_path=temp_log_file,
        )

        assert record is not None
        assert record["probability"] is None
        assert record["decision_threshold"] is None
        assert record["risk_tier"] is None
        assert record["latency_ms"] is None
        assert record["input_char_length"] == 0
        assert record["input_sha256"] is None

    def test_logging_failure_does_not_raise(self, temp_log_file):
        """Simulate an unhandled disk write error and verify function catches it safely."""
        with patch("src.services.prediction_logger.open", side_effect=OSError("Disk write error")):
            result = log_prediction(
                endpoint="/predict",
                model_name="JobClarity-Baseline",
                model_version="1.0.0",
                probability=0.25,
                decision_threshold=0.5,
                risk_tier="Low",
                job_description="Sample job description with valid length.",
                log_file_path=temp_log_file,
            )
            # Logger returns None on failure and does NOT raise an exception
            assert result is None

    def test_get_recent_logs_and_clear(self, temp_log_file):
        """Test retrieving recent logs and clearing the log file."""
        for i in range(5):
            log_prediction(
                endpoint="/predict",
                model_name="JobClarity-Baseline",
                model_version="1.0.0",
                probability=0.1 * i,
                decision_threshold=0.5,
                risk_tier="Low",
                log_file_path=temp_log_file,
            )

        entries = get_recent_prediction_logs(limit=3, log_file_path=temp_log_file)
        assert len(entries) == 3

        clear_prediction_logs(log_file_path=temp_log_file)
        assert not temp_log_file.exists()
        assert get_recent_prediction_logs(log_file_path=temp_log_file) == []


class TestEndpointLoggingIntegration:
    """Integration tests verifying /predict and /predict/staging record logs properly."""

    def test_predict_endpoint_records_prediction_log(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text, tmp_path
    ):
        """Calling /predict should record a structured log entry."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        test_log = tmp_path / "predict_endpoint_test.jsonl"
        with patch("src.services.prediction_logger.get_prediction_log_path", return_value=test_log):
            response = client.post("/predict", json={"job_description": sample_legitimate_text})
            assert response.status_code == 200

            # Verify response schema contract remains 100% untouched
            validated = PredictionResponse(**response.json())
            assert validated.prediction == "Real"

            assert test_log.exists()
            lines = test_log.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["endpoint"] == "/predict"
            assert entry["model_name"] == "JobClarity-Baseline"
            assert entry["probability"] == 0.08
            assert entry["latency_ms"] is not None
            assert entry["input_char_length"] == len(sample_legitimate_text)

    def test_predict_staging_endpoint_records_prediction_log(
        self, client, monkeypatch, sample_legitimate_text, tmp_path
    ):
        """Calling /predict/staging should record a structured log entry with staging metadata."""
        mock_staging_pred = MagicMock()
        mock_staging_pred.is_loaded.return_value = True
        mock_staging_pred.predict.return_value = {
            "job_description_snippet": sample_legitimate_text[:80],
            "fraud_probability": 0.0912,
            "risk_tier": "LOW",
            "action": "PASS",
            "calibrated_thresholds": {"review": 0.14, "high_risk": 0.33},
            "explanation_summary": {"suspicious_features_detected": [], "total_features_evaluated": 10021},
            "staging_version": "staging-candidate-d-1.0.0",
        }
        monkeypatch.setattr("src.api.routes.get_staging_predictor", lambda: mock_staging_pred)

        test_log = tmp_path / "predict_staging_endpoint_test.jsonl"
        with patch("src.services.prediction_logger.get_prediction_log_path", return_value=test_log):
            response = client.post("/predict/staging", json={"job_description": sample_legitimate_text})
            assert response.status_code == 200

            validated = StagingPredictionResponse(**response.json())
            assert validated.action == "PASS"

            assert test_log.exists()
            entry = json.loads(test_log.read_text(encoding="utf-8").strip())
            assert entry["endpoint"] == "/predict/staging"
            assert entry["model_name"] == "JobClarity-Candidate-D"
            assert entry["risk_tier"] == "LOW"
            assert entry["action"] == "PASS"

    def test_logging_failure_does_not_break_predict_inference(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        """CRITICAL: If logging fails completely, /predict must still return HTTP 200 and inference results."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        with patch("src.api.routes.log_prediction", side_effect=RuntimeError("Logging crashed!")):
            response = client.post("/predict", json={"job_description": sample_legitimate_text})
            # Still succeeds!
            assert response.status_code == 200
            assert response.json()["prediction"] == "Real"

    def test_logging_failure_does_not_break_staging_inference(
        self, client, monkeypatch, sample_legitimate_text
    ):
        """CRITICAL: If logging fails completely, /predict/staging must still return HTTP 200."""
        mock_staging_pred = MagicMock()
        mock_staging_pred.is_loaded.return_value = True
        mock_staging_pred.predict.return_value = {
            "job_description_snippet": sample_legitimate_text[:80],
            "fraud_probability": 0.22,
            "risk_tier": "REVIEW",
            "action": "FLAG_FOR_REVIEW",
            "calibrated_thresholds": {"review": 0.14, "high_risk": 0.33},
            "explanation_summary": None,
            "staging_version": "staging-candidate-d-1.0.0",
        }
        monkeypatch.setattr("src.api.routes.get_staging_predictor", lambda: mock_staging_pred)

        with patch("src.api.routes.log_prediction", side_effect=IOError("Storage disk offline!")):
            response = client.post("/predict/staging", json={"job_description": sample_legitimate_text})
            assert response.status_code == 200
            assert response.json()["action"] == "FLAG_FOR_REVIEW"
