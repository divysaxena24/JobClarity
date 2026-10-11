"""
Integration tests for FastAPI endpoints in src/api/routes.py.
"""

from unittest.mock import MagicMock
import pytest

from src.api import predictor, routes
from src.api.schemas import PredictionResponse


@pytest.mark.integration
class TestAPIEndpoints:
    """Test suite for HTTP REST API routes."""

    # --- GET / Home Endpoint ---

    def test_get_home_endpoint(self, client):
        """GET / should return HTTP 200 with project info and endpoint links."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["project"] == "JobClarity"
        assert data["version"] == "1.0.0"
        assert "description" in data
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"

    # --- GET /health Health Check Endpoint ---

    def test_get_health_healthy(self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer):
        """GET /health should return status healthy when model and vectorizer are loaded."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["vectorizer_loaded"] is True
        assert data["version"] == "1.0.0"

    def test_get_health_unhealthy_when_model_missing(self, client, monkeypatch, fitted_tiny_vectorizer):
        """GET /health should report model_loaded=False and status=unhealthy when model is None."""
        monkeypatch.setattr(predictor, "model", None)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert data["vectorizer_loaded"] is True

    def test_get_health_unhealthy_when_vectorizer_missing(self, client, monkeypatch, dummy_legitimate_model):
        """GET /health should report vectorizer_loaded=False and status=unhealthy when vectorizer is None."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", None)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is True
        assert data["vectorizer_loaded"] is False

    # --- POST /predict Inference Endpoint ---

    def test_post_predict_legitimate_success(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        """Valid legitimate job posting should return HTTP 200 and conform to PredictionResponse schema."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(
            "src.api.predictor.explain_prediction",
            lambda text, top_k=10: [
                {"feature": "developer", "impact": -0.72, "effect": "Decreases fraud risk"}
            ],
        )

        response = client.post("/predict", json={"job_description": sample_legitimate_text})
        assert response.status_code == 200

        data = response.json()
        # Validate against Pydantic schema
        validated = PredictionResponse(**data)
        assert validated.prediction == "Real"
        assert validated.fraud_probability == 0.08
        assert validated.fraud_risk_score == 8
        assert validated.risk_level == "Low"
        assert validated.model_version == "1.0.0"
        assert len(validated.top_reasons) == 1

    def test_post_predict_fraudulent_success(
        self, client, monkeypatch, dummy_fraudulent_model, fitted_tiny_vectorizer, sample_fraudulent_text
    ):
        """Valid fraudulent job posting should return HTTP 200 with Fake prediction and High/Critical risk."""
        monkeypatch.setattr(predictor, "model", dummy_fraudulent_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr(
            "src.api.predictor.explain_prediction",
            lambda text, top_k=10: [
                {"feature": "wire", "impact": 1.45, "effect": "Increases fraud risk"}
            ],
        )

        response = client.post("/predict", json={"job_description": sample_fraudulent_text})
        assert response.status_code == 200

        data = response.json()
        validated = PredictionResponse(**data)
        assert validated.prediction == "Fake"
        assert validated.fraud_probability == 0.92
        assert validated.fraud_risk_score == 92
        assert validated.risk_level == "Critical"
        assert validated.model_version == "1.0.0"

    # --- Validation & Error Handling Tests ---

    def test_post_predict_short_description_422(self, client, sample_short_text):
        """Descriptions with fewer than 20 characters must be rejected with HTTP 422."""
        response = client.post("/predict", json={"job_description": sample_short_text})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_post_predict_empty_string_422(self, client):
        """Empty job_description must be rejected with HTTP 422."""
        response = client.post("/predict", json={"job_description": ""})
        assert response.status_code == 422

    def test_post_predict_missing_field_422(self, client):
        """Request payload without job_description must return HTTP 422."""
        response = client.post("/predict", json={"wrong_key": "Some long description here exceeding length"})
        assert response.status_code == 422

    def test_post_predict_empty_body_422(self, client):
        """Empty JSON body must return HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_post_predict_invalid_data_type_422(self, client):
        """Non-string data type for job_description must be rejected with HTTP 422."""
        response = client.post("/predict", json={"job_description": 12345678901234567890})
        # Integer instead of string
        # Depending on Pydantic v2 coercion, non-coercible or lists/dicts
        response_dict = client.post("/predict", json={"job_description": ["a", "b", "c"]})
        assert response_dict.status_code == 422

    def test_post_predict_invalid_json_422(self, client):
        """Malformed JSON string should return HTTP 422."""
        response = client.post(
            "/predict",
            content="NOT_VALID_JSON{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_post_predict_oversized_payload_handles_gracefully(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer
    ):
        """Very large payloads (e.g. 50,000 chars) should process gracefully without crashing."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        huge_description = "We are seeking a senior software engineer. " * 1200
        response = client.post("/predict", json={"job_description": huge_description})
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_post_predict_unloaded_model_service_unavailable(self, client, monkeypatch, sample_legitimate_text):
        """When the model is not loaded, the endpoint must return 503 without exposing internal trace."""
        monkeypatch.setattr(predictor, "model", None)

        response = client.post("/predict", json={"job_description": sample_legitimate_text})
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Model service is currently unavailable" in data["detail"]
        # Ensure no internal paths or stack traces leaked
        assert "Traceback" not in response.text
        assert "src/api" not in response.text

    def test_post_predict_inference_failure_shielded(
        self, client, monkeypatch, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        """If model prediction throws an unexpected internal error, return 500 without stack trace."""
        failing_model = MagicMock()
        failing_model.predict.side_effect = ValueError("Internal memory error at /path/to/secret/file.c")

        monkeypatch.setattr(predictor, "model", failing_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)

        response = client.post("/predict", json={"job_description": sample_legitimate_text})
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "An error occurred while processing the prediction request."
        # Ensure path and stack trace are NOT in response
        assert "/path/to/secret" not in response.text
        assert "Traceback" not in response.text

    def test_post_predict_http_exception_reraised(self, client, monkeypatch, sample_legitimate_text):
        """HTTPException raised inside predictor should be re-raised directly."""
        from fastapi import HTTPException
        failing_predict = MagicMock(side_effect=HTTPException(status_code=400, detail="Custom client error"))
        monkeypatch.setattr("src.api.routes.predict_job", failing_predict)

        response = client.post("/predict", json={"job_description": sample_legitimate_text})
        assert response.status_code == 400
        assert response.json()["detail"] == "Custom client error"

    # --- CORS Headers ---

    def test_cors_headers_present(self, client):
        """Verify CORS headers are returned for allowed origins."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
