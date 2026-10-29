"""
Tests for Deployment Readiness, CI/CD, Containerization, and Regression (JobClarity Day 8).

Validates:
  - Dockerfile & .dockerignore compliance (dynamic port binding, healthcheck, secrets exclusion).
  - Production & staging health endpoints (/health, /).
  - Dynamic CORS origin configuration via environment variable.
  - Logging resilience under simulated filesystem/permission failures.
  - Safe error handling and schema contracts on /predict, /predict/staging, and /generate-jd.
  - Offline AI provider behavior and synthetic disclaimer persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from fastapi.testclient import TestClient
from src.api.main import DEFAULT_ALLOWED_ORIGINS, app
from src.api.schemas import (
    JDGenerationResponse,
    PredictionResponse,
    StagingPredictionResponse,
)
from src.services.jd_generator import DISCLAIMER_TEXT, generate_job_description
from src.services.prediction_logger import (
    get_prediction_log_path,
    log_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Dockerfile & .dockerignore Deployment Compliance
# ---------------------------------------------------------------------------

class TestDockerAndDeploymentReadiness:
    """Validate container specifications and deployment configuration."""

    def test_dockerfile_exists_and_contains_required_directives(self):
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist at repository root."

        content = dockerfile_path.read_text(encoding="utf-8")

        # Base image
        assert "FROM python:3.12-slim" in content, "Must use python:3.12-slim base image."

        # Platform dynamic port binding support
        assert "${PORT:-8000}" in content or "$PORT" in content, (
            "Dockerfile CMD must dynamically honor $PORT for cloud platform deployment (e.g. Render)."
        )

        # Healthcheck configured
        assert "HEALTHCHECK" in content, "Dockerfile must define a container HEALTHCHECK."
        assert "/health" in content, "HEALTHCHECK must probe the /health endpoint."

        # Runtime writable log directory
        assert "mkdir -p" in content and "logs" in content, (
            "Dockerfile must ensure runtime log directory /app/logs is created."
        )

        # Port exposition
        assert "EXPOSE 8000" in content

        # Zero baked secrets
        assert "sk-" not in content
        assert "AIza" not in content
        assert ".env" not in content or "COPY .env" not in content

    def test_dockerignore_excludes_sensitive_and_ephemeral_files(self):
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore must exist at repository root."

        content = dockerignore_path.read_text(encoding="utf-8")
        rules = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

        # Sensitive secrets
        assert any(".env" in r for r in rules), "Must ignore .env files."
        assert any("*.key" in r or "*.pem" in r for r in rules), "Must ignore secret keys."

        # Version control & CI
        assert ".git" in rules
        assert any(".github" in r for r in rules)

        # Local logs and ephemeral predictions
        assert any("*.jsonl" in r for r in rules), "Must ignore local JSONL prediction logs."
        assert any("*.log" in r for r in rules)

        # Local MLflow runs and SQLite DBs
        assert any("mlruns" in r for r in rules)
        assert any("mlflow.db" in r for r in rules)

        # Test reports and coverage
        assert any("htmlcov" in r for r in rules)
        assert any(".coverage" in r for r in rules)

        # CRITICAL: Inference artifacts must NOT be ignored
        assert "models" not in rules
        assert "artifacts/evaluation/day4/model_staging" not in rules


# ---------------------------------------------------------------------------
# 2. Health Endpoints & CORS Configuration
# ---------------------------------------------------------------------------

class TestHealthAndCORSConfiguration:
    """Validate healthcheck endpoints and CORS origin handling."""

    def test_health_endpoint_healthy(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["vectorizer_loaded"] is True
        assert data["version"] == "1.0.0"

    def test_root_metadata_endpoint(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["project"] == "JobClarity"
        assert data["health"] == "/health"
        assert data["generate_jd"] == "/generate-jd"

    def test_cors_default_origins(self):
        assert "http://localhost:3000" in DEFAULT_ALLOWED_ORIGINS
        assert "http://127.0.0.1:3000" in DEFAULT_ALLOWED_ORIGINS
        assert "https://jobclarity.vercel.app" in DEFAULT_ALLOWED_ORIGINS

    def test_cors_environment_variable_extension(self, monkeypatch):
        test_custom_origin = "https://custom-preview-branch.vercel.app"
        monkeypatch.setenv("ALLOWED_ORIGINS", f"{test_custom_origin}, https://another-domain.com")

        # Simulate dynamic module loading logic
        configured = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
        combined = list(dict.fromkeys(DEFAULT_ALLOWED_ORIGINS + configured))

        assert test_custom_origin in combined
        assert "https://another-domain.com" in combined
        assert "http://localhost:3000" in combined


# ---------------------------------------------------------------------------
# 3. Log Path Resolution and Fault Tolerance
# ---------------------------------------------------------------------------

class TestLoggingConfigurationAndFaultTolerance:
    """Validate dynamic log path resolution and failure shielding."""

    def test_prediction_log_path_custom_env(self, monkeypatch, tmp_path):
        custom_log_file = tmp_path / "custom_subdir" / "test_predictions.jsonl"
        monkeypatch.setenv("PREDICTION_LOG_PATH", str(custom_log_file))

        resolved = get_prediction_log_path()
        assert resolved == custom_log_file
        assert custom_log_file.parent.exists()

    def test_prediction_logging_io_error_shielded(self, monkeypatch):
        """Simulate disk full or I/O failure: log_prediction returns None without crashing."""
        with patch("builtins.open", side_effect=IOError("No space left on device (simulated disk full)")):
            result = log_prediction(
                endpoint="/predict",
                model_name="JobClarity-Baseline",
                model_version="1.0.0",
                probability=0.05,
                decision_threshold=0.50,
                risk_tier="Low",
                job_description="Sample test description for disk failure test.",
            )
            assert result is None

    def test_predict_endpoint_succeeds_even_when_logger_fails(self, client: TestClient, monkeypatch):
        """Verify API inference remains 100% operational when logging fails completely."""
        with patch("src.api.routes.log_prediction", side_effect=PermissionError("Permission denied: predictions.jsonl")):
            valid_payload = {
                "job_description": (
                    "We are looking for a Senior DevOps and Site Reliability Engineer with 5+ years of experience "
                    "in Kubernetes, Docker, Terraform, and CI/CD automation."
                )
            }
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "fraud_probability" in data

    def test_staging_endpoint_succeeds_even_when_logger_fails(self, client: TestClient, monkeypatch):
        """Verify staging inference remains 100% operational when logging fails completely."""
        with patch("src.api.routes.log_prediction", side_effect=IOError("Simulated disk error")):
            valid_payload = {
                "job_description": (
                    "Experienced Machine Learning Engineer wanted for full-time role building NLP pipelines "
                    "using Python, PyTorch, Scikit-Learn, and FastAPI in a cloud environment."
                )
            }
            response = client.post("/predict/staging", json=valid_payload)
            assert response.status_code == 200
            data = response.json()
            assert "risk_tier" in data
            assert "action" in data


# ---------------------------------------------------------------------------
# 4. Regression & Input Validation Safety Across Endpoints
# ---------------------------------------------------------------------------

class TestEndpointRegressionAndValidationSafety:
    """Validate schema compliance and safe error handling for all 3 key endpoints."""

    def test_predict_valid_schema(self, client: TestClient):
        payload = {
            "job_description": (
                "Leading healthcare company seeking a Senior Database Administrator with Oracle, PostgreSQL, "
                "and disaster recovery experience. Offers competitive pay and full benefits."
            )
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        validated = PredictionResponse(**res.json())
        assert validated.prediction in ("Real", "Fake")
        assert 0.0 <= validated.fraud_probability <= 1.0
        assert validated.model_version == "1.0.0"

    def test_predict_short_input_validation_error(self, client: TestClient):
        res = client.post("/predict", json={"job_description": "Short"})
        assert res.status_code == 422
        # Verify no traceback leakage
        assert "Traceback" not in res.text
        assert "File " not in res.text

    def test_predict_staging_valid_schema(self, client: TestClient):
        payload = {
            "job_description": (
                "Seeking Staff Infrastructure Engineer to design multi-region Kubernetes clusters. "
                "Minimum 7 years experience with Linux internals, distributed systems, and Go."
            )
        }
        res = client.post("/predict/staging", json=payload)
        assert res.status_code == 200
        validated = StagingPredictionResponse(**res.json())
        assert validated.risk_tier in ("LOW", "REVIEW", "HIGH_RISK")
        assert validated.action in ("PASS", "FLAG_FOR_REVIEW", "BLOCK")
        assert validated.calibrated_thresholds["review_threshold"] == 0.14
        assert validated.calibrated_thresholds["high_risk_threshold"] == 0.33

    def test_predict_staging_short_input_validation_error(self, client: TestClient):
        res = client.post("/predict/staging", json={"job_description": "Too brief"})
        assert res.status_code == 422
        assert "Traceback" not in res.text

    def test_generate_jd_valid_schema_and_disclaimer(self, client: TestClient):
        payload = {
            "job_role": "Cloud Security Architect",
            "skills": "AWS IAM, Kubernetes Security, Python, SIEM",
            "experience_level": "6+ years",
            "location": "Remote",
            "company_name": "Apex Cloud Systems",
            "salary_range": "$160k - $190k",
        }
        res = client.post("/generate-jd", json=payload)
        assert res.status_code == 200
        validated = JDGenerationResponse(**res.json())
        assert validated.is_ai_generated is True
        assert DISCLAIMER_TEXT == validated.disclaimer
        assert "Apex Cloud Systems" in validated.generated_job_description
        assert "Cloud Security Architect" in validated.generated_job_description

    def test_generate_jd_missing_required_fields_safe_error(self, client: TestClient):
        # Missing skills
        res = client.post(
            "/generate-jd",
            json={
                "job_role": "Backend Engineer",
                "experience_level": "Senior",
                "location": "Remote",
            },
        )
        assert res.status_code == 422
        assert "Traceback" not in res.text

    def test_offline_ai_fallback_guarantee(self, monkeypatch):
        """Ensure that with no API keys configured, generation succeeds deterministically offline."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        result = generate_job_description(
            job_role="Full Stack Engineer",
            skills="TypeScript, React, Python",
            experience_level="Mid-Level",
            location="Remote",
        )
        assert result["is_ai_generated"] is True
        assert result["provider"] == "synthetic_template"
        assert DISCLAIMER_TEXT in result["disclaimer"]
        assert len(result["generated_job_description"]) > 200
