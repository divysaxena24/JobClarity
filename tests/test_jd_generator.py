"""
Tests for synthetic JD Generation Service and API Endpoint (JobClarity Day 7 - Phase C).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.api import predictor
from src.api.schemas import JDGenerationRequest, JDGenerationResponse, PredictionResponse
from src.services.jd_generator import (
    DISCLAIMER_TEXT,
    call_gemini_generator,
    call_openai_generator,
    generate_job_description,
    generate_synthetic_template_jd,
    validate_jd_inputs,
)


class TestInputValidation:
    """Validation test suite for JD generation parameters."""

    def test_valid_inputs_pass(self):
        result = validate_jd_inputs(
            job_role="Senior DevOps Engineer",
            skills="Docker, Kubernetes, Terraform, AWS",
            experience_level="5+ years",
            location="Remote",
            company_name="CloudScale Systems",
            salary_range="$140k - $170k",
        )
        assert result["job_role"] == "Senior DevOps Engineer"
        assert result["skills"] == "Docker, Kubernetes, Terraform, AWS"
        assert result["experience_level"] == "5+ years"
        assert result["location"] == "Remote"
        assert result["company_name"] == "CloudScale Systems"
        assert result["salary_range"] == "$140k - $170k"

    def test_skills_as_list_normalized(self):
        result = validate_jd_inputs(
            job_role="Data Scientist",
            skills=["Python", "PyTorch", "SQL"],
            experience_level="Mid-Level",
            location="New York, NY",
        )
        assert result["skills"] == "Python, PyTorch, SQL"
        assert result["company_name"] == "Acme Innovations"
        assert "Competitive compensation" in result["salary_range"]

    def test_missing_job_role_raises(self):
        with pytest.raises(ValueError, match="job_role is required"):
            validate_jd_inputs(
                job_role="   ",
                skills="Python",
                experience_level="Senior",
                location="Remote",
            )

    def test_missing_skills_raises(self):
        with pytest.raises(ValueError, match="skills must be a non-empty string"):
            validate_jd_inputs(
                job_role="Software Engineer",
                skills="",
                experience_level="Senior",
                location="Remote",
            )

    def test_empty_skills_list_raises(self):
        with pytest.raises(ValueError, match="skills list cannot be empty"):
            validate_jd_inputs(
                job_role="Software Engineer",
                skills=[],
                experience_level="Senior",
                location="Remote",
            )

    def test_missing_experience_raises(self):
        with pytest.raises(ValueError, match="experience_level is required"):
            validate_jd_inputs(
                job_role="Software Engineer",
                skills="Python",
                experience_level="",
                location="Remote",
            )

    def test_missing_location_raises(self):
        with pytest.raises(ValueError, match="location is required"):
            validate_jd_inputs(
                job_role="Software Engineer",
                skills="Python",
                experience_level="Senior",
                location="",
            )

    def test_oversized_role_raises(self):
        with pytest.raises(ValueError, match="must not exceed 120 characters"):
            validate_jd_inputs(
                job_role="Developer " * 20,
                skills="Python",
                experience_level="Senior",
                location="Remote",
            )


class TestSyntheticTemplateGeneration:
    """Tests for deterministic synthetic template generator."""

    def test_template_contains_all_input_attributes(self):
        params = {
            "job_role": "Machine Learning Engineer",
            "skills": "XGBoost, SHAP, Scikit-Learn, FastAPI",
            "experience_level": "4+ years",
            "location": "Hybrid (San Francisco, CA)",
            "company_name": "Apex Intelligence",
            "salary_range": "$160,000 - $190,000",
        }
        text = generate_synthetic_template_jd(params)
        assert "Apex Intelligence" in text
        assert "Machine Learning Engineer" in text
        assert "Hybrid (San Francisco, CA)" in text
        assert "XGBoost" in text
        assert "$160,000 - $190,000" in text
        # Must be long enough to be an extensive job posting and satisfy /predict >= 20 chars
        assert len(text) > 500

    def test_generate_job_description_structure_and_disclaimer(self):
        output = generate_job_description(
            job_role="Full Stack Engineer",
            skills="React, TypeScript, Node.js",
            experience_level="Mid-Level",
            location="Remote",
        )
        assert output["is_ai_generated"] is True
        assert output["provider"] == "synthetic_template"
        assert output["disclaimer"] == DISCLAIMER_TEXT
        assert "risk estimates" in output["disclaimer"]
        assert len(output["generated_job_description"]) > 100


class TestAIProviderIntegration:
    """Tests verifying external AI provider calling and graceful fallback."""

    def test_openai_mock_success(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OpenAI generated realistic job description content for testing."}}]
        }

        with patch("httpx.Client.post", return_value=mock_response):
            monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
            output = generate_job_description(
                job_role="Backend Engineer",
                skills="Go, Kubernetes",
                experience_level="Senior",
                location="Remote",
                preferred_provider="openai",
            )
            assert output["provider"] == "openai"
            assert "OpenAI generated realistic" in output["generated_job_description"]

    def test_openai_failure_falls_back_to_synthetic(self, monkeypatch):
        with patch("httpx.Client.post", side_effect=Exception("OpenAI API unreachable")):
            monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
            output = generate_job_description(
                job_role="Backend Engineer",
                skills="Go, Kubernetes",
                experience_level="Senior",
                location="Remote",
                preferred_provider="openai",
            )
            # Falls back cleanly to template without throwing
            assert output["provider"] == "synthetic_template"
            assert len(output["generated_job_description"]) > 100
            assert "Backend Engineer" in output["generated_job_description"]

    def test_gemini_mock_success(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini generated realistic job posting for test."}]}}]
        }

        with patch("httpx.Client.post", return_value=mock_response):
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
            output = generate_job_description(
                job_role="MLOps Architect",
                skills="MLflow, Docker, Python",
                experience_level="Lead",
                location="Seattle, WA",
                preferred_provider="gemini",
            )
            assert output["provider"] == "gemini"
            assert "Gemini generated realistic" in output["generated_job_description"]

    def test_gemini_failure_falls_back_to_synthetic(self, monkeypatch):
        with patch("httpx.Client.post", side_effect=Exception("Gemini quota exceeded")):
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
            output = generate_job_description(
                job_role="MLOps Architect",
                skills="MLflow, Docker, Python",
                experience_level="Lead",
                location="Seattle, WA",
                preferred_provider="gemini",
            )
            assert output["provider"] == "synthetic_template"
            assert len(output["generated_job_description"]) > 100


class TestJDGenerationAPIEndpoint:
    """Integration tests for POST /generate-jd endpoint."""

    def test_post_generate_jd_endpoint_success(self, client):
        payload = {
            "job_role": "Security Analyst",
            "skills": "SIEM, SOC, Network Forensics, Python",
            "experience_level": "3+ years",
            "location": "Austin, TX",
            "company_name": "CyberShield Defense",
            "salary_range": "$115,000 - $135,000",
        }
        response = client.post("/generate-jd", json=payload)
        assert response.status_code == 200

        data = response.json()
        validated = JDGenerationResponse(**data)
        assert validated.is_ai_generated is True
        assert "CyberShield Defense" in validated.generated_job_description
        assert "Security Analyst" in validated.generated_job_description
        assert "risk estimates" in validated.disclaimer

    def test_post_generate_jd_validation_error(self, client):
        # Missing required skills field
        payload = {
            "job_role": "DevOps Engineer",
            "experience_level": "Senior",
            "location": "Remote",
        }
        response = client.post("/generate-jd", json=payload)
        assert response.status_code == 422

    def test_generate_and_analyze_end_to_end_flow(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer
    ):
        """End-to-end integration: Generate a synthetic JD and submit it directly to /predict."""
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        # 1. Generate synthetic JD
        gen_response = client.post(
            "/generate-jd",
            json={
                "job_role": "Full Stack Software Developer",
                "skills": "React, Python, PostgreSQL, Docker",
                "experience_level": "4 years",
                "location": "Remote",
            },
        )
        assert gen_response.status_code == 200
        gen_data = gen_response.json()
        generated_text = gen_data["generated_job_description"]
        assert len(generated_text) > 50

        # 2. Submit generated text to /predict endpoint
        predict_response = client.post(
            "/predict",
            json={"job_description": generated_text},
        )
        assert predict_response.status_code == 200
        pred_data = predict_response.json()
        validated_pred = PredictionResponse(**pred_data)
        assert validated_pred.prediction == "Real"
        assert 0.0 <= validated_pred.fraud_probability <= 1.0
