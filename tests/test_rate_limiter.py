"""
Unit and integration tests for the rate limiter (src/api/rate_limiter.py).
"""

from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.rate_limiter import (
    SlidingWindowRateLimiter,
    get_client_ip,
    is_valid_ip,
    rate_limit_dependency,
    rate_limiter,
)
from src.api.main import app
from src.api import predictor


class DummyRequest:
    """Mock Request object for isolated IP extraction tests."""

    def __init__(self, headers: dict | None = None, client_host: str | None = None):
        self.headers = headers or {}
        self.client = MagicMock(host=client_host) if client_host is not None else None
        self.url = MagicMock(path="/predict")


class TestIPValidationAndExtraction:
    """Test IP address validation and trusted reverse-proxy extraction logic."""

    def test_is_valid_ip(self):
        assert is_valid_ip("192.168.1.1") is True
        assert is_valid_ip("10.0.0.5") is True
        assert is_valid_ip("127.0.0.1") is True
        assert is_valid_ip("::1") is True
        assert is_valid_ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334") is True

        # Invalid cases
        assert is_valid_ip("not-an-ip") is False
        assert is_valid_ip("<script>alert(1)</script>") is False
        assert is_valid_ip("256.256.256.256") is False
        assert is_valid_ip("1.2.3.4.5") is False
        assert is_valid_ip("") is False
        assert is_valid_ip(None) is False  # type: ignore

    def test_get_client_ip_from_single_x_forwarded_for(self):
        req = DummyRequest(headers={"x-forwarded-for": "198.51.100.42"})
        assert get_client_ip(req, trust_proxy_headers=True) == "198.51.100.42"

    def test_get_client_ip_from_comma_separated_x_forwarded_for(self):
        # Render sets client IP first, followed by proxies
        req = DummyRequest(headers={"x-forwarded-for": "203.0.113.195, 70.41.3.18, 150.172.238.178"})
        assert get_client_ip(req, trust_proxy_headers=True) == "203.0.113.195"

    def test_get_client_ip_from_x_real_ip(self):
        req = DummyRequest(headers={"x-real-ip": "198.51.100.88"})
        assert get_client_ip(req, trust_proxy_headers=True) == "198.51.100.88"

    def test_get_client_ip_direct_client_fallback(self):
        req = DummyRequest(headers={}, client_host="192.168.5.10")
        assert get_client_ip(req, trust_proxy_headers=True) == "192.168.5.10"

    def test_get_client_ip_ignores_spoofed_or_invalid_forwarded_for(self):
        req = DummyRequest(headers={"x-forwarded-for": "malicious-host"}, client_host="10.20.30.40")
        # Should reject "malicious-host" and fall back to client.host
        assert get_client_ip(req, trust_proxy_headers=True) == "10.20.30.40"

    def test_get_client_ip_untrusted_proxy_mode_ignores_headers(self):
        req = DummyRequest(headers={"x-forwarded-for": "1.2.3.4"}, client_host="10.20.30.40")
        assert get_client_ip(req, trust_proxy_headers=False) == "10.20.30.40"


class TestSlidingWindowRateLimiter:
    """Test core algorithmic rate limiter behavior in isolation."""

    def test_rate_limiter_allows_up_to_limit(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=3, window_seconds=60)
        t0 = 1000.0

        # Requests 1, 2, 3 must be allowed
        ok1, rem1, _ = limiter.check_rate_limit("client_a", current_time=t0)
        assert ok1 is True
        assert rem1 == 2

        ok2, rem2, _ = limiter.check_rate_limit("client_a", current_time=t0 + 1)
        assert ok2 is True
        assert rem2 == 1

        ok3, rem3, _ = limiter.check_rate_limit("client_a", current_time=t0 + 2)
        assert ok3 is True
        assert rem3 == 0

        # Request 4 must be blocked
        ok4, rem4, retry4 = limiter.check_rate_limit("client_a", current_time=t0 + 3)
        assert ok4 is False
        assert rem4 == 0
        assert retry4 == 57  # 1000 + 60 - 1003 = 57 seconds

    def test_rate_limiter_independent_clients(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_seconds=60)
        t0 = 1000.0

        # Exhaust client_a
        limiter.check_rate_limit("client_a", current_time=t0)
        limiter.check_rate_limit("client_a", current_time=t0)
        ok_a, _, _ = limiter.check_rate_limit("client_a", current_time=t0)
        assert ok_a is False

        # client_b should still be allowed
        ok_b, _, _ = limiter.check_rate_limit("client_b", current_time=t0)
        assert ok_b is True

    def test_sliding_window_expiration(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=1, window_seconds=10)
        t0 = 1000.0

        ok1, _, _ = limiter.check_rate_limit("client_a", current_time=t0)
        assert ok1 is True

        # Within window: blocked
        ok2, _, _ = limiter.check_rate_limit("client_a", current_time=t0 + 5)
        assert ok2 is False

        # After window (t0 + 11): allowed again
        ok3, _, _ = limiter.check_rate_limit("client_a", current_time=t0 + 11)
        assert ok3 is True


@pytest.mark.integration
class TestRateLimiterFastAPIIntegration:
    """Test HTTP 429 and rate limit headers through FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def reset_limiter(self):
        rate_limiter.reset()
        yield
        rate_limiter.reset()

    def test_predict_returns_rate_limit_headers_on_success(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        response = client.post(
            "/predict",
            json={"job_description": sample_legitimate_text},
            headers={"x-forwarded-for": "100.64.0.1"},
        )
        assert response.status_code == 200
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers

    def test_predict_exceeds_rate_limit_returns_429(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])

        # Set a low limit for quick testing
        monkeypatch.setattr(rate_limiter, "_requests_per_minute", 2)

        test_ip = "192.0.2.1"
        headers = {"x-forwarded-for": test_ip}

        # Request 1: 200
        r1 = client.post("/predict", json={"job_description": sample_legitimate_text}, headers=headers)
        assert r1.status_code == 200
        assert r1.headers["x-ratelimit-remaining"] == "1"

        # Request 2: 200
        r2 = client.post("/predict", json={"job_description": sample_legitimate_text}, headers=headers)
        assert r2.status_code == 200
        assert r2.headers["x-ratelimit-remaining"] == "0"

        # Request 3: 429
        r3 = client.post("/predict", json={"job_description": sample_legitimate_text}, headers=headers)
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
        assert "x-ratelimit-remaining" in r3.headers
        assert r3.headers["x-ratelimit-remaining"] == "0"
        data = r3.json()
        assert "Rate limit exceeded" in data["detail"]

    def test_health_check_unaffected_by_rate_limiting(
        self, client, monkeypatch, dummy_legitimate_model, fitted_tiny_vectorizer, sample_legitimate_text
    ):
        monkeypatch.setattr(predictor, "model", dummy_legitimate_model)
        monkeypatch.setattr(predictor, "vectorizer", fitted_tiny_vectorizer)
        monkeypatch.setattr("src.api.predictor.explain_prediction", lambda text, top_k=10: [])
        monkeypatch.setattr(rate_limiter, "_requests_per_minute", 1)

        test_ip = "192.0.2.99"
        headers = {"x-forwarded-for": test_ip}

        # Exhaust limit on /predict
        client.post("/predict", json={"job_description": sample_legitimate_text}, headers=headers)
        blocked = client.post("/predict", json={"job_description": sample_legitimate_text}, headers=headers)
        assert blocked.status_code == 429

        # Health endpoint must still return 200 OK!
        health_res = client.get("/health", headers=headers)
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "healthy"

        # Home endpoint must also return 200 OK!
        home_res = client.get("/", headers=headers)
        assert home_res.status_code == 200
