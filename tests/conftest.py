"""
Shared pytest fixtures for JobClarity test suite.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


@pytest.fixture
def sample_legitimate_text() -> str:
    """A realistic legitimate job description exceeding minimum character length."""
    return (
        "Senior Backend Python Engineer needed at TechCorp. Requirements: 4+ years "
        "of experience with FastAPI, PostgreSQL, Docker, and distributed systems. "
        "Competitive salary, full healthcare, and remote work options provided."
    )


@pytest.fixture
def sample_fraudulent_text() -> str:
    """A realistic fraudulent job description exceeding minimum character length."""
    return (
        "URGENT: Work from home and earn $4500 weekly! No interview or experience required. "
        "Wire transfer registration fee to receive immediate package processing assignment. "
        "Guaranteed cash payouts daily."
    )


@pytest.fixture
def sample_short_text() -> str:
    """A job description below the 20-character minimum threshold."""
    return "Hiring devs now"


@pytest.fixture
def tiny_corpus() -> list[str]:
    """A small text corpus for lightweight vectorizer fitting in tests."""
    return [
        "senior backend python engineer remote full time",
        "work from home earn money wire transfer fee cash",
        "administrative assistant office data entry clerk",
        "software developer react nodejs postgresql cloud",
        "urgent immediate start guaranteed income weekly payout",
    ]


@pytest.fixture
def fitted_tiny_vectorizer(tiny_corpus) -> TfidfVectorizer:
    """A lightweight fitted TfidfVectorizer for fast, isolated testing."""
    vec = TfidfVectorizer(max_features=20, stop_words="english")
    vec.fit(tiny_corpus)
    return vec


class DummyModel:
    """Lightweight deterministic model for isolating inference logic."""

    def __init__(self, pred_class: int = 0, fraud_prob: float = 0.15):
        self.pred_class = pred_class
        self.fraud_prob = fraud_prob

    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.pred_class, dtype=int)

    def predict_proba(self, X):
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        prob_fraud = self.fraud_prob
        prob_real = 1.0 - prob_fraud
        return np.tile([prob_real, prob_fraud], (n_samples, 1))


@pytest.fixture
def dummy_legitimate_model() -> DummyModel:
    """Deterministic dummy model predicting Legitimate (0)."""
    return DummyModel(pred_class=0, fraud_prob=0.08)


@pytest.fixture
def dummy_fraudulent_model() -> DummyModel:
    """Deterministic dummy model predicting Fraudulent (1)."""
    return DummyModel(pred_class=1, fraud_prob=0.92)


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    """Small synthetic dataset matching cleaned_jobs.csv schema."""
    data = {
        "title": [
            "Software Engineer",
            "Urgent Package Handler",
            "Data Analyst",
            "Cash Wire Agent",
            "Marketing Specialist",
            "Earn From Home Clerk",
        ],
        "company_profile": [
            "Established tech company building SaaS tools.",
            "International shipping provider with rapid growth.",
            "Analytics consultancy for global retailers.",
            "Financial services network.",
            "Digital marketing and SEO agency.",
            "Home business opportunities worldwide.",
        ],
        "description": [
            "Develop clean APIs in Python and maintain PostgreSQL databases.",
            "Receive packages at home and reship immediately. High commission.",
            "Analyze customer metrics using SQL and Python visualization libraries.",
            "Transfer funds between accounts. Keep 10% commission on all wires.",
            "Manage search ad campaigns and optimize customer acquisition funnels.",
            "No skills required. Process online forms for daily wire transfers.",
        ],
        "requirements": [
            "BS in Computer Science or 3+ years professional software engineering.",
            "Must have bank account and internet access. No background check.",
            "Experience with SQL, pandas, and data reporting dashboards.",
            "Ability to receive wire transfers and send money via Western Union.",
            "Google Ads certification and experience with marketing attribution.",
            "Any age eligible. Immediate start without resume review.",
        ],
        "benefits": [
            "Health, dental, 401k matching, and flexible PTO.",
            "Uncapped daily cash bonuses.",
            "Remote work allowance and education stipend.",
            "Earn up to $1000 daily from home.",
            "Generous benefits and annual performance bonus.",
            "No fees after initial registration payment.",
        ],
        "fraudulent": [0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    df["combined_text"] = df[
        ["title", "company_profile", "description", "requirements", "benefits"]
    ].agg(" ".join, axis=1)
    return df


@pytest.fixture
def production_artifacts_integrity():
    """Records timestamps of production artifacts before a test to ensure they are untouched."""
    model_paths = [
        PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl",
        PROJECT_ROOT / "models" / "xgboost_model_tuned.pkl",
        PROJECT_ROOT / "models" / "xgboost_model.pkl",
    ]
    timestamps = {}
    for p in model_paths:
        if p.exists():
            timestamps[p] = p.stat().st_mtime_ns

    yield timestamps

    for p, initial_mtime in timestamps.items():
        assert p.exists(), f"Production artifact {p} was deleted during test execution!"
        assert (
            p.stat().st_mtime_ns == initial_mtime
        ), f"CRITICAL: Production artifact {p} was modified during test execution!"
