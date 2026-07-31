"""
Model loading utilities for JobClarity.
"""

from pathlib import Path
import joblib

# Project root = JobClarity/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Models directory
MODEL_DIR = PROJECT_ROOT / "models"


def load_vectorizer():
    """
    Load the saved TF-IDF vectorizer.
    """
    return joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )


def load_model():
    """
    Load the tuned XGBoost model.
    """
    return joblib.load(
        MODEL_DIR / "xgboost_model_tuned.pkl"
    )