"""
Model loading utilities for JobClarity.
"""

from pathlib import Path
import joblib

# Project root = JobClarity/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Models directory
MODEL_DIR = PROJECT_ROOT / "models"


def load_vectorizer(path: Path | None = None):
    """
    Load the saved TF-IDF vectorizer.
    """
    target = Path(path) if path is not None else (MODEL_DIR / "tfidf_vectorizer.pkl")
    if not target.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found: {target}")
    return joblib.load(target)


def load_model(path: Path | None = None):
    """
    Load the tuned XGBoost model.
    """
    target = Path(path) if path is not None else (MODEL_DIR / "xgboost_model_tuned.pkl")
    if not target.exists():
        raise FileNotFoundError(f"Model artifact not found: {target}")
    return joblib.load(target)