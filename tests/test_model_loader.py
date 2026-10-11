"""
Unit tests for model and vectorizer loading in src/models/model_loader.py.
"""

from pathlib import Path
import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.models.model_loader import MODEL_DIR, load_model, load_vectorizer


@pytest.mark.unit
class TestModelLoader:
    """Test suite for model artifact loading and validation."""

    def test_load_vectorizer_custom_path_success(self, tmp_path):
        """Loading a valid saved vectorizer from a custom path should succeed."""
        vec = TfidfVectorizer(max_features=5)
        vec.fit(["sample text one", "sample text two"])

        vec_path = tmp_path / "test_vec.pkl"
        joblib.dump(vec, vec_path)

        loaded_vec = load_vectorizer(path=vec_path)
        assert isinstance(loaded_vec, TfidfVectorizer)
        assert list(loaded_vec.vocabulary_.keys()) == list(vec.vocabulary_.keys())

    def test_load_model_custom_path_success(self, tmp_path):
        """Loading a valid saved model from a custom path should succeed."""
        model = XGBClassifier(n_estimators=2, max_depth=2, random_state=42)
        X = [[0, 1], [1, 0], [1, 1], [0, 0]]
        y = [0, 1, 1, 0]
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        loaded_model = load_model(path=model_path)
        assert isinstance(loaded_model, XGBClassifier)
        assert loaded_model.n_estimators == 2

    def test_load_vectorizer_missing_file_raises_error(self, tmp_path):
        """Attempting to load a non-existent vectorizer file should raise FileNotFoundError."""
        missing_path = tmp_path / "non_existent_vectorizer.pkl"
        with pytest.raises(FileNotFoundError, match="Vectorizer artifact not found"):
            load_vectorizer(path=missing_path)

    def test_load_model_missing_file_raises_error(self, tmp_path):
        """Attempting to load a non-existent model file should raise FileNotFoundError."""
        missing_path = tmp_path / "non_existent_model.pkl"
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            load_model(path=missing_path)

    def test_load_corrupted_file_raises_error(self, tmp_path):
        """Attempting to load a corrupted or truncated file should raise a deserialization error."""
        corrupt_path = tmp_path / "corrupted_model.pkl"
        corrupt_path.write_bytes(b"CORRUPTED_BYTES_NOT_JOBLIB")

        with pytest.raises(Exception):
            load_model(path=corrupt_path)

    def test_default_production_artifacts_exist_and_load(self, production_artifacts_integrity):
        """Verify the production artifacts in models/ exist and can be loaded without modification."""
        prod_vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        prod_model_path = MODEL_DIR / "xgboost_model_tuned.pkl"

        if prod_vec_path.exists():
            vec = load_vectorizer()
            assert isinstance(vec, TfidfVectorizer)

        if prod_model_path.exists():
            mod = load_model()
            assert isinstance(mod, XGBClassifier)
