"""
Unit tests for feature engineering utilities in src/features/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.feature_builder import create_combined_text
from src.features.tfidf_vectorizer import create_tfidf_vectorizer
from src.preprocessing.text_cleaner import clean_text


@pytest.mark.unit
class TestFeatureBuilder:
    """Test suite for feature engineering and vectorizer utilities."""

    def test_create_combined_text_standard(self, synthetic_dataframe):
        """Valid DataFrame with expected text columns should concatenate properly."""
        df = synthetic_dataframe.copy()
        df = create_combined_text(df)

        assert "combined_text" in df.columns
        assert len(df["combined_text"]) == len(df)
        assert df["combined_text"].dtype == object

        # Verify that all 5 text elements appear in the combined text for row 0
        row0_combined = df.loc[0, "combined_text"]
        assert df.loc[0, "title"] in row0_combined
        assert df.loc[0, "company_profile"] in row0_combined
        assert df.loc[0, "description"] in row0_combined
        assert df.loc[0, "requirements"] in row0_combined
        assert df.loc[0, "benefits"] in row0_combined

    def test_create_combined_text_with_missing_and_nan(self):
        """Missing or NaN values in text columns should be filled gracefully without breaking."""
        df = pd.DataFrame(
            {
                "title": ["Software Engineer", np.nan],
                "company_profile": [np.nan, "Company Profile"],
                "description": ["Build apps", "Write code"],
                "requirements": [None, "Python skills"],
                "benefits": ["Great pay", ""],
            }
        )
        result_df = create_combined_text(df)
        assert "combined_text" in result_df.columns
        assert not result_df["combined_text"].isna().any()
        assert "Software Engineer" in result_df.loc[0, "combined_text"]
        assert "Build apps" in result_df.loc[0, "combined_text"]
        assert "Company Profile" in result_df.loc[1, "combined_text"]

    def test_create_combined_text_missing_required_column(self):
        """If a mandatory text column is missing, KeyError should be raised."""
        df = pd.DataFrame(
            {
                "title": ["Job 1"],
                "description": ["Desc 1"],
                # Missing company_profile, requirements, benefits
            }
        )
        with pytest.raises(KeyError):
            create_combined_text(df)

    def test_create_combined_text_invalid_input(self):
        """Non-DataFrame inputs should raise appropriate TypeError or AttributeError."""
        with pytest.raises((TypeError, AttributeError)):
            create_combined_text("not a dataframe")

        with pytest.raises((TypeError, AttributeError)):
            create_combined_text(["list", "of", "items"])

    def test_create_combined_text_empty_dataframe(self):
        """Empty DataFrame with required columns should return an empty DataFrame with combined_text."""
        df = pd.DataFrame(
            columns=["title", "company_profile", "description", "requirements", "benefits"]
        )
        res = create_combined_text(df)
        assert "combined_text" in res.columns
        assert len(res) == 0

    # --- TF-IDF Vectorizer Factory Tests ---

    def test_create_tfidf_vectorizer_defaults(self):
        """create_tfidf_vectorizer should return an unfitted TfidfVectorizer with default hyperparams."""
        vec = create_tfidf_vectorizer()
        assert isinstance(vec, TfidfVectorizer)
        assert vec.max_features == 10000
        assert vec.ngram_range == (1, 2)
        assert vec.stop_words == "english"

    def test_create_tfidf_vectorizer_custom_params(self):
        """Custom max_features and ngram_range parameters should be respected."""
        vec = create_tfidf_vectorizer(max_features=500, ngram_range=(1, 1))
        assert vec.max_features == 500
        assert vec.ngram_range == (1, 1)

    def test_tfidf_vectorizer_fit_and_transform(self, tiny_corpus):
        """Created vectorizer should successfully fit on text and produce sparse matrix."""
        vec = create_tfidf_vectorizer(max_features=10, ngram_range=(1, 1))
        X = vec.fit_transform(tiny_corpus)
        assert X.shape[0] == len(tiny_corpus)
        assert X.shape[1] <= 10
        assert len(vec.get_feature_names_out()) <= 10

    def test_consistency_training_vs_inference_cleaning(self):
        """Verify that training feature builder and inference predictor share the exact same clean_text logic."""
        sample_input = "<b>Urgent</b>: Work from home at https://job.com, call 123-456-7890!"
        cleaned = clean_text(sample_input)

        # In both pipelines, clean_text strips HTML, URLs, phone numbers, and lowercases
        assert "<b>" not in cleaned
        assert "https" not in cleaned
        assert "123" not in cleaned
        assert cleaned == "urgent work from home at call"
