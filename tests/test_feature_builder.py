"""
Unit tests for feature engineering utilities in src/features/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.feature_builder import (
    create_combined_text,
    prepare_job_description_text,
)
from src.features.tfidf_vectorizer import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_text,
)
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
        assert pd.api.types.is_string_dtype(df["combined_text"])
        assert all(isinstance(val, str) for val in df["combined_text"])

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

    def test_fit_tfidf_vectorizer_and_transform_text(self, tiny_corpus):
        """fit_tfidf_vectorizer should fit on train corpus and transform_text should transform without refitting."""
        train_corpus = tiny_corpus[:3]
        test_corpus = tiny_corpus[3:]

        vec = create_tfidf_vectorizer(max_features=10, ngram_range=(1, 1))
        fitted_vec, X_train = fit_tfidf_vectorizer(vec, train_corpus)

        assert fitted_vec is vec
        assert X_train.shape[0] == len(train_corpus)
        train_vocab = set(fitted_vec.vocabulary_.keys())

        # Transform test corpus
        X_test = transform_text(fitted_vec, test_corpus)
        assert X_test.shape[0] == len(test_corpus)
        assert X_test.shape[1] == X_train.shape[1]

        # Vocabulary must remain strictly unchanged after transform
        assert set(fitted_vec.vocabulary_.keys()) == train_vocab

    def test_prepare_job_description_text_string_parity(self):
        """String input should undergo exact clean_text preprocessing matching inference."""
        raw_text = "<b>Senior Developer</b> needed! Visit https://example.com or call 123-456-7890."
        prepared = prepare_job_description_text(raw_text)
        expected = clean_text(raw_text)

        assert prepared == expected
        assert "senior developer needed visit or call" in prepared
        assert "<b>" not in prepared
        assert "http" not in prepared

    def test_prepare_job_description_text_dataframe(self, synthetic_dataframe):
        """DataFrame input should extract description column and clean all rows consistently."""
        df = synthetic_dataframe.copy()
        prepared_series = prepare_job_description_text(df, text_column="description")

        assert isinstance(prepared_series, pd.Series)
        assert len(prepared_series) == len(df)
        for i, val in enumerate(prepared_series):
            assert val == clean_text(df.loc[i, "description"])

    def test_prepare_job_description_text_handles_nan_gracefully(self):
        """NaN descriptions should become empty strings without inventing text or failing."""
        df = pd.DataFrame({"description": ["Valid description text", None, np.nan, ""]})
        res = prepare_job_description_text(df)

        assert len(res) == 4
        assert res.iloc[0] == "valid description text"
        assert res.iloc[1] == ""
        assert res.iloc[2] == ""
        assert res.iloc[3] == ""

    def test_prepare_job_description_text_missing_column_raises(self):
        """DataFrame missing required description column should raise KeyError."""
        df = pd.DataFrame({"random_col": ["Some text"]})
        with pytest.raises(KeyError, match="Required column 'description' not found"):
            prepare_job_description_text(df, text_column="description")

    def test_prepare_job_description_text_unsupported_type_raises(self):
        """Non-supported input types should raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            prepare_job_description_text(12345)


