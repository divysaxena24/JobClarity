"""
Unit and integration tests for Day 4 Session 1 feature engineering experiments.

Covers:
  - SuspiciousLanguageExtractor feature extraction correctness & determinism
  - Empty, None, and whitespace edge case handling
  - Surface anomaly detection (emails, phones, URLs, currency symbols)
  - Training-serving parity between batch and single text inputs
  - MetadataFeatureExtractor backward-compatible inference fallback
  - CompositeFeaturePipeline composition and sparse transformation
  - Safety constraints (production models cannot be overwritten)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.features.feature_pipeline import (
    CompositeFeaturePipeline,
    MetadataFeatureExtractor,
    create_char_tfidf_vectorizer,
    create_hybrid_tfidf_union,
    create_word_tfidf_vectorizer,
)
from src.features.suspicious_language import SuspiciousLanguageExtractor


class TestSuspiciousLanguageExtractor:
    """Test suite for domain-specific suspicious language and text anomaly extraction."""

    @pytest.fixture
    def extractor(self) -> SuspiciousLanguageExtractor:
        return SuspiciousLanguageExtractor()

    def test_feature_count_and_names(self, extractor):
        """Extractor must define exactly 21 features matching FEATURE_NAMES."""
        names = extractor.get_feature_names_out()
        assert len(names) == 21
        assert "urgency_score" in names
        assert "financial_promise_score" in names
        assert "low_barrier_score" in names
        assert "logistics_scam_score" in names
        assert "contact_evasion_score" in names
        assert "has_email" in names
        assert "email_count" in names
        assert "has_phone" in names
        assert "phone_count" in names
        assert "has_url" in names
        assert "url_count" in names
        assert "currency_symbol_count" in names
        assert "uppercase_ratio" in names

    def test_empty_and_none_handling(self, extractor):
        """Empty or None inputs must return an all-zero 21-element vector without error."""
        res_empty = extractor.transform("")
        assert res_empty.shape == (1, 21)
        assert np.all(res_empty == 0.0)

        res_none = extractor.transform([None])
        assert res_none.shape == (1, 21)
        assert np.all(res_none == 0.0)

        res_whitespace = extractor.transform("    \n\t  ")
        assert res_whitespace.shape == (1, 21)
        assert res_whitespace[0, 16] == float(len("    \n\t  "))  # char_count is index 16
        assert res_whitespace[0, 18] == 0.0  # word_count is index 18

    def test_lexical_scam_detection(self, extractor):
        """Accurately detects urgency, financial promises, and low barrier cues."""
        text = "URGENT hiring immediately! Earn $5000 weekly wire transfer with no experience needed."
        feats = extractor.transform(text)[0]
        feature_dict = dict(zip(extractor.FEATURE_NAMES, feats))

        assert feature_dict["urgency_score"] >= 1.0  # "urgent", "immediately"
        assert feature_dict["financial_promise_score"] >= 1.0  # "wire transfer", "earn ... weekly"
        assert feature_dict["low_barrier_score"] >= 1.0  # "no experience"
        assert feature_dict["currency_symbol_count"] >= 1.0  # "$"
        assert feature_dict["exclamation_count"] >= 1.0  # "!"

    def test_surface_anomaly_detection(self, extractor):
        """Detects raw surface emails, phone numbers, and messaging evasion handles."""
        text = "Contact recruiter on WhatsApp +1-234-567-8900 or email scammer@fakejob.org directly."
        feats = extractor.transform(text)[0]
        feature_dict = dict(zip(extractor.FEATURE_NAMES, feats))

        assert feature_dict["has_email"] == 1.0
        assert feature_dict["has_phone"] == 1.0
        assert feature_dict["contact_evasion_score"] >= 1.0  # "whatsapp"

    def test_logistics_reshipping_scam_detection(self, extractor):
        """Detects package forwarding and mystery shopping scam keywords."""
        text = "Work as a mystery shopper or package reship agent for commission."
        feats = extractor.transform(text)[0]
        feature_dict = dict(zip(extractor.FEATURE_NAMES, feats))

        assert feature_dict["logistics_scam_score"] >= 1.0

    def test_uppercase_ratio_calculation(self, extractor):
        """Uppercase ratio measures capitalization intensity accurately."""
        all_upper = "HELLO WORLD"
        feats_upper = extractor.transform(all_upper)[0]
        upper_dict = dict(zip(extractor.FEATURE_NAMES, feats_upper))
        assert upper_dict["uppercase_ratio"] == 1.0

        all_lower = "hello world"
        feats_lower = extractor.transform(all_lower)[0]
        lower_dict = dict(zip(extractor.FEATURE_NAMES, feats_lower))
        assert lower_dict["uppercase_ratio"] == 0.0

    def test_single_string_vs_dataframe_parity(self, extractor):
        """Single string input produces identical feature values to DataFrame batch input."""
        sample_text = "Urgent: wire transfer needed immediately for package reship."
        feats_single = extractor.transform(sample_text)
        feats_batch = extractor.transform(pd.DataFrame({"description": [sample_text]}))

        np.testing.assert_allclose(feats_single, feats_batch, rtol=1e-5)

    def test_determinism(self, extractor):
        """Repeated transformations on the same input yield identical outputs."""
        text = "Senior Python engineer with Kubernetes experience."
        res1 = extractor.transform(text)
        res2 = extractor.transform(text)
        np.testing.assert_array_equal(res1, res2)


class TestMetadataFeatureExtractor:
    """Test suite for optional metadata extraction with backward compatibility."""

    @pytest.fixture
    def meta_extractor(self) -> MetadataFeatureExtractor:
        return MetadataFeatureExtractor()

    def test_metadata_names(self, meta_extractor):
        """Feature names include binary flags and employment type one-hot encodings."""
        names = meta_extractor.get_feature_names_out()
        assert len(names) == 9  # 3 flags + 5 emp types + 1 missing
        assert "has_company_logo" in names
        assert "emp_type_full-time" in names
        assert "emp_type_missing" in names

    def test_backward_compatibility_on_raw_string(self, meta_extractor):
        """When called with a string (current API inference contract), outputs neutral fallback vector."""
        feats = meta_extractor.transform("A job description without metadata.")
        assert feats.shape == (1, 9)
        # All binary flags default to -1.0 (unknown)
        assert feats[0, 0] == -1.0  # logo
        assert feats[0, 1] == -1.0  # questions
        assert feats[0, 2] == -1.0  # telecommuting
        # Employment missing indicator is 1.0
        assert feats[0, -1] == 1.0

    def test_dataframe_extraction(self, meta_extractor):
        """Extracts metadata accurately when provided in a DataFrame."""
        df = pd.DataFrame({
            "description": ["Job text 1", "Job text 2"],
            "has_company_logo": [1, 0],
            "has_questions": [1, 0],
            "telecommuting": [0, 1],
            "employment_type": ["Full-time", "Contract"],
        })
        feats = meta_extractor.transform(df)
        assert feats.shape == (2, 9)
        assert feats[0, 0] == 1.0  # logo present
        assert feats[1, 0] == 0.0  # logo absent
        assert feats[0, 3] == 1.0  # full-time
        assert feats[1, 5] == 1.0  # contract


class TestCompositeFeaturePipeline:
    """Test suite for CompositeFeaturePipeline composition and training-serving parity."""

    def test_hybrid_feature_union_output_shape(self):
        """Word + Char FeatureUnion produces concatenated sparse matrix."""
        texts = [
            "Senior python engineer distributed systems microservices cloud aws.",
            "Urgent wire transfer earn 5000 weekly no experience needed.",
        ]
        union = create_hybrid_tfidf_union(word_max_features=50, char_max_features=30)
        X = union.fit_transform(texts)
        assert X.shape[0] == 2
        assert X.shape[1] <= 80

    def test_composite_pipeline_training_serving_parity(self):
        """Single description at inference produces exact same feature vector as in batch."""
        texts = [
            "Senior python developer microservices cloud postgresql.",
            "Earn unlimited commission wire transfer weekly.",
        ]
        pipe = CompositeFeaturePipeline(
            tfidf_model=create_word_tfidf_vectorizer(max_features=100),
            use_suspicious_features=True,
            use_metadata_features=False,
        )
        X_batch = pipe.fit_transform(texts)

        # Single string inference
        X_single = pipe.transform(texts[0])

        assert X_single.shape[1] == X_batch.shape[1]
        np.testing.assert_allclose(X_batch[0].toarray(), X_single.toarray(), rtol=1e-5)

    def test_production_artifacts_unmodified(self, tmp_path):
        """Feature pipeline operations never touch models/ artifacts."""
        project_root = Path(__file__).resolve().parents[1]
        prod_models = project_root / "models"
        mtimes = {
            f: f.stat().st_mtime_ns
            for f in prod_models.iterdir()
            if f.is_file() and f.suffix in [".pkl", ".joblib"]
        }

        # Run pipeline operations
        pipe = CompositeFeaturePipeline(
            tfidf_model=create_word_tfidf_vectorizer(max_features=50),
            use_suspicious_features=True,
        )
        pipe.fit_transform(["Test job description text for verification."])

        # Verify no mtime changed
        for f, original_mtime in mtimes.items():
            assert f.stat().st_mtime_ns == original_mtime, f"CRITICAL: {f} was modified!"
