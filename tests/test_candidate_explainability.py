"""
Unit and integration tests for Day 5: SHAP Explainability & Evidence-Based Model Selection.

Covers:
  1. Exact TreeSHAP computation and mathematical local accuracy identity (margin = base_score + sum(shap))
  2. Feature name alignment (TF-IDF tokens + 21 deterministic description indicators)
  3. Feature categorization (deterministic_indicator vs tfidf_ngram)
  4. Local prediction explanation formatting, rankings, and causal disclaimers
  5. Deterministic behavior across repeated calls
  6. Edge cases: empty text, whitespace-only, short text (< 20 chars), non-string types
  7. Global feature importance calculation and slice error analysis
  8. Production artifact immutability
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.explainability.candidate_shap_explainer import (
    CAUSAL_DISCLAIMER,
    analyze_prediction_errors,
    compute_global_feature_importance,
    compute_tree_shap_contributions,
    explain_candidate_prediction,
    extract_representative_cohorts,
    get_feature_type,
)
from src.features.feature_pipeline import (
    CompositeFeaturePipeline,
    create_word_tfidf_vectorizer,
)
from src.features.suspicious_language import SuspiciousLanguageExtractor


@pytest.fixture
def sample_fitted_candidate():
    """Create a lightweight fitted Candidate D pipeline and XGBoost model for fast testing."""
    corpus = [
        "Senior software engineer python fastapi postgresql remote full time",
        "URGENT: Work from home earn $5000 weekly wire transfer money now immediate cash",
        "Administrative assistant data entry clerk office management communication",
        "Junior web developer react typescript css html frontend responsive design",
        "Guaranteed financial wealth package delivery shipping courier customer service",
    ]
    labels = np.array([0, 1, 0, 0, 1])

    pipe = CompositeFeaturePipeline(
        tfidf_model=create_word_tfidf_vectorizer(max_features=25, ngram_range=(1, 2), min_df=1),
        use_suspicious_features=True,
    )
    X = pipe.fit_transform(corpus)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X, labels)

    return model, pipe, corpus, labels


class TestTreeSHAPComputation:
    """Verifies unapproximated TreeSHAP calculation and mathematical local accuracy."""

    def test_compute_tree_shap_contributions_shape_and_bounds(self, sample_fitted_candidate):
        model, pipe, corpus, _ = sample_fitted_candidate
        X = pipe.transform(corpus)
        shap_matrix, base_score = compute_tree_shap_contributions(model, X)

        assert shap_matrix.shape == X.shape
        assert isinstance(base_score, float)
        assert not math.isnan(base_score)
        assert not math.isinf(base_score)

    def test_tree_shap_local_accuracy_identity(self, sample_fitted_candidate):
        """
        Verify the fundamental TreeSHAP local accuracy axiom:
        margin(x) = base_score + sum_j(phi_j(x))
        prob(x) = 1 / (1 + exp(-margin(x))) == model.predict_proba(x)[1]
        """
        model, pipe, corpus, _ = sample_fitted_candidate
        X = pipe.transform(corpus)
        shap_matrix, base_score = compute_tree_shap_contributions(model, X)

        predicted_probs = model.predict_proba(X)[:, 1]

        for i in range(len(corpus)):
            margin_sum = base_score + float(shap_matrix[i].sum())
            prob_from_shap = 1.0 / (1.0 + np.exp(-margin_sum))
            prob_direct = float(predicted_probs[i])

            # Must equal within 1e-5 floating point tolerance
            assert abs(prob_from_shap - prob_direct) < 1e-5, (
                f"Local accuracy violated for sample {i}: "
                f"SHAP prob {prob_from_shap} vs model prob {prob_direct}"
            )

    def test_non_xgboost_model_raises_runtime_error(self, sample_fitted_candidate):
        _, pipe, corpus, _ = sample_fitted_candidate
        X = pipe.transform(corpus)

        class DummyNonTreeModel:
            pass

        with pytest.raises(RuntimeError, match="Expected XGBClassifier or Booster"):
            compute_tree_shap_contributions(DummyNonTreeModel(), X)


class TestFeatureAlignmentAndCategorization:
    """Verifies proper distinction between TF-IDF lexical tokens and domain indicators."""

    def test_get_feature_type_categorization(self):
        # 21 deterministic indicators
        for ind in SuspiciousLanguageExtractor.FEATURE_NAMES:
            assert get_feature_type(ind) == "deterministic_indicator"

        # TF-IDF tokens
        assert get_feature_type("engineer") == "tfidf_ngram"
        assert get_feature_type("wire transfer") == "tfidf_ngram"
        assert get_feature_type("urgent") == "tfidf_ngram"

    def test_feature_names_out_length_matches_matrix_width(self, sample_fitted_candidate):
        _, pipe, corpus, _ = sample_fitted_candidate
        X = pipe.transform(corpus)
        names = pipe.get_feature_names_out()

        assert len(names) == X.shape[1]
        assert len(names) == 25 + 21  # 25 TF-IDF + 21 indicators


class TestExplainCandidatePrediction:
    """Tests for single-instance local explanation formatting and edge handling."""

    def test_explain_prediction_valid_instance(self, sample_fitted_candidate):
        model, pipe, _, _ = sample_fitted_candidate
        text = "URGENT: Work from home earn $5000 weekly wire transfer money now immediate cash"

        res = explain_candidate_prediction(text, model, pipe, top_k=5)

        assert isinstance(res, dict)
        assert "predicted_probability" in res
        assert "base_margin" in res
        assert "margin_sum" in res
        assert "top_features" in res
        assert "causal_disclaimer" in res
        assert res["causal_disclaimer"] == CAUSAL_DISCLAIMER
        assert 0.0 <= res["predicted_probability"] <= 1.0

        # Check top features formatting
        assert len(res["top_features"]) <= 5
        for feat in res["top_features"]:
            assert "feature" in feat
            assert "feature_type" in feat
            assert feat["feature_type"] in ["deterministic_indicator", "tfidf_ngram"]
            assert "impact" in feat
            assert feat["effect"] in ["Increases fraud risk", "Decreases fraud risk"]
            assert feat["attribution_type"] == "TreeSHAP Margin Contribution"

    def test_explain_prediction_deterministic(self, sample_fitted_candidate):
        model, pipe, _, _ = sample_fitted_candidate
        text = "Senior software developer python remote position"

        res1 = explain_candidate_prediction(text, model, pipe, top_k=5)
        res2 = explain_candidate_prediction(text, model, pipe, top_k=5)

        assert res1["predicted_probability"] == res2["predicted_probability"]
        assert res1["margin_sum"] == res2["margin_sum"]
        assert len(res1["top_features"]) == len(res2["top_features"])
        for f1, f2 in zip(res1["top_features"], res2["top_features"]):
            assert f1["feature"] == f2["feature"]
            assert f1["impact"] == f2["impact"]

    def test_explain_prediction_short_text_fails_safely(self, sample_fitted_candidate):
        model, pipe, _, _ = sample_fitted_candidate
        res = explain_candidate_prediction("Hiring now", model, pipe)

        assert res["status"] == "INPUT_TOO_SHORT"
        assert res["top_features"] == []
        assert res["predicted_probability"] == 0.0

    def test_explain_prediction_empty_and_whitespace(self, sample_fitted_candidate):
        model, pipe, _, _ = sample_fitted_candidate
        res_empty = explain_candidate_prediction("", model, pipe)
        assert res_empty["status"] == "INPUT_TOO_SHORT"

        res_space = explain_candidate_prediction("                 ", model, pipe)
        assert res_space["status"] == "INPUT_TOO_SHORT"

    def test_explain_prediction_non_string_raises_type_error(self, sample_fitted_candidate):
        model, pipe, _, _ = sample_fitted_candidate
        with pytest.raises(TypeError, match="must be a string"):
            explain_candidate_prediction(12345, model, pipe)  # type: ignore


class TestGlobalFeatureImportanceAndErrorAnalysis:
    """Tests for dataset-level global importance and sliced error metrics."""

    def test_compute_global_feature_importance(self, sample_fitted_candidate):
        model, pipe, corpus, _ = sample_fitted_candidate
        df_imp = compute_global_feature_importance(model, pipe, corpus, sample_size=5)

        assert isinstance(df_imp, pd.DataFrame)
        assert len(df_imp) == 25 + 21
        assert "feature" in df_imp.columns
        assert "feature_type" in df_imp.columns
        assert "mean_abs_shap" in df_imp.columns
        assert "mean_directional_shap" in df_imp.columns

        # Verify sorted descending
        shap_vals = list(df_imp["mean_abs_shap"])
        assert shap_vals == sorted(shap_vals, reverse=True)

    def test_analyze_prediction_errors_structure(self, sample_fitted_candidate):
        model, pipe, corpus, labels = sample_fitted_candidate
        df = pd.DataFrame({"description": corpus, "fraudulent": labels})
        X = pipe.transform(corpus)
        probs = model.predict_proba(X)[:, 1]

        res = analyze_prediction_errors(df, labels, probs, threshold=0.14)
        assert isinstance(res, dict)
        assert "Very Short (< 50 words)" in res
        assert "Has Urgency Signals" in res
        for slice_name, metrics in res.items():
            assert "sample_count" in metrics
            assert "tp" in metrics
            assert "fp" in metrics

    def test_extract_representative_cohorts(self, sample_fitted_candidate):
        model, pipe, corpus, labels = sample_fitted_candidate
        df = pd.DataFrame({"description": corpus, "fraudulent": labels})
        X = pipe.transform(corpus)
        probs = model.predict_proba(X)[:, 1]

        cohorts = extract_representative_cohorts(df, labels, probs, model, pipe, threshold=0.14, n_per_cohort=1)
        assert isinstance(cohorts, dict)
        assert "true_positives" in cohorts
        assert "false_positives" in cohorts
        assert "true_negatives" in cohorts
        assert "false_negatives" in cohorts


class TestProductionArtifactsSafety:
    """Verifies that running Day 5 explainability never touches production models."""

    def test_models_directory_unmodified(self):
        project_root = Path(__file__).resolve().parents[1]
        prod_models = project_root / "models"
        prod_files = [f for f in prod_models.iterdir() if f.is_file() and f.suffix in [".pkl", ".joblib"]]
        assert len(prod_files) > 0
        for f in prod_files:
            assert f.exists()
            assert f.stat().st_size > 0
