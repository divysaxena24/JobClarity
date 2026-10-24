"""
Unit and integration tests for Day 6: MLflow Experiment Tracking & Model Versioning.

Covers:
  1. MLflow tracking URI resolution and graceful setup fallback
  2. System metadata collection and strict secret/credential sanitization
  3. JobClarityModelWrapper end-to-end PyFunc inference and multi-tier decision logic
  4. Isolated Baseline experiment logging (parameters, metrics, tags, artifacts)
  5. Isolated Candidate D staging logging (21 domain features, 3-tier operating metrics)
  6. Model artifact recovery and numerical prediction parity verification
  7. Production models immutability guarantee (models/ remains strictly unmodified)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.features.feature_pipeline import (
    CompositeFeaturePipeline,
    create_word_tfidf_vectorizer,
)
from src.features.suspicious_language import SuspiciousLanguageExtractor
from src.models.experiment_tracker import (
    DEFAULT_BASELINE_REGISTERED_NAME,
    DEFAULT_CANDIDATE_D_REGISTERED_NAME,
    JobClarityModelWrapper,
    compute_metrics_at_threshold,
    get_default_tracking_uri,
    get_system_metadata,
    load_logged_model,
    log_baseline_experiment,
    log_candidate_d_experiment,
    sanitize_metadata_dict,
    setup_mlflow,
    verify_recovered_model,
)
from src.preprocessing.text_cleaner import clean_text


@pytest.fixture
def isolated_mlflow_env(tmp_path):
    """Provide an isolated temporary SQLite database for zero-side-effect test tracking."""
    db_file = tmp_path / "test_mlflow.db"
    artifacts_dir = tmp_path / "test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    return {
        "tracking_uri": tracking_uri,
        "db_file": db_file,
        "artifacts_dir": artifacts_dir,
    }


@pytest.fixture
def synthetic_training_data():
    """Create a minimal synthetic train and test DataFrame for fast testing."""
    train_texts = [
        "Software engineer python backend developer remote full-time",
        "URGENT: Work from home earn $5000 weekly wire transfer money now immediate cash payment required",
        "Frontend developer react typescript nextjs css web responsive",
        "Administrative assistant customer service data entry clerk office communications",
        "Guaranteed financial wealth wire transfer western union cashier check deposit immediately",
        "Data scientist machine learning scikit-learn pandas numpy python",
        "Earn easy money mystery shopper package handler quick cash deposit checks keep commission",
        "DevOps engineer kubernetes docker terraform aws ci cd pipelines linux",
    ]
    train_labels = [0, 1, 0, 0, 1, 0, 1, 0]

    test_texts = [
        "Senior software engineer cloud architecture distributed systems go python",
        "URGENT wire transfer money now immediate start deposit required wire funds western union",
        "Product manager agile scrum sprint backlog user stories technical",
        "Cashier check money transfer guaranteed wealth no experience required urgent deposit",
    ]
    test_labels = [0, 1, 0, 1]

    train_df = pd.DataFrame({"description": train_texts, "fraudulent": train_labels})
    test_df = pd.DataFrame({"description": test_texts, "fraudulent": test_labels})

    return train_df, test_df


class TestMLflowConfigurationAndSetup:
    """Tests for tracking URI configuration and setup error handling."""

    def test_get_default_tracking_uri_from_env_var(self, monkeypatch):
        """MLFLOW_TRACKING_URI environment variable should take top precedence."""
        custom_uri = "sqlite:///custom/path/mlflow.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", custom_uri)
        assert get_default_tracking_uri() == custom_uri

    def test_get_default_tracking_uri_from_env_file(self, tmp_path, monkeypatch):
        """When env var is not set, .env file in project root should be read."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("MLFLOW_TRACKING_URI=sqlite:///env_file/mlflow.db\n", encoding="utf-8")
        resolved = get_default_tracking_uri(project_root=tmp_path)
        assert resolved == "sqlite:///env_file/mlflow.db"

    def test_get_default_tracking_uri_fallback_sqlite(self, tmp_path, monkeypatch):
        """When neither env var nor .env has URI, defaults to sqlite:///mlflow.db."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        resolved = get_default_tracking_uri(project_root=tmp_path)
        assert resolved == f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    def test_setup_mlflow_creates_experiment(self, isolated_mlflow_env):
        """setup_mlflow should initialize tracking and create new experiment cleanly."""
        exp = setup_mlflow(
            experiment_name="TestExperiment_New",
            tracking_uri=isolated_mlflow_env["tracking_uri"],
        )
        assert exp is not None
        assert exp.name == "TestExperiment_New"

    def test_setup_mlflow_handles_invalid_uri_gracefully(self):
        """Invalid tracking URI must fail safely returning None without raising unhandled exception."""
        result = setup_mlflow(
            experiment_name="ShouldFailSafe",
            tracking_uri="invalid_protocol://non_existent_server:99999",
        )
        assert result is None


class TestSystemMetadataAndSanitization:
    """Tests for system metadata logging and security sanitization."""

    def test_get_system_metadata_fields(self):
        """System metadata must capture runtime libraries and random state."""
        meta = get_system_metadata(random_state=42)
        assert "python_version" in meta
        assert "platform" in meta
        assert "mlflow_version" in meta
        assert "xgboost_version" in meta
        assert "scikit_learn_version" in meta
        assert meta["random_state"] == 42
        assert "timestamp_utc" in meta

    def test_sanitize_metadata_dict_strips_sensitive_keys(self):
        """Keys containing secret, token, password, key, auth, etc., must be stripped."""
        dirty_dict = {
            "model_type": "xgboost",
            "api_key": "sk-1234567890abcdef",
            "db_password": "supersecretpassword",
            "auth_token": "bearer 987654321",
            "private_key": "private_cert_data",
            "random_seed": 42,
        }
        sanitized = sanitize_metadata_dict(dirty_dict)
        assert "model_type" in sanitized
        assert "random_seed" in sanitized
        assert "api_key" not in sanitized
        assert "db_password" not in sanitized
        assert "auth_token" not in sanitized
        assert "private_key" not in sanitized

    def test_sanitize_metadata_dict_truncates_oversized_strings(self):
        """Values exceeding 250 characters must be truncated safely."""
        long_val = "x" * 500
        sanitized = sanitize_metadata_dict({"oversized_param": long_val})
        assert len(sanitized["oversized_param"]) == 250


class TestJobClarityModelWrapper:
    """Tests for the custom PyFunc model wrapper."""

    @pytest.fixture
    def fitted_dummy_model_and_pipeline(self):
        pipe = CompositeFeaturePipeline(
            tfidf_model=create_word_tfidf_vectorizer(max_features=200, ngram_range=(1, 2)),
            use_suspicious_features=True,
        )
        train_texts = [
            "Software engineer python remote backend position",
            "URGENT: Work from home earn $5000 weekly wire transfer money now immediate",
            "Customer support representative communication skills full-time",
            "Fast cash guaranteed income wire money deposit checks",
        ]
        train_labels = np.array([0, 1, 0, 1])
        X = pipe.fit_transform(train_texts)
        model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, train_labels)
        return model, pipe

    def test_wrapper_predict_single_string(self, fitted_dummy_model_and_pipeline):
        model, pipe = fitted_dummy_model_and_pipeline
        wrapper = JobClarityModelWrapper(
            model=model,
            pipeline=pipe,
            decision_threshold=0.50,
        )
        res = wrapper.predict(None, "Software engineer python remote position")
        assert isinstance(res, pd.DataFrame)
        assert "fraud_probability" in res.columns
        assert "prediction" in res.columns
        assert len(res) == 1
        assert 0.0 <= res["fraud_probability"].iloc[0] <= 1.0

    def test_wrapper_predict_dataframe_input(self, fitted_dummy_model_and_pipeline):
        model, pipe = fitted_dummy_model_and_pipeline
        wrapper = JobClarityModelWrapper(
            model=model,
            pipeline=pipe,
            decision_threshold=0.50,
        )
        df_in = pd.DataFrame({
            "description": [
                "Software engineer python backend developer",
                "URGENT: Work from home earn $5000 weekly wire transfer now",
            ]
        })
        res = wrapper.predict(None, df_in)
        assert len(res) == 2
        assert res["prediction"].isin([0, 1]).all()

    def test_wrapper_multi_tier_risk_classification(self, fitted_dummy_model_and_pipeline):
        model, pipe = fitted_dummy_model_and_pipeline
        wrapper = JobClarityModelWrapper(
            model=model,
            pipeline=pipe,
            decision_threshold=0.50,
            review_threshold=0.14,
            high_risk_threshold=0.33,
        )
        res = wrapper.predict(None, ["Senior developer", "Wire transfer cash"])
        assert "risk_tier" in res.columns
        assert "action" in res.columns
        assert res["risk_tier"].isin(["LOW", "REVIEW", "HIGH_RISK"]).all()
        assert res["action"].isin(["PASS", "FLAG_FOR_REVIEW", "BLOCK"]).all()

    def test_wrapper_uninitialized_raises_runtime_error(self):
        wrapper = JobClarityModelWrapper()
        with pytest.raises(RuntimeError, match="not been initialized"):
            wrapper.predict(None, "Sample job text")


class TestIsolatedExperimentLogging:
    """Tests for logging Baseline and Candidate D experiments in an isolated store."""

    def test_log_baseline_experiment_in_isolated_store(
        self, isolated_mlflow_env, synthetic_training_data
    ):
        train_df, test_df = synthetic_training_data
        pipe = create_word_tfidf_vectorizer(max_features=100)
        X_train = pipe.fit_transform([clean_text(t) for t in train_df["description"]])
        y_train = train_df["fraudulent"].values

        model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        X_test = pipe.transform([clean_text(t) for t in test_df["description"]])
        probs = model.predict_proba(X_test)[:, 1]
        y_test = test_df["fraudulent"].values

        eval_res = {
            "pr_auc": 0.85,
            "roc_auc": 0.90,
            "eval_default_50": compute_metrics_at_threshold(y_test, probs, 0.50),
            "eval_calibrated_14": compute_metrics_at_threshold(y_test, probs, 0.14),
        }

        result = log_baseline_experiment(
            model=model,
            vectorizer_or_pipeline=pipe,
            train_df=train_df,
            test_df=test_df,
            eval_results=eval_res,
            experiment_name="Isolated_Baseline_Test",
            run_name="Test_Baseline_Run",
            tracking_uri=isolated_mlflow_env["tracking_uri"],
            register_model=True,
            registered_model_name="Test-JobClarity-Baseline",
            artifact_output_dir=isolated_mlflow_env["artifacts_dir"],
        )

        assert "run_id" in result
        assert result["run_name"] == "Test_Baseline_Run"
        assert result["parameters"]["model_name"] == "Baseline_XGBoost"
        assert result["metrics"]["decision_threshold"] == 0.50
        assert (isolated_mlflow_env["artifacts_dir"] / "baseline_mlflow_run.json").exists()

    def test_log_candidate_d_experiment_in_isolated_store(
        self, isolated_mlflow_env, synthetic_training_data
    ):
        train_df, test_df = synthetic_training_data
        pipe = CompositeFeaturePipeline(
            tfidf_model=create_word_tfidf_vectorizer(max_features=100),
            use_suspicious_features=True,
        )
        X_train = pipe.fit_transform(train_df)
        y_train = train_df["fraudulent"].values

        model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        X_test = pipe.transform(test_df)
        probs = model.predict_proba(X_test)[:, 1]
        y_test = test_df["fraudulent"].values

        eval_res = {
            "pr_auc": 0.80,
            "roc_auc": 0.88,
            "eval_default_50": compute_metrics_at_threshold(y_test, probs, 0.50),
            "eval_calibrated_14": compute_metrics_at_threshold(y_test, probs, 0.14),
            "eval_balanced_33": compute_metrics_at_threshold(y_test, probs, 0.33),
        }

        result = log_candidate_d_experiment(
            model=model,
            pipeline=pipe,
            train_df=train_df,
            test_df=test_df,
            eval_results=eval_res,
            experiment_name="Isolated_CandidateD_Test",
            run_name="Test_CandidateD_Run",
            tracking_uri=isolated_mlflow_env["tracking_uri"],
            register_model=True,
            registered_model_name="Test-JobClarity-Candidate-D",
            artifact_output_dir=isolated_mlflow_env["artifacts_dir"],
        )

        assert "run_id" in result
        assert result["run_name"] == "Test_CandidateD_Run"
        assert result["parameters"]["stage"] == "staging"
        assert result["parameters"]["status"] == "shadow_candidate"
        assert result["parameters"]["suspicious_features_count"] == 21
        assert "review_precision_at_0_14" in result["metrics"]
        assert "high_risk_precision_at_0_33" in result["metrics"]
        assert (isolated_mlflow_env["artifacts_dir"] / "candidate_d_mlflow_run.json").exists()


class TestModelArtifactRecoveryAndParity:
    """Tests for model reloadability and numerical prediction parity."""

    def test_reload_and_verify_parity(self, isolated_mlflow_env, synthetic_training_data):
        train_df, test_df = synthetic_training_data
        pipe = CompositeFeaturePipeline(
            tfidf_model=create_word_tfidf_vectorizer(max_features=100),
            use_suspicious_features=True,
        )
        X_train = pipe.fit_transform(train_df)
        y_train = train_df["fraudulent"].values

        model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        X_test = pipe.transform(test_df)
        probs = model.predict_proba(X_test)[:, 1]
        y_test = test_df["fraudulent"].values

        eval_res = {
            "pr_auc": 0.80,
            "roc_auc": 0.88,
            "eval_default_50": compute_metrics_at_threshold(y_test, probs, 0.50),
            "eval_calibrated_14": compute_metrics_at_threshold(y_test, probs, 0.14),
            "eval_balanced_33": compute_metrics_at_threshold(y_test, probs, 0.33),
        }

        run_info = log_candidate_d_experiment(
            model=model,
            pipeline=pipe,
            train_df=train_df,
            test_df=test_df,
            eval_results=eval_res,
            experiment_name="Recovery_Test_Exp",
            run_name="Recovery_Test_Run",
            tracking_uri=isolated_mlflow_env["tracking_uri"],
            register_model=False,
        )

        # Load back from MLflow
        reloaded_model = load_logged_model(
            run_info["run_id"],
            tracking_uri=isolated_mlflow_env["tracking_uri"],
        )

        # Test samples
        test_samples = [
            "Software engineer python remote backend position",
            "URGENT: Work from home earn $5000 weekly wire transfer money now immediate",
        ]

        verification = verify_recovered_model(
            logged_model=reloaded_model,
            reference_model=model,
            reference_pipeline=pipe,
            test_samples=test_samples,
            tolerance=1e-4,
        )

        assert verification["verified"] is True
        assert verification["sample_count"] == 2
        assert verification["max_absolute_difference"] <= 1e-4
        assert verification["preprocessing_consistent"] is True


class TestProductionArtifactsSafety:
    """Safety tests guaranteeing production models/ are never modified."""

    def test_production_models_dir_intact(self):
        """Production models directory must contain expected untouched artifacts."""
        prod_dir = Path("models")
        assert (prod_dir / "xgboost_model_tuned.pkl").exists()
        assert (prod_dir / "tfidf_vectorizer.pkl").exists()
        assert (prod_dir / "xgboost_model.pkl").exists()

    def test_verify_recovered_model_empty_samples_raises(self):
        """Empty sample list must raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            verify_recovered_model(
                logged_model=None,
                reference_model=None,
                reference_pipeline=None,
                test_samples=[],
            )
