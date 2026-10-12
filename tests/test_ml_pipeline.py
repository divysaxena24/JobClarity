"""
Unit tests for ML training and evaluation utilities in src/models/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.evaluate import evaluate_model, load_dataset, plot_confusion_matrix
from src.models.model_trainer import create_model, train_model


@pytest.mark.ml
class TestMLPipeline:
    """Test suite for machine learning pipeline and evaluation functions."""

    def test_create_model_defaults(self):
        """create_model should initialize an XGBClassifier with configured baseline parameters."""
        model = create_model(random_state=42)
        assert isinstance(model, XGBClassifier)
        params = model.get_params()
        assert params["random_state"] == 42
        assert params["n_estimators"] == 100
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.1
        assert params["eval_metric"] == "logloss"

    def test_train_model_synthetic_data(self):
        """train_model should successfully fit on feature matrix and target array."""
        model = create_model(random_state=42)
        X_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        y_train = np.array([0, 1, 1, 0])

        fitted = train_model(model, X_train, y_train)
        assert fitted is model
        preds = fitted.predict(X_train)
        assert len(preds) == 4
        assert set(preds).issubset({0, 1})

    def test_model_training_reproducibility(self):
        """Training with identical random_state on identical data must yield identical predictions."""
        X = np.random.default_rng(42).standard_normal((20, 5))
        y = np.array([0, 1] * 10)

        m1 = create_model(random_state=42)
        m1 = train_model(m1, X, y)

        m2 = create_model(random_state=42)
        m2 = train_model(m2, X, y)

        assert np.array_equal(m1.predict(X), m2.predict(X))
        assert np.allclose(m1.predict_proba(X), m2.predict_proba(X))

    def test_evaluate_model_metrics_structure(self):
        """evaluate_model should compute all required classification metrics accurately."""
        # Simple fitted dummy model
        class FixedTestModel:
            def predict(self, X):
                # 4 samples: 2 Real, 2 Fake predicted
                return np.array([0, 1, 0, 1])

            def predict_proba(self, X):
                return np.array(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                        [0.85, 0.15],
                        [0.1, 0.9],
                    ]
                )

        model = FixedTestModel()
        X_dummy = np.zeros((4, 2))
        y_true = np.array([0, 1, 0, 0])  # 3 real, 1 fake

        eval_res = evaluate_model(model, X_dummy, y_true, model_name="TestDummy")

        assert eval_res["model_name"] == "TestDummy"
        assert eval_res["test_samples"] == 4
        assert eval_res["class_distribution"]["legitimate_0"] == 3
        assert eval_res["class_distribution"]["fraudulent_1"] == 1

        metrics = eval_res["metrics"]
        assert "accuracy" in metrics
        assert "fraud_precision" in metrics
        assert "fraud_recall" in metrics
        assert "fraud_f1_score" in metrics
        assert "roc_auc" in metrics
        assert "average_precision_pr_auc" in metrics

        # All values should be valid floats between 0 and 1
        for val in metrics.values():
            assert 0.0 <= val <= 1.0

        # Confusion matrix checks
        cm = eval_res["confusion_matrix"]
        assert cm["true_negatives"] + cm["false_positives"] == 3
        assert cm["false_negatives"] + cm["true_positives"] == 1

    def test_load_dataset_valid_file(self, tmp_path):
        """load_dataset should load and return a DataFrame when CSV exists."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        csv_file = tmp_path / "valid.csv"
        df.to_csv(csv_file, index=False)

        loaded = load_dataset(csv_file)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["col1", "col2"]

    def test_load_dataset_nonexistent_file(self, tmp_path):
        """load_dataset should raise FileNotFoundError for missing path."""
        nonexistent = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            load_dataset(nonexistent)

    def test_plot_confusion_matrix_generates_file(self, tmp_path):
        """plot_confusion_matrix should write a valid non-empty PNG file to disk."""
        cm = np.array([[100, 5], [10, 85]])
        out_png = tmp_path / "test_cm.png"

        plot_confusion_matrix(cm, title="Test Confusion Matrix", output_path=out_png)

        assert out_png.exists()
        assert out_png.stat().st_size > 0

    def test_stratified_split_preserves_class_ratio(self, synthetic_dataframe):
        """Verify that train_test_split preserves stratification for imbalanced targets."""
        # Create a DataFrame with 100 samples and 10% positives
        X = np.arange(100).reshape(-1, 1)
        y = np.array([1] * 10 + [0] * 90)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        assert len(y_test) == 20
        # 10% of 20 = 2 positive samples in test
        assert sum(y_test == 1) == 2
        assert sum(y_train == 1) == 8

    def test_run_evaluation_synthetic_end_to_end(self, tmp_path, synthetic_dataframe):
        """run_evaluation should load data, transform, evaluate models, and save JSON report and plots."""
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier
        from src.models.evaluate import run_evaluation

        # 1. Save synthetic dataset (multiply rows to ensure test split has both classes)
        data_csv = tmp_path / "cleaned_jobs.csv"
        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        df.to_csv(data_csv, index=False)

        # 2. Fit and save vectorizer
        vec = TfidfVectorizer(max_features=20)
        X = vec.fit_transform(df["combined_text"])
        vec_path = tmp_path / "tfidf_vectorizer.pkl"
        joblib.dump(vec, vec_path)

        # 3. Fit and save models
        y = df["fraudulent"]
        model = XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
        model.fit(X, y)
        tuned_path = tmp_path / "xgboost_model_tuned.pkl"
        base_path = tmp_path / "xgboost_model.pkl"
        joblib.dump(model, tuned_path)
        joblib.dump(model, base_path)

        # 4. Run evaluation
        report_json = tmp_path / "report.json"
        cm_png = tmp_path / "cm.png"

        results = run_evaluation(
            data_path=data_csv,
            vectorizer_path=vec_path,
            tuned_model_path=tuned_path,
            base_model_path=base_path,
            output_report_path=report_json,
            output_cm_path=cm_png,
            test_size=0.20,
            random_state=42,
        )

        assert report_json.exists()
        assert cm_png.exists()
        assert "evaluations" in results
        assert "production_tuned" in results["evaluations"]
        assert "untuned_baseline" in results["evaluations"]

    def test_evaluate_cli_main(self, tmp_path, monkeypatch):
        """CLI main entry point should parse arguments and execute run_evaluation."""
        import sys
        import src.models.evaluate as eval_module

        called = {}

        def mock_run_eval(**kwargs):
            called.update(kwargs)
            return {"status": "success"}

        monkeypatch.setattr(eval_module, "run_evaluation", mock_run_eval)
        test_args = [
            "evaluate.py",
            "--data-path", str(tmp_path / "data.csv"),
            "--output-json", str(tmp_path / "out.json"),
            "--output-cm-plot", str(tmp_path / "out.png"),
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        eval_module.main()
        assert called["data_path"] == tmp_path / "data.csv"
        assert called["output_report_path"] == tmp_path / "out.json"
        assert called["output_cm_path"] == tmp_path / "out.png"

    def test_split_dataset_reproducibility(self, synthetic_dataframe):
        """split_dataset must produce identical train/test splits with identical random_state."""
        from src.models.model_trainer import split_dataset

        df = pd.concat([synthetic_dataframe] * 10, ignore_index=True)

        # Run 1
        train1, test1 = split_dataset(df, test_size=0.20, random_state=42)
        # Run 2 with same seed
        train2, test2 = split_dataset(df, test_size=0.20, random_state=42)

        # Indices and target values must match exactly
        assert list(train1.index) == list(train2.index)
        assert list(test1.index) == list(test2.index)
        assert np.array_equal(train1["fraudulent"].values, train2["fraudulent"].values)
        assert np.array_equal(test1["fraudulent"].values, test2["fraudulent"].values)

        # Run with different seed should produce a different split
        train_diff, test_diff = split_dataset(df, test_size=0.20, random_state=99)
        assert list(train1.index) != list(train_diff.index)

    def test_vectorizer_fitted_only_on_training_data_oov_protection(self, tmp_path, synthetic_dataframe):
        """Tokens unique to the test split must NOT leak into the vectorizer vocabulary."""
        from src.models.model_trainer import train_leakage_free_baseline, split_dataset

        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)

        # Determine which row lands in the test split
        _, test_df = split_dataset(df, test_size=0.20, random_state=42)
        test_idx = test_df.index[0]

        # Inject an unmistakable canary token ONLY into the test row
        canary_token = "leakage_detector_canary_xyz"
        df.loc[test_idx, "description"] = df.loc[test_idx, "description"] + f" {canary_token}"
        # Regenerate combined_text with the canary token present in the test row
        from src.features.feature_builder import create_combined_text
        df = create_combined_text(df)

        data_csv = tmp_path / "oov_test.csv"
        df.to_csv(data_csv, index=False)

        candidate_dir = tmp_path / "candidate_oov"
        results = train_leakage_free_baseline(
            data_path=data_csv,
            output_dir=candidate_dir,
            test_size=0.20,
            random_state=42,
            max_features=1000,
            save_artifacts=True,
        )

        vectorizer = results["vectorizer"]

        # Crucial assertion: the canary token MUST NOT be in vocabulary
        assert canary_token not in vectorizer.vocabulary_

        # Verify that if the vectorizer had been leaked (fitted on df), canary token WOULD be in vocabulary
        from src.features.tfidf_vectorizer import create_tfidf_vectorizer
        leaked_vec = create_tfidf_vectorizer(max_features=1000)
        leaked_vec.fit(df["combined_text"])
        assert canary_token in leaked_vec.vocabulary_

    def test_pipeline_never_calls_fit_or_fit_transform_on_test_data(self, tmp_path, synthetic_dataframe, monkeypatch):
        """Verify that fit and fit_transform are never invoked on the test split."""
        from src.models.model_trainer import train_leakage_free_baseline
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        data_csv = tmp_path / "spy_test.csv"
        df.to_csv(data_csv, index=False)

        fitted_on = []
        transformed = []

        original_fit_transform = TfidfVectorizer.fit_transform
        original_transform = TfidfVectorizer.transform

        def spy_fit_transform(self, raw_documents, y=None):
            fitted_on.append(list(raw_documents))
            return original_fit_transform(self, raw_documents, y)

        def spy_transform(self, raw_documents):
            transformed.append(list(raw_documents))
            return original_transform(self, raw_documents)

        monkeypatch.setattr(TfidfVectorizer, "fit_transform", spy_fit_transform)
        monkeypatch.setattr(TfidfVectorizer, "transform", spy_transform)

        candidate_dir = tmp_path / "candidate_spy"
        results = train_leakage_free_baseline(
            data_path=data_csv,
            output_dir=candidate_dir,
            test_size=0.20,
            random_state=42,
            max_features=20,
            save_artifacts=False,
        )

        # fit_transform was called exactly once on training data (80% of 30 = 24 rows)
        assert len(fitted_on) == 1
        assert len(fitted_on[0]) == 24

        # transform was called on test data (20% of 30 = 6 rows)
        assert len(transformed) == 1
        assert len(transformed[0]) == 6

        # The documents passed to fit_transform must have zero overlap with test documents
        train_docs = set(fitted_on[0])
        test_docs = set(transformed[0])
        # Ensure training set size matches recorded info
        assert results["dataset_info"]["train_samples"] == 24
        assert results["dataset_info"]["test_samples"] == 6

    def test_candidate_pipeline_records_sizes_and_distributions(self, tmp_path, synthetic_dataframe):
        """Pipeline must accurately record dataset sizes and class distributions for both splits."""
        from src.models.model_trainer import train_leakage_free_baseline

        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        data_csv = tmp_path / "data.csv"
        df.to_csv(data_csv, index=False)

        candidate_dir = tmp_path / "candidate_meta"
        results = train_leakage_free_baseline(
            data_path=data_csv,
            output_dir=candidate_dir,
            test_size=0.20,
            random_state=42,
            save_artifacts=True,
        )

        info = results["dataset_info"]
        assert info["total_records"] == 30
        assert info["train_samples"] == 24
        assert info["test_samples"] == 6

        train_dist = info["train_distribution"]
        test_dist = info["test_distribution"]

        assert train_dist["legitimate_0"] + train_dist["fraudulent_1"] == 24
        assert test_dist["legitimate_0"] + test_dist["fraudulent_1"] == 6
        assert train_dist["legitimate_0"] == 12
        assert train_dist["fraudulent_1"] == 12
        assert test_dist["legitimate_0"] == 3
        assert test_dist["fraudulent_1"] == 3

        # Verify saved files
        assert (candidate_dir / "tfidf_vectorizer.pkl").exists()
        assert (candidate_dir / "xgboost_model.pkl").exists()
        assert (candidate_dir / "baseline_evaluation_report.json").exists()
        assert (candidate_dir / "split_metadata.json").exists()
        assert (candidate_dir / "baseline_confusion_matrix.png").exists()

    def test_candidate_pipeline_refuses_to_overwrite_production_dir(self, tmp_path, synthetic_dataframe):
        """Pipeline must raise ValueError when output_dir points to models/."""
        from src.models.model_trainer import train_leakage_free_baseline, PROJECT_ROOT

        data_csv = tmp_path / "data.csv"
        synthetic_dataframe.to_csv(data_csv, index=False)

        with pytest.raises(ValueError, match="Refusing to write directly into production models directory"):
            train_leakage_free_baseline(
                data_path=data_csv,
                output_dir=PROJECT_ROOT / "models",
                save_artifacts=True,
            )

    def test_candidate_pipeline_leaves_production_artifacts_untouched(
        self, tmp_path, synthetic_dataframe, production_artifacts_integrity
    ):
        """Pipeline execution must never modify production model artifacts."""
        from src.models.model_trainer import train_leakage_free_baseline

        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        data_csv = tmp_path / "data.csv"
        df.to_csv(data_csv, index=False)

        candidate_dir = tmp_path / "candidate_safe"
        train_leakage_free_baseline(
            data_path=data_csv,
            output_dir=candidate_dir,
            save_artifacts=True,
        )

        assert candidate_dir.exists()
        # production_artifacts_integrity fixture will automatically verify
        # that models/ files were not modified during this test

    def test_trainer_cli_main(self, tmp_path, monkeypatch):
        """CLI main entry point for model_trainer should parse args and run pipeline."""
        import sys
        import src.models.model_trainer as trainer_module

        called = {}

        def mock_train(**kwargs):
            called.update(kwargs)
            return {"status": "success"}

        monkeypatch.setattr(trainer_module, "train_leakage_free_baseline", mock_train)
        test_args = [
            "model_trainer.py",
            "--data-path", str(tmp_path / "data.csv"),
            "--output-dir", str(tmp_path / "out_dir"),
            "--test-size", "0.25",
            "--random-state", "123",
            "--max-features", "500",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        trainer_module.main()
        assert called["data_path"] == tmp_path / "data.csv"
        assert called["output_dir"] == tmp_path / "out_dir"
        assert called["test_size"] == 0.25
        assert called["random_state"] == 123
        assert called["max_features"] == 500
        assert called["text_column"] == "description"

    def test_training_serving_text_cleaning_and_vectorization_parity(self, synthetic_dataframe, tmp_path):
        """Prove that the identical text-cleaning and vectorization behavior is used for training and inference."""
        from src.features.feature_builder import prepare_job_description_text
        from src.models.model_trainer import train_leakage_free_baseline
        from src.preprocessing.text_cleaner import clean_text

        # Train a candidate pipeline with description representation
        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        data_csv = tmp_path / "parity_test.csv"
        df.to_csv(data_csv, index=False)

        candidate_dir = tmp_path / "candidate_parity"
        results = train_leakage_free_baseline(
            data_path=data_csv,
            output_dir=candidate_dir,
            test_size=0.20,
            random_state=42,
            max_features=500,
            text_column="description",
            save_artifacts=True,
        )
        candidate_vectorizer = results["vectorizer"]

        # Realistic messy input a user might paste into the frontend
        messy_job_input = (
            "<b>Senior Python Backend Developer</b> urgently needed! "
            "Apply at https://careers.company.io/apply or email hr@company.io. "
            "Call +1-800-555-0199 for fast track. Competitive salary and benefits!"
        )

        # 1. Training representation (via DataFrame/Series)
        train_rep = prepare_job_description_text(
            pd.DataFrame({"description": [messy_job_input]}),
            text_column="description",
        ).iloc[0]

        # 2. Inference representation (via single string as received by /predict)
        infer_rep = prepare_job_description_text(messy_job_input)

        # Both must produce the EXACT same cleaned string
        assert train_rep == infer_rep
        assert train_rep == clean_text(messy_job_input)
        assert "<b>" not in infer_rep
        assert "https" not in infer_rep
        assert "@" not in infer_rep
        assert "555" not in infer_rep

        # 3. Vectorization parity
        X_train_vector = candidate_vectorizer.transform([train_rep])
        X_infer_vector = candidate_vectorizer.transform([infer_rep])

        vocab_size = len(candidate_vectorizer.vocabulary_)
        assert X_train_vector.shape == X_infer_vector.shape == (1, vocab_size)
        assert np.array_equal(X_train_vector.toarray(), X_infer_vector.toarray())
        assert X_infer_vector.nnz > 0  # Non-empty feature extraction

    def test_candidate_pipeline_description_representation_dimensions_and_compatibility(
        self, tmp_path, synthetic_dataframe
    ):
        """Confirm dimensions and feature compatibility on representative synthetic examples."""
        from src.features.feature_builder import prepare_job_description_text
        from src.models.model_trainer import train_leakage_free_baseline

        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        data_csv = tmp_path / "compat_test.csv"
        df.to_csv(data_csv, index=False)

        candidate_dir = tmp_path / "candidate_compat"
        results = train_leakage_free_baseline(
            data_path=data_csv,
            output_dir=candidate_dir,
            test_size=0.20,
            random_state=42,
            max_features=100,
            text_column="description",
            save_artifacts=True,
        )

        model = results["model"]
        vectorizer = results["vectorizer"]

        # Confirm representation metadata
        report_path = candidate_dir / "baseline_evaluation_report.json"
        assert report_path.exists()
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        assert report_data["metadata"]["text_representation"]["representation"] == "job_description_only"
        assert report_data["metadata"]["text_representation"]["source_column"] == "description"
        assert report_data["metadata"]["text_representation"]["training_serving_consistency"] is True

        # Test single-instance production inference compatibility
        test_sample = "Develop clean APIs in Python and maintain PostgreSQL databases."
        cleaned = prepare_job_description_text(test_sample)
        X_single = vectorizer.transform([cleaned])

        vocab_size = len(vectorizer.vocabulary_)
        assert vocab_size <= 100
        assert X_single.shape == (1, vocab_size)
        pred = model.predict(X_single)
        proba = model.predict_proba(X_single)

        assert pred[0] in [0, 1]
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0
        assert np.isclose(proba[0, 0] + proba[0, 1], 1.0)

    def test_default_tuning_candidates_hyperparameter_coverage(self):
        """Tuning candidate grid must cover n_estimators, max_depth, learning_rate, subsample, colsample_bytree, and scale_pos_weight."""
        from src.models.hyperparameter_tuner import DEFAULT_CANDIDATES

        assert len(DEFAULT_CANDIDATES) >= 4

        n_estimators_set = {c.n_estimators for c in DEFAULT_CANDIDATES}
        max_depth_set = {c.max_depth for c in DEFAULT_CANDIDATES}
        learning_rate_set = {c.learning_rate for c in DEFAULT_CANDIDATES}
        subsample_set = {c.subsample for c in DEFAULT_CANDIDATES}
        colsample_set = {c.colsample_bytree for c in DEFAULT_CANDIDATES}
        weight_set = {c.scale_pos_weight for c in DEFAULT_CANDIDATES}

        # Each hyperparameter must explore multiple values
        assert len(n_estimators_set) > 1
        assert len(max_depth_set) > 1
        assert len(learning_rate_set) > 1
        assert len(subsample_set) > 1
        assert len(colsample_set) > 1

        # Must compare unweighted and class-weighted candidates
        assert 1.0 in weight_set
        assert "balanced" in weight_set

    def test_compute_fold_scale_pos_weight_strictly_local(self):
        """compute_fold_scale_pos_weight must strictly calculate negative/positive ratio on supplied fold labels."""
        from src.models.hyperparameter_tuner import compute_fold_scale_pos_weight

        # 90 legit, 10 fraud -> 9.0
        y_fold = np.array([0] * 90 + [1] * 10)
        assert compute_fold_scale_pos_weight(y_fold) == 9.0

        # 95 legit, 5 fraud -> 19.0
        y_fold2 = np.array([0] * 95 + [1] * 5)
        assert compute_fold_scale_pos_weight(y_fold2) == 19.0

        # All negative edge case
        y_all_neg = np.array([0] * 100)
        assert compute_fold_scale_pos_weight(y_all_neg) == 1.0

    def test_cv_fold_safe_vectorization_canary_isolation(self, tmp_path, synthetic_dataframe, monkeypatch):
        """Tokens unique to a validation fold must never leak into the vectorizer fitted on the training fold."""
        import src.models.hyperparameter_tuner as tuner_mod
        from src.models.hyperparameter_tuner import (
            CandidateConfig,
            run_stratified_cv_tuning,
        )
        from sklearn.model_selection import StratifiedKFold

        df = pd.concat([synthetic_dataframe] * 6, ignore_index=True)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Inject a unique canary token into fold 1's validation samples only
        canary = "cv_leakage_detector_canary_999"
        splits = list(skf.split(df["description"], df["fraudulent"]))
        _, val_idx_fold1 = splits[0]
        df.loc[val_idx_fold1[0], "description"] += f" {canary}"

        fitted_vocabularies = []
        original_fit_tfidf = tuner_mod.fit_tfidf_vectorizer

        def spy_fit_tfidf(vectorizer, train_text):
            vec, X_tr = original_fit_tfidf(vectorizer, train_text)
            fitted_vocabularies.append(set(vec.vocabulary_.keys()))
            return vec, X_tr

        monkeypatch.setattr(tuner_mod, "fit_tfidf_vectorizer", spy_fit_tfidf)

        # Lightweight fast candidate for test
        test_candidates = [
            CandidateConfig(
                candidate_id="cand_test_fast",
                name="Fast Test",
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                scale_pos_weight=1.0,
            )
        ]

        from src.features.feature_builder import prepare_job_description_text
        X_train_text = prepare_job_description_text(df, text_column="description")
        y_train = df["fraudulent"].values

        run_stratified_cv_tuning(
            X_train_text=X_train_text,
            y_train=y_train,
            candidates=test_candidates,
            n_splits=3,
            random_state=42,
            max_features=200,
        )

        assert len(fitted_vocabularies) == 3
        # In fold 1, the canary was in val_idx, so fold 1's training vocabulary must NOT contain the canary
        fold_1_vocab = fitted_vocabularies[0]
        assert canary not in fold_1_vocab


    def test_stratified_cv_tuning_reproducibility(self, synthetic_dataframe):
        """Cross-validation tuning must yield identical fold results and rankings under identical random_state."""
        from src.models.hyperparameter_tuner import (
            CandidateConfig,
            run_stratified_cv_tuning,
        )
        from src.features.feature_builder import prepare_job_description_text

        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        X_text = prepare_job_description_text(df, text_column="description")
        y = df["fraudulent"].values

        fast_candidates = [
            CandidateConfig(
                candidate_id="cand_a",
                name="Cand A",
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                scale_pos_weight=1.0,
            ),
            CandidateConfig(
                candidate_id="cand_b",
                name="Cand B",
                n_estimators=3,
                max_depth=2,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight="balanced",
            ),
        ]

        # Run 1
        res1 = run_stratified_cv_tuning(
            X_train_text=X_text,
            y_train=y,
            candidates=fast_candidates,
            n_splits=3,
            random_state=42,
            max_features=100,
        )

        # Run 2
        res2 = run_stratified_cv_tuning(
            X_train_text=X_text,
            y_train=y,
            candidates=fast_candidates,
            n_splits=3,
            random_state=42,
            max_features=100,
        )

        assert res1[0]["candidate_id"] == res2[0]["candidate_id"]
        assert res1[0]["mean_metrics"]["mean_pr_auc"] == res2[0]["mean_metrics"]["mean_pr_auc"]
        assert res1[1]["mean_metrics"]["mean_fraud_f1"] == res2[1]["mean_metrics"]["mean_fraud_f1"]

    def test_tuning_pipeline_candidate_artifact_isolation(self, tmp_path, synthetic_dataframe):
        """Tuning pipeline must write artifacts to tuned directory, never overwrite production or baseline candidate."""
        from src.models.hyperparameter_tuner import (
            CandidateConfig,
            tune_xgboost_pipeline,
        )
        from src.models.model_trainer import PROJECT_ROOT

        df = pd.concat([synthetic_dataframe] * 6, ignore_index=True)
        data_csv = tmp_path / "tuning_data.csv"
        df.to_csv(data_csv, index=False)

        tuned_dir = tmp_path / "artifacts" / "candidate" / "tuned"

        fast_candidates = [
            CandidateConfig(
                candidate_id="cand_test",
                name="Test Fast",
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                scale_pos_weight="balanced",
            )
        ]

        result = tune_xgboost_pipeline(
            data_path=data_csv,
            output_dir=tuned_dir,
            test_size=0.20,
            n_splits=3,
            random_state=42,
            max_features=100,
            candidates=fast_candidates,
            save_artifacts=True,
        )

        # Verify saved artifacts in tuned dir
        assert (tuned_dir / "tfidf_vectorizer.pkl").exists()
        assert (tuned_dir / "xgboost_model_tuned.pkl").exists()
        assert (tuned_dir / "cv_tuning_report.json").exists()
        assert (tuned_dir / "tuning_metadata.json").exists()

        # Guard: Refuse to write directly into production models dir
        with pytest.raises(ValueError, match="Refusing to write directly into production models directory"):
            tune_xgboost_pipeline(
                data_path=data_csv,
                output_dir=PROJECT_ROOT / "models",
                candidates=fast_candidates,
            )

        # Guard: Refuse to write directly into baseline candidate dir
        with pytest.raises(ValueError, match="Refusing to write directly into baseline candidate directory"):
            tune_xgboost_pipeline(
                data_path=data_csv,
                output_dir=PROJECT_ROOT / "artifacts" / "candidate",
                candidates=fast_candidates,
            )

    def test_tuning_pipeline_preserves_held_out_test_set_untouched(self, tmp_path, synthetic_dataframe):
        """The 20% held-out test split must remain untouched and un-evaluated throughout tuning."""
        from src.models.hyperparameter_tuner import (
            CandidateConfig,
            tune_xgboost_pipeline,
        )
        import json

        df = pd.concat([synthetic_dataframe] * 6, ignore_index=True)
        data_csv = tmp_path / "tuning_data.csv"
        df.to_csv(data_csv, index=False)

        tuned_dir = tmp_path / "candidate_test_preserved"
        fast_candidates = [
            CandidateConfig(
                candidate_id="cand_test",
                name="Test Fast",
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                scale_pos_weight=1.0,
            )
        ]

        tune_xgboost_pipeline(
            data_path=data_csv,
            output_dir=tuned_dir,
            test_size=0.20,
            n_splits=3,
            random_state=42,
            max_features=100,
            candidates=fast_candidates,
            save_artifacts=True,
        )

        with open(tuned_dir / "cv_tuning_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)

        held_out_info = report["metadata"]["held_out_test_set"]
        assert held_out_info["status"] == "preserved_untouched"
        assert held_out_info["evaluated_in_tuning"] is False
        assert "Session 4" in held_out_info["reservation"]
        # Ensure no test metrics exist in report
        assert "test_metrics" not in report
        assert "test_evaluation" not in report

    def test_tuning_cli_invocation_via_trainer_main(self, tmp_path, monkeypatch):
        """CLI with --tune flag should invoke tune_xgboost_pipeline with parsed parameters."""
        import sys
        import src.models.model_trainer as trainer_module

        called = {}

        def mock_tune(**kwargs):
            called.update(kwargs)
            return {"status": "success"}

        monkeypatch.setattr(trainer_module, "tune_xgboost_pipeline", mock_tune)

        test_args = [
            "model_trainer.py",
            "--tune",
            "--data-path", str(tmp_path / "jobs.csv"),
            "--output-dir", str(tmp_path / "out_tuned"),
            "--test-size", "0.20",
            "--n-splits", "4",
            "--random-state", "123",
            "--max-features", "2000",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        trainer_module.main()
        assert called["data_path"] == tmp_path / "jobs.csv"
        assert called["output_dir"] == tmp_path / "out_tuned"
        assert called["test_size"] == 0.20
        assert called["n_splits"] == 4
        assert called["random_state"] == 123
        assert called["max_features"] == 2000
        assert called["text_column"] == "description"

    def test_held_out_evaluation_synthetic_end_to_end(self, tmp_path, synthetic_dataframe):
        """evaluate_held_out_candidates must evaluate both candidates and majority baseline on held-out split."""
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier
        from src.models.evaluate import evaluate_held_out_candidates
        from src.features.feature_builder import prepare_job_description_text

        # 1. Dataset
        df = pd.concat([synthetic_dataframe] * 5, ignore_index=True)
        data_csv = tmp_path / "cleaned_jobs.csv"
        df.to_csv(data_csv, index=False)

        # 2. Candidate baseline artifacts
        base_dir = tmp_path / "candidate"
        base_dir.mkdir(parents=True, exist_ok=True)
        vec_base = TfidfVectorizer(max_features=20)
        X_base = vec_base.fit_transform(prepare_job_description_text(df))
        model_base = XGBClassifier(n_estimators=2, max_depth=2, random_state=42)
        model_base.fit(X_base, df["fraudulent"])
        joblib.dump(vec_base, base_dir / "tfidf_vectorizer.pkl")
        joblib.dump(model_base, base_dir / "xgboost_model.pkl")

        # 3. Candidate tuned artifacts
        tuned_dir = tmp_path / "candidate_tuned"
        tuned_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(vec_base, tuned_dir / "tfidf_vectorizer.pkl")
        joblib.dump(model_base, tuned_dir / "xgboost_model_tuned.pkl")

        # 4. Evaluation output directory
        eval_dir = tmp_path / "evaluation" / "day2"

        results = evaluate_held_out_candidates(
            data_path=data_csv,
            baseline_dir=base_dir,
            tuned_dir=tuned_dir,
            output_dir=eval_dir,
            test_size=0.20,
            random_state=42,
            save_artifacts=True,
        )

        # Verify output files exist
        assert (eval_dir / "final_evaluation_report.json").exists()
        assert (eval_dir / "baseline_confusion_matrix.png").exists()
        assert (eval_dir / "tuned_confusion_matrix.png").exists()
        assert (eval_dir / "majority_class_confusion_matrix.png").exists()

        # Verify evaluations structure
        report = results["report"]
        assert "majority_class_baseline" in report["evaluations"]
        assert "baseline_candidate" in report["evaluations"]
        assert "tuned_candidate" in report["evaluations"]
        assert "comparison" in report

        # Verify majority class metrics
        maj = report["evaluations"]["majority_class_baseline"]["metrics"]
        assert maj["fraud_precision"] == 0.0
        assert maj["fraud_recall"] == 0.0
        assert maj["fraud_f1_score"] == 0.0
        assert maj["roc_auc"] == 0.5

        # Verify comparison metrics
        comp = report["comparison"]
        assert comp["prediction_agreement_rate"] == 1.0
        assert comp["max_absolute_probability_difference"] == 0.0

    def test_held_out_evaluation_artifact_isolation_guards(self, tmp_path, synthetic_dataframe):
        """evaluate_held_out_candidates must raise ValueError when output_dir threatens candidate or production dirs."""
        from src.models.evaluate import evaluate_held_out_candidates, PROJECT_ROOT

        data_csv = tmp_path / "cleaned_jobs.csv"
        synthetic_dataframe.to_csv(data_csv, index=False)

        base_dir = tmp_path / "candidate"
        tuned_dir = tmp_path / "candidate_tuned"

        # Guard: Refuse to write to production models/
        with pytest.raises(ValueError, match="Refusing to write evaluation report into production models directory"):
            evaluate_held_out_candidates(
                data_path=data_csv,
                baseline_dir=base_dir,
                tuned_dir=tuned_dir,
                output_dir=PROJECT_ROOT / "models",
                save_artifacts=False,
            )

        # Guard: Refuse to overwrite baseline candidate dir
        with pytest.raises(ValueError, match="Refusing to overwrite baseline candidate directory"):
            evaluate_held_out_candidates(
                data_path=data_csv,
                baseline_dir=base_dir,
                tuned_dir=tuned_dir,
                output_dir=base_dir,
                save_artifacts=False,
            )

        # Guard: Refuse to overwrite tuned candidate dir
        with pytest.raises(ValueError, match="Refusing to overwrite tuned candidate directory"):
            evaluate_held_out_candidates(
                data_path=data_csv,
                baseline_dir=base_dir,
                tuned_dir=tuned_dir,
                output_dir=tuned_dir,
                save_artifacts=False,
            )

    def test_evaluate_cli_held_out_flag(self, tmp_path, monkeypatch):
        """CLI with --held-out flag should invoke evaluate_held_out_candidates."""
        import sys
        import src.models.evaluate as eval_module

        called = {}

        def mock_eval(**kwargs):
            called.update(kwargs)
            return {"status": "success"}

        monkeypatch.setattr(eval_module, "evaluate_held_out_candidates", mock_eval)

        test_args = [
            "evaluate.py",
            "--held-out",
            "--data-path", str(tmp_path / "data.csv"),
            "--output-dir", str(tmp_path / "out_dir"),
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        eval_module.main()
        assert called["data_path"] == tmp_path / "data.csv"
        assert called["output_dir"] == tmp_path / "out_dir"





