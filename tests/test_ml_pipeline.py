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
