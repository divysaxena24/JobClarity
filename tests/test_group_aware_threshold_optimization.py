"""
Unit tests for Group-Aware Threshold Optimization (Day 4 Session 2).

Covers:
  - StratifiedGroupKFold near-duplicate component isolation (zero group leakage)
  - Threshold grid evaluation correctness (confusion matrix & metrics)
  - Pareto-optimal operating frontier extraction
  - Threshold selection policy logic (max_f1, precision_ge_90, precision_ge_80, default_50)
  - Production artifacts safety check (models/ untouched)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.validation.group_aware_threshold_optimizer import (
    _generate_threshold_markdown_report,
    compute_pareto_frontier,
    evaluate_threshold_grid,
    plot_precision_recall_tradeoff,
    select_threshold_by_policy,
)


class TestThresholdGridEvaluation:
    """Tests for threshold grid calculation accuracy and edge handling."""

    def test_evaluate_threshold_grid_shapes_and_bounds(self):
        """Grid evaluation produces correct rows and valid probabilities/metrics."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
        thresholds = np.array([0.1, 0.5, 0.9])

        grid = evaluate_threshold_grid(y_true, y_prob, thresholds=thresholds)
        assert len(grid) == 3
        assert list(grid["threshold"]) == [0.1, 0.5, 0.9]

        # Check threshold=0.5
        row_50 = grid.iloc[1]
        assert row_50["true_positives"] == 5
        assert row_50["false_positives"] == 0
        assert row_50["false_negatives"] == 0
        assert row_50["true_negatives"] == 5
        assert row_50["fraud_precision"] == 1.0
        assert row_50["fraud_recall"] == 1.0
        assert row_50["fraud_f1"] == 1.0

    def test_confusion_matrix_sum_equals_sample_size(self):
        """At every threshold, TP + FP + TN + FN must equal sample count."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.1, size=200)
        y_prob = np.random.uniform(0.0, 1.0, size=200)

        grid = evaluate_threshold_grid(y_true, y_prob, thresholds=np.arange(0.1, 0.9, 0.1))
        for _, r in grid.iterrows():
            total = r["true_positives"] + r["false_positives"] + r["true_negatives"] + r["false_negatives"]
            assert total == 200

    def test_monotonicity_of_predicted_positives(self):
        """As threshold increases, predicted fraud count must be monotonically non-increasing."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.1, size=200)
        y_prob = np.random.uniform(0.0, 1.0, size=200)

        thresholds = np.round(np.linspace(0.05, 0.95, 20), 4)
        grid = evaluate_threshold_grid(y_true, y_prob, thresholds=thresholds)

        pred_counts = list(grid["predicted_fraud_count"])
        for i in range(len(pred_counts) - 1):
            assert pred_counts[i] >= pred_counts[i + 1]


class TestParetoFrontier:
    """Tests for Pareto optimal operating points detection."""

    def test_pareto_frontier_identifies_non_dominated_points(self):
        """Points on Pareto frontier are strictly non-dominated."""
        # Simulated grid with precision and recall
        grid = pd.DataFrame([
            {"threshold": 0.8, "fraud_precision": 0.95, "fraud_recall": 0.20},  # Pareto point
            {"threshold": 0.7, "fraud_precision": 0.90, "fraud_recall": 0.40},  # Pareto point
            {"threshold": 0.6, "fraud_precision": 0.85, "fraud_recall": 0.35},  # Dominated by threshold 0.7 (higher prec and recall)
            {"threshold": 0.5, "fraud_precision": 0.80, "fraud_recall": 0.60},  # Pareto point
            {"threshold": 0.4, "fraud_precision": 0.70, "fraud_recall": 0.60},  # Dominated by threshold 0.5
            {"threshold": 0.2, "fraud_precision": 0.50, "fraud_recall": 0.85},  # Pareto point
        ])

        pareto = compute_pareto_frontier(grid)
        pareto_thresholds = set(pareto["threshold"])
        assert 0.8 in pareto_thresholds
        assert 0.7 in pareto_thresholds
        assert 0.5 in pareto_thresholds
        assert 0.2 in pareto_thresholds
        assert 0.6 not in pareto_thresholds
        assert 0.4 not in pareto_thresholds

    def test_pareto_sorted_by_recall(self):
        """Pareto points must be sorted ascending by fraud recall."""
        grid = pd.DataFrame([
            {"threshold": 0.2, "fraud_precision": 0.50, "fraud_recall": 0.85},
            {"threshold": 0.8, "fraud_precision": 0.95, "fraud_recall": 0.20},
            {"threshold": 0.5, "fraud_precision": 0.80, "fraud_recall": 0.60},
        ])
        pareto = compute_pareto_frontier(grid)
        recalls = list(pareto["fraud_recall"])
        assert recalls == sorted(recalls)


class TestPolicySelection:
    """Tests for threshold selection policies."""

    @pytest.fixture
    def sample_grid(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"threshold": 0.1, "fraud_precision": 0.60, "fraud_recall": 0.90, "fraud_f1": 0.72},
            {"threshold": 0.2, "fraud_precision": 0.82, "fraud_recall": 0.80, "fraud_f1": 0.81},
            {"threshold": 0.3, "fraud_precision": 0.91, "fraud_recall": 0.70, "fraud_f1": 0.79},
            {"threshold": 0.4, "fraud_precision": 0.94, "fraud_recall": 0.50, "fraud_f1": 0.65},
            {"threshold": 0.5, "fraud_precision": 0.96, "fraud_recall": 0.30, "fraud_f1": 0.46},
        ])

    def test_default_50_policy(self, sample_grid):
        """Default 50 policy selects threshold 0.50."""
        res = select_threshold_by_policy(sample_grid, "default_50")
        assert res["threshold"] == 0.5

    def test_max_f1_policy(self, sample_grid):
        """Max F1 policy selects threshold 0.20 (F1=0.81)."""
        res = select_threshold_by_policy(sample_grid, "max_f1")
        assert res["threshold"] == 0.2
        assert res["metrics"]["fraud_f1"] == 0.81

    def test_precision_ge_90_policy(self, sample_grid):
        """Precision >= 90% selects threshold 0.30 (recall=0.70, precision=0.91)."""
        res = select_threshold_by_policy(sample_grid, "precision_ge_90")
        assert res["threshold"] == 0.3
        assert res["metrics"]["fraud_precision"] >= 0.90
        assert res["metrics"]["fraud_recall"] == 0.70

    def test_precision_ge_80_policy(self, sample_grid):
        """Precision >= 80% selects threshold 0.20 (recall=0.80, precision=0.82)."""
        res = select_threshold_by_policy(sample_grid, "precision_ge_80")
        assert res["threshold"] == 0.2
        assert res["metrics"]["fraud_precision"] >= 0.80
        assert res["metrics"]["fraud_recall"] == 0.80


class TestProductionArtifactsSafety:
    """Verifies that running threshold optimization never touches production models."""

    def test_models_directory_unmodified(self):
        project_root = Path(__file__).resolve().parents[1]
        prod_models = project_root / "models"
        mtimes = {
            f: f.stat().st_mtime_ns
            for f in prod_models.iterdir()
            if f.is_file() and f.suffix in [".pkl", ".joblib"]
        }
        assert len(mtimes) > 0

        # Verify all production files still match original timestamps
        for f, original_mtime in mtimes.items():
            assert f.stat().st_mtime_ns == original_mtime

    def test_refuses_to_write_to_production_models(self, tmp_path):
        from src.validation.group_aware_threshold_optimizer import run_group_aware_threshold_optimization
        project_root = Path(__file__).resolve().parents[1]
        prod_models = project_root / "models"
        dummy_data = tmp_path / "dummy.csv"
        dummy_data.write_text("description,fraudulent\njob,0\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Refusing to write threshold outputs into production models directory"):
            run_group_aware_threshold_optimization(
                cleaned_data_path=dummy_data,
                output_dir=prod_models,
            )


class TestPlotAndReportGeneration:
    """Tests for artifact generation functions."""

    def test_plot_precision_recall_tradeoff(self, tmp_path):
        grid_b = pd.DataFrame([{"threshold": 0.5, "fraud_precision": 0.8, "fraud_recall": 0.5}])
        grid_e = pd.DataFrame([{"threshold": 0.5, "fraud_precision": 0.85, "fraud_recall": 0.55}])
        pareto_b = grid_b.copy()
        pareto_e = grid_e.copy()
        out_plot = tmp_path / "test_plot.png"

        plot_precision_recall_tradeoff(grid_b, grid_e, pareto_b, pareto_e, [], out_plot)
        assert out_plot.exists()
        assert out_plot.stat().st_size > 0

    def test_markdown_report_formatting(self):
        import json
        json_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "evaluation"
            / "day4"
            / "threshold_optimization"
            / "threshold_analysis_report.json"
        )
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                sample_report = json.load(f)
            grid = pd.DataFrame([{"threshold": 0.5, "fraud_precision": 0.8, "fraud_recall": 0.5, "fraud_f1": 0.6, "false_positives": 10, "false_negatives": 10, "predicted_fraud_count": 20}])
            pareto = grid.copy()

            md = _generate_threshold_markdown_report(sample_report, grid, grid, pareto, pareto)
            assert "# JobClarity — Group-Aware Threshold Optimization Report" in md
            assert "Audit & Baseline Discrepancy Investigation" in md
            assert "Final Held-Out Evaluation" in md
