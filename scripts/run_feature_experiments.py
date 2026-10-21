"""
Day 4 Session 1 — Feature Engineering Experiment Runner Script.

Executes 5-fold Group-Aware Cross-Validation across 6 feature engineering
configurations and writes complete reports and artifacts to:
  artifacts/evaluation/day4/feature_engineering/

Usage:
    python scripts/run_feature_experiments.py

Safety:
    - Never overwrites production models in models/
    - Never overwrites prior Day 2 or Day 3 evaluation reports
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_experiment_runner import run_feature_engineering_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv"
    out_dir = PROJECT_ROOT / "artifacts" / "evaluation" / "day4" / "feature_engineering"

    if not data_path.exists():
        logger.error("Dataset not found at %s", data_path)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("JobClarity — Feature Engineering Experiments")
    logger.info("Day 4, Session 1")
    logger.info("=" * 70)
    logger.info("Dataset: %s", data_path)
    logger.info("Output:  %s", out_dir)

    report = run_feature_engineering_experiments(
        cleaned_data_path=data_path,
        output_dir=out_dir,
        random_state=42,
        save_artifacts=True,
        overwrite=True,
    )

    logger.info("=" * 70)
    logger.info("CROSS-VALIDATION EXPERIMENT SUMMARY (Ranked by PR-AUC)")
    logger.info("=" * 70)
    for exp in report["experiments"]:
        cv = exp["cv_summary"]
        logger.info(
            "%-55s | PR-AUC: %.4f ± %.4f | F1: %.4f ± %.4f | Rec: %.2f%% | Prec: %.2f%% | Runtime: %.1fs",
            exp["experiment_name"],
            cv["pr_auc_mean"],
            cv["pr_auc_std"],
            cv["fraud_f1_mean"],
            cv["fraud_f1_std"],
            cv["fraud_recall_mean"] * 100,
            cv["fraud_precision_mean"] * 100,
            cv.get("runtime_seconds", 0.0),
        )

    best = report["best_deployable_candidate"]
    base_test = report["baseline_held_out_metrics"]
    cand_test = best["held_out_test_metrics"]

    logger.info("=" * 70)
    logger.info("FINAL HELD-OUT TEST CONFIRMATION")
    logger.info("=" * 70)
    logger.info(
        "Baseline (Exp 1):       F1: %.4f | Recall: %.2f%% | Prec: %.2f%% | PR-AUC: %.4f",
        base_test["fraud_f1_score"],
        base_test["fraud_recall"] * 100,
        base_test["fraud_precision"] * 100,
        base_test["average_precision_pr_auc"],
    )
    logger.info(
        "Selected (%s): F1: %.4f | Recall: %.2f%% | Prec: %.2f%% | PR-AUC: %.4f",
        best["name"],
        cand_test["fraud_f1_score"],
        cand_test["fraud_recall"] * 100,
        cand_test["fraud_precision"] * 100,
        cand_test["average_precision_pr_auc"],
    )
    logger.info("=" * 70)
    logger.info("All artifacts saved to: %s", out_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
