"""
Day 4 Session 2 — Group-Aware Threshold Optimization Runner Script.

Executes leakage-safe 5-fold StratifiedGroupKFold cross-validation on near-duplicate
connected components, evaluates fine-grained threshold sweeps, determines Pareto-optimal
frontiers, selects calibrated decision thresholds on validation data, and evaluates
on the preserved held-out test partition once.

Usage:
    python scripts/run_threshold_experiments.py

Safety:
    - Never overwrites production models in models/
    - Never modifies API schemas or frontend contracts
    - Never tunes against the held-out test set
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.group_aware_threshold_optimizer import (
    run_group_aware_threshold_optimization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv"
    out_dir = PROJECT_ROOT / "artifacts" / "evaluation" / "day4" / "threshold_optimization"

    if not data_path.exists():
        logger.error("Dataset not found at %s", data_path)
        sys.exit(1)

    logger.info("=" * 75)
    logger.info("JobClarity — Group-Aware Threshold Optimization")
    logger.info("Day 4, Session 2")
    logger.info("=" * 75)
    logger.info("Dataset: %s", data_path)
    logger.info("Output:  %s", out_dir)

    report = run_group_aware_threshold_optimization(
        cleaned_data_path=data_path,
        output_dir=out_dir,
        random_state=42,
        overwrite=True,
    )

    base_sel = report["selected_policies_oof"]["baseline"]["max_f1"]
    exp5_sel = report["selected_policies_oof"]["exp5"]["max_f1"]
    test_eval = report["final_held_out_evaluations"]

    logger.info("=" * 75)
    logger.info("CALIBRATED THRESHOLD SELECTION (OOF Validation)")
    logger.info("=" * 75)
    logger.info(
        "Baseline (Default tau=0.50):    OOF F1: %.4f | Recall: %.2f%% | Prec: %.2f%%",
        report["selected_policies_oof"]["baseline"]["default_50"]["metrics"]["fraud_f1"],
        report["selected_policies_oof"]["baseline"]["default_50"]["metrics"]["fraud_recall"] * 100,
        report["selected_policies_oof"]["baseline"]["default_50"]["metrics"]["fraud_precision"] * 100,
    )
    logger.info(
        "Baseline (Calibrated tau=%.2f):  OOF F1: %.4f | Recall: %.2f%% | Prec: %.2f%%",
        base_sel["threshold"],
        base_sel["metrics"]["fraud_f1"],
        base_sel["metrics"]["fraud_recall"] * 100,
        base_sel["metrics"]["fraud_precision"] * 100,
    )
    logger.info(
        "Exp 5    (Default tau=0.50):    OOF F1: %.4f | Recall: %.2f%% | Prec: %.2f%%",
        report["selected_policies_oof"]["exp5"]["default_50"]["metrics"]["fraud_f1"],
        report["selected_policies_oof"]["exp5"]["default_50"]["metrics"]["fraud_recall"] * 100,
        report["selected_policies_oof"]["exp5"]["default_50"]["metrics"]["fraud_precision"] * 100,
    )
    logger.info(
        "Exp 5    (Calibrated tau=%.2f):  OOF F1: %.4f | Recall: %.2f%% | Prec: %.2f%%",
        exp5_sel["threshold"],
        exp5_sel["metrics"]["fraud_f1"],
        exp5_sel["metrics"]["fraud_recall"] * 100,
        exp5_sel["metrics"]["fraud_precision"] * 100,
    )

    logger.info("=" * 75)
    logger.info("FINAL HELD-OUT TEST EVALUATION (N=3,576)")
    logger.info("=" * 75)
    logger.info(
        "Baseline (Default tau=0.50):    Test F1: %.4f | Recall: %.2f%% (%d/240) | Prec: %.2f%%",
        test_eval["baseline_default_50"]["fraud_f1"],
        test_eval["baseline_default_50"]["fraud_recall"] * 100,
        test_eval["baseline_default_50"]["true_positives"],
        test_eval["baseline_default_50"]["fraud_precision"] * 100,
    )
    logger.info(
        "Baseline (Calibrated tau=%.2f):  Test F1: %.4f | Recall: %.2f%% (%d/240) | Prec: %.2f%%",
        test_eval["baseline_selected_oof"]["threshold"],
        test_eval["baseline_selected_oof"]["fraud_f1"],
        test_eval["baseline_selected_oof"]["fraud_recall"] * 100,
        test_eval["baseline_selected_oof"]["true_positives"],
        test_eval["baseline_selected_oof"]["fraud_precision"] * 100,
    )
    logger.info(
        "Exp 5    (Default tau=0.50):    Test F1: %.4f | Recall: %.2f%% (%d/240) | Prec: %.2f%%",
        test_eval["exp5_default_50"]["fraud_f1"],
        test_eval["exp5_default_50"]["fraud_recall"] * 100,
        test_eval["exp5_default_50"]["true_positives"],
        test_eval["exp5_default_50"]["fraud_precision"] * 100,
    )
    logger.info(
        "Exp 5    (Calibrated tau=%.2f):  Test F1: %.4f | Recall: %.2f%% (%d/240) | Prec: %.2f%%",
        test_eval["exp5_selected_oof"]["threshold"],
        test_eval["exp5_selected_oof"]["fraud_f1"],
        test_eval["exp5_selected_oof"]["fraud_recall"] * 100,
        test_eval["exp5_selected_oof"]["true_positives"],
        test_eval["exp5_selected_oof"]["fraud_precision"] * 100,
    )
    logger.info("=" * 75)
    logger.info("All artifacts successfully generated under %s", out_dir)
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
