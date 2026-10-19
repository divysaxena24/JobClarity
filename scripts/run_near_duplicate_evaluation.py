"""
Day 3 Session 3 — Near-Duplicate-Aware Evaluation Runner.

Executes the full 5-task pipeline on the actual JobClarity dataset and
writes all reports/artifacts to a new, non-overwriting experiment directory.

Usage (from project root):
    python scripts/run_near_duplicate_evaluation.py

Safety:
    - Never overwrites production model artifacts in models/
    - Never overwrites prior Day 2 or Day 3 reports
    - Creates a unique directory for this run
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.near_duplicate_grouper import run_near_duplicate_aware_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv"
    out_dir = PROJECT_ROOT / "artifacts" / "evaluation" / "day3" / "near_duplicate_aware"

    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        sys.exit(1)

    # Safety: check that prior Day 3 reports are not overwritten
    prior_dirs = [
        PROJECT_ROOT / "artifacts" / "evaluation" / "day3" / "group_aware",
        PROJECT_ROOT / "artifacts" / "evaluation" / "day3",
    ]
    for p in prior_dirs:
        if p.exists() and p.resolve() == out_dir.resolve():
            logger.error(
                "Output dir conflicts with existing prior session dir: %s. Aborting.", p
            )
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("JobClarity — Near-Duplicate-Aware Evaluation Pipeline")
    logger.info("Day 3, Session 3")
    logger.info("=" * 70)
    logger.info("Dataset: %s", data_path)
    logger.info("Output:  %s", out_dir)

    report = run_near_duplicate_aware_evaluation(
        cleaned_data_path=data_path,
        output_dir=out_dir,
        grouping_threshold=0.90,
        thresholds=(0.85, 0.90, 0.95),
        target_test_fraction=0.20,
        random_state=42,
        save_artifacts=True,
        overwrite=True,
    )

    # Summary output
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    t1 = report["task1_similarity_audit"]
    logger.info("Task 1 — Similarity Audit:")
    for key, data in t1["thresholds"].items():
        logger.info(
            "  Threshold %.2f: %d test rows (%.2f%%)",
            data["threshold"], data["test_row_count"], data["test_row_pct"],
        )

    t2s = report["task2_component_analysis"]["component_stats"]
    logger.info("Task 2 — Components (threshold=%.2f):", t2s["grouping_threshold"])
    logger.info("  Non-empty components: %d", t2s["n_components_non_empty"])
    logger.info("  Multi-row components: %d", t2s["n_multi_row_components"])
    logger.info("  Largest component:    %d rows", t2s["component_size_distribution"]["max"])

    t3 = report["task3_split"]["split_metadata"]
    tr = t3["train_partition"]
    te = t3["test_partition"]
    logger.info("Task 3 — Split:")
    logger.info("  Train: %d rows (%d fraud, %d legit)", tr["records"], tr["fraudulent_1"], tr["legitimate_0"])
    logger.info("  Test:  %d rows (%d fraud, %d legit)", te["records"], te["fraudulent_1"], te["legitimate_0"])
    logger.info("  Component overlap: %d", t3["contamination_checks"]["component_overlap_count"])

    resid = report["task3_split"]["residual_similarities"]
    logger.info("Task 3 — Residual similarities post-split:")
    for key, data in resid.items():
        if isinstance(data, dict) and "threshold" in data:
            logger.info(
                "  Threshold %.2f: %d rows (%.2f%%)",
                data["threshold"], data["residual_test_row_count"], data["residual_test_row_pct"],
            )

    t4 = report["task4_evaluation"]
    m = t4["metrics"]
    logger.info("Task 4 — Evaluation Metrics:")
    logger.info("  Accuracy:    %.4f", m["accuracy"])
    logger.info("  Fraud Prec:  %.4f", m["fraud_precision"])
    logger.info("  Fraud Recall:%.4f", m["fraud_recall"])
    logger.info("  Fraud F1:    %.4f", m["fraud_f1_score"])
    logger.info("  ROC-AUC:     %s", m["roc_auc"])
    logger.info("  PR-AUC:      %s", m["average_precision_pr_auc"])

    logger.info("=" * 70)
    logger.info("All artifacts saved to: %s", out_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
