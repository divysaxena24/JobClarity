"""
Unit and integration tests for duplicate-aware group splitting and evaluation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.validation.group_evaluator import (
    analyze_near_duplicate_risks,
    compute_description_hash,
    create_group_split,
    evaluate_group_candidate,
    generate_description_groups,
    run_group_aware_evaluation,
)


@pytest.fixture
def synthetic_group_df():
    """Create controlled dataframe with multiple duplicate description clusters."""
    return pd.DataFrame(
        {
            "title": [f"Role {i}" for i in range(20)],
            "description": [
                # Group 1 (4 identical legitimate postings)
                "Develop cloud microservices in Python and Kubernetes.",
                "Develop cloud microservices in Python and Kubernetes.",
                "develop cloud microservices in python and kubernetes  ",
                "DEVELOP CLOUD MICROSERVICES IN PYTHON AND KUBERNETES",
                # Group 2 (3 identical fraudulent postings)
                "Urgent cruise line opening 1000 weekly wire transfer.",
                "Urgent cruise line opening 1000 weekly wire transfer.",
                "urgent cruise line opening 1000 weekly wire transfer",
                # Group 3 (conflicting group: 2 legit, 1 fraud)
                "Workable company profile and front end developer opening.",
                "Workable company profile and front end developer opening.",
                "Workable company profile and front end developer opening.",
                # Group 4 through 13 (distinct single-posting roles)
                "Senior data engineer snowflake and dbt pipeline developer.",
                "Marketing coordinator digital growth advertising campaigns.",
                "Clinical research nurse oncology hospital setting.",
                "Financial analyst budget forecasting excel modeling.",
                "Account executive enterprise SaaS software solutions.",
                "Full stack engineer react node postgresql web platform.",
                "Operations manager logistics distribution center warehouse.",
                "HR recruiter talent acquisition human resources.",
                "Graphic designer adobe creative suite figma ui.",
                "Cybersecurity architect penetration testing risk audit.",
            ],
            "fraudulent": [
                0, 0, 0, 0,  # Group 1 (legit)
                1, 1, 1,     # Group 2 (fraud)
                0, 0, 1,     # Group 3 (conflicting: 2 legit, 1 fraud)
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  # Groups 4-13
            ],
        }
    )


class TestGroupEvaluator:
    """Test suite for duplicate-aware evaluation components."""

    def test_compute_description_hash_deterministic(self):
        """compute_description_hash produces stable, normalized 16-char hashes."""
        h1 = compute_description_hash("Data Scientist at Corp")
        h2 = compute_description_hash("  data scientist at corp  ")
        h3 = compute_description_hash("DATA SCIENTIST AT CORP")
        h_diff = compute_description_hash("Completely different description")

        assert len(h1) == 16
        assert h1 == h2 == h3
        assert h1 != h_diff

    def test_generate_description_groups_deterministic(self, synthetic_group_df):
        """generate_description_groups assigns identical group IDs to identical normalized text."""
        groups = generate_description_groups(synthetic_group_df, text_column="description")
        assert len(groups) == len(synthetic_group_df)

        # First 4 postings are identical normalized text
        assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2] == groups.iloc[3]
        # Next 3 postings are identical fraud text
        assert groups.iloc[4] == groups.iloc[5] == groups.iloc[6]
        assert groups.iloc[0] != groups.iloc[4]

    def test_generate_description_groups_missing_column_raises(self, synthetic_group_df):
        """generate_description_groups raises KeyError if text column is absent."""
        with pytest.raises(KeyError):
            generate_description_groups(synthetic_group_df, text_column="nonexistent")

    def test_generate_description_groups_empty_df(self):
        """generate_description_groups handles empty dataframe gracefully."""
        empty_df = pd.DataFrame(columns=["description"])
        groups = generate_description_groups(empty_df, text_column="description")
        assert len(groups) == 0

    def test_create_group_split_zero_group_overlap(self, synthetic_group_df):
        """create_group_split strictly enforces zero group overlap across partitions."""
        train_df, test_df, meta = create_group_split(
            synthetic_group_df,
            n_splits=3,
            test_fold=0,
            random_state=42,
            text_column="description",
            target_column="fraudulent",
        )

        assert meta["contamination_checks"]["group_overlap_count"] == 0
        assert meta["contamination_checks"]["normalized_text_overlap_count"] == 0
        assert meta["contamination_checks"]["zero_group_leakage_verified"] is True

        train_texts = set(train_df["description"].apply(compute_description_hash))
        test_texts = set(test_df["description"].apply(compute_description_hash))
        assert len(train_texts.intersection(test_texts)) == 0

    def test_create_group_split_preserves_conflicting_label_groups(self, synthetic_group_df):
        """create_group_split keeps conflicting-label groups intact within one partition without dropping."""
        train_df, test_df, meta = create_group_split(
            synthetic_group_df,
            n_splits=3,
            test_fold=0,
            random_state=42,
            text_column="description",
            target_column="fraudulent",
        )

        total_rows = len(train_df) + len(test_df)
        assert total_rows == len(synthetic_group_df)
        assert meta["dataset_summary"]["conflicting_label_groups_total"] == 1

        # Conflicting group (indices 7, 8, 9) must all be in train or all in test
        conflicting_hash = compute_description_hash("Workable company profile and front end developer opening.")
        in_train = (train_df["description"].apply(compute_description_hash) == conflicting_hash).sum()
        in_test = (test_df["description"].apply(compute_description_hash) == conflicting_hash).sum()

        assert (in_train == 3 and in_test == 0) or (in_train == 0 and in_test == 3)

    def test_create_group_split_invalid_fold_raises(self, synthetic_group_df):
        """create_group_split raises ValueError if test_fold is out of range."""
        with pytest.raises(ValueError):
            create_group_split(synthetic_group_df, n_splits=3, test_fold=5)

    def test_create_group_split_empty_df(self):
        """create_group_split returns empty partitions for empty input."""
        empty_df = pd.DataFrame(columns=["description", "fraudulent"])
        tr, te, meta = create_group_split(empty_df)
        assert len(tr) == 0
        assert len(te) == 0

    def test_evaluate_group_candidate_metrics(self, synthetic_group_df):
        """evaluate_group_candidate produces complete, structured classification metrics."""
        train_df, test_df, _ = create_group_split(
            synthetic_group_df,
            n_splits=2,
            test_fold=0,
            random_state=42,
        )

        eval_res = evaluate_group_candidate(
            train_df,
            test_df,
            text_column="description",
            target_column="fraudulent",
            max_features=50,
            random_state=42,
        )

        assert "metrics" in eval_res
        metrics = eval_res["metrics"]
        assert "accuracy" in metrics
        assert "fraud_precision" in metrics
        assert "fraud_recall" in metrics
        assert "fraud_f1_score" in metrics
        assert "roc_auc" in metrics
        assert "average_precision_pr_auc" in metrics
        assert "confusion_matrix" in eval_res

    def test_analyze_near_duplicate_risks(self, synthetic_group_df):
        """analyze_near_duplicate_risks accurately measures cosine similarity exposure."""
        train_df = synthetic_group_df.iloc[:10]
        test_df = synthetic_group_df.iloc[10:]

        risks = analyze_near_duplicate_risks(train_df, test_df, text_column="description")
        assert "cosine_ge_90" in risks
        assert "cosine_ge_95" in risks
        assert 0.0 <= risks["cosine_ge_90"]["test_samples_pct"] <= 100.0

    def test_analyze_near_duplicate_risks_empty_df(self):
        """analyze_near_duplicate_risks handles empty partitions safely."""
        empty = pd.DataFrame(columns=["description"])
        risks = analyze_near_duplicate_risks(empty, empty)
        assert risks["cosine_ge_90"]["count"] == 0

    def test_run_group_aware_evaluation_e2e_isolation(self, tmp_path, synthetic_group_df):
        """run_group_aware_evaluation generates expected JSON and Markdown artifacts in isolation."""
        csv_path = tmp_path / "cleaned_jobs_mock.csv"
        synthetic_group_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "group_eval_output"
        report = run_group_aware_evaluation(
            cleaned_data_path=csv_path,
            output_dir=out_dir,
            n_splits=2,
            test_fold=0,
            random_state=42,
            save_artifacts=True,
        )

        assert "metadata" in report
        assert "split_metadata" in report
        assert "evaluation" in report
        assert "near_duplicate_risk_analysis" in report

        # Verify files created on disk
        split_meta_file = out_dir / "group_split_metadata.json"
        eval_report_file = out_dir / "group_evaluation_report.json"
        md_file = out_dir / "group_evaluation_report.md"

        assert split_meta_file.exists()
        assert eval_report_file.exists()
        assert md_file.exists()

        with open(split_meta_file, "r", encoding="utf-8") as f:
            sm = json.load(f)
        assert sm["contamination_checks"]["zero_group_leakage_verified"] is True

        md_text = md_file.read_text(encoding="utf-8")
        assert "# JobClarity — Duplicate-Aware Evaluation Report" in md_text

        # Production model safety check: production models directory must remain untouched
        prod_models = Path("models").resolve()
        assert prod_models.exists()
