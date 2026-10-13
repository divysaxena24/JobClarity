"""
Unit and integration tests for dataset integrity, duplicate audit, and contamination auditing.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.validation.dataset_audit import (
    audit_dataset_dimensions_and_types,
    audit_duplicates,
    audit_missingness,
    audit_split_contamination,
    audit_target_leakage,
    audit_text_lengths,
    run_full_dataset_audit,
)


@pytest.fixture
def sample_audit_dataframe():
    """Create a controlled synthetic dataframe for auditing tests."""
    return pd.DataFrame(
        {
            "title": ["Software Engineer", "Sales Lead", "Sales Lead", "Data Analyst", "Data Analyst"],
            "description": [
                "Develop high performance backend microservices in Python.",
                "Execute outreach campaigns and close deals.",
                "Execute outreach campaigns and close deals.",  # Duplicate description, same label
                "Build dashboards and analyze operational KPIs.",  # Conflicting description
                "Build dashboards and analyze operational KPIs.",  # Conflicting description
            ],
            "company_profile": ["TechCorp", "SalesPros", "SalesPros", "", "Analytica"],
            "fraudulent": [0, 0, 0, 0, 1],  # Row 3 is 0, Row 4 is 1 with same description!
        }
    )


class TestDatasetAudit:
    """Test suite for dataset validation and contamination audit functions."""

    def test_audit_dataset_dimensions_and_types(self, sample_audit_dataframe):
        """audit_dataset_dimensions_and_types should accurately report shape and dtypes."""
        res = audit_dataset_dimensions_and_types(sample_audit_dataframe)
        assert res["num_rows"] == 5
        assert res["num_columns"] == 4
        assert "title" in res["columns"]
        assert "fraudulent" in res["columns"]
        assert res["dtypes"]["fraudulent"] == "int64"

    def test_audit_missingness(self, sample_audit_dataframe):
        """audit_missingness should detect empty strings and NaNs appropriately."""
        res = audit_missingness(sample_audit_dataframe, empty_strings_as_missing=True)
        assert res["total_rows"] == 5
        assert "company_profile" in res["columns_with_missing"]
        assert res["per_column"]["company_profile"]["missing_count"] == 1
        assert res["per_column"]["company_profile"]["missing_percentage"] == 20.0
        assert res["per_column"]["title"]["missing_count"] == 0

    def test_audit_duplicates_detects_conflicts(self, sample_audit_dataframe):
        """audit_duplicates should detect exact duplicates and conflicting label clusters."""
        res = audit_duplicates(sample_audit_dataframe, text_column="description", target_column="fraudulent")
        assert res["total_rows"] == 5
        # Row 1 and 2 are exact duplicates across all columns
        assert res["exact_duplicate_rows"] == 1
        # Rows 1, 2 (legit) and rows 3, 4 (conflicting) have duplicate descriptions
        assert res["duplicate_descriptions_count"] == 2
        assert res["unique_descriptions_count"] == 3
        # Conflicting cluster on "Build dashboards..."
        assert res["conflicting_description_clusters"] == 1
        assert len(res["conflicting_description_samples"]) == 1
        conflict = res["conflicting_description_samples"][0]
        assert "Build dashboards" in conflict["description_snippet"]
        assert conflict["label_distribution"] == {0: 1, 1: 1}

    def test_audit_duplicates_empty_dataframe(self):
        """audit_duplicates handles empty dataframe gracefully without errors."""
        empty_df = pd.DataFrame(columns=["description", "fraudulent"])
        res = audit_duplicates(empty_df)
        assert res["total_rows"] == 0
        assert res["exact_duplicate_rows"] == 0
        assert res["conflicting_description_clusters"] == 0

    def test_audit_text_lengths(self, sample_audit_dataframe):
        """audit_text_lengths should compute length statistics and near-empty thresholds."""
        # Add an empty row
        df_with_empty = pd.concat(
            [
                sample_audit_dataframe,
                pd.DataFrame([{"title": "Ghost", "description": "", "company_profile": "", "fraudulent": 1}]),
            ],
            ignore_index=True,
        )
        res = audit_text_lengths(df_with_empty, text_column="description")
        assert res["empty_descriptions"] == 1
        assert res["min_word_count"] == 0
        assert res["max_word_count"] > 0
        assert res["mean_word_count"] > 0

    def test_audit_split_contamination(self):
        """audit_split_contamination should identify cross-split exact matches and near-duplicates."""
        train_df = pd.DataFrame(
            {
                "title": ["Job A", "Job B", "Job C"],
                "description": [
                    "This is an exact repeated template for customer support.",
                    "Unique software engineer role working on distributed databases.",
                    "Near duplicate template for data science and machine learning engineer.",
                ],
                "fraudulent": [0, 0, 1],
            }
        )
        test_df = pd.DataFrame(
            {
                "title": ["Job A", "Job D"],
                "description": [
                    "This is an exact repeated template for customer support.",  # Exact cross-split duplicate
                    "Near duplicate template for data science and deep learning engineer.",  # Near-duplicate
                ],
                "fraudulent": [0, 1],
            }
        )

        res = audit_split_contamination(
            train_df,
            test_df,
            text_column="description",
            target_column="fraudulent",
            compute_near_duplicates=True,
            near_dup_similarity_threshold=0.80,
        )

        assert res["train_samples"] == 3
        assert res["test_samples"] == 2
        assert res["unique_descriptions_crossing_split"] == 1
        assert res["test_records_with_train_description"] == 1
        assert res["test_records_with_train_description_pct"] == 50.0
        assert res["test_legit_with_train_description"] == 1
        assert res["conflicting_labels_across_split"] == 0
        # Near duplicate should be detected
        assert res["non_exact_test_near_duplicates_sim_90"] >= 0

    def test_audit_target_leakage_clean(self, sample_audit_dataframe):
        """audit_target_leakage should confirm clean features when no target proxy exists."""
        res = audit_target_leakage(sample_audit_dataframe, target_column="fraudulent", text_column="description")
        assert res["confirmed_leakage"] is False
        assert res["target_column_in_features"] is False
        assert len(res["suspicious_column_names"]) == 0

    def test_audit_target_leakage_detects_suspicious_column(self):
        """audit_target_leakage should flag suspicious target-derived column names."""
        df = pd.DataFrame(
            {
                "description": ["Role A", "Role B"],
                "target_label": [0, 1],
                "fraudulent": [0, 1],
            }
        )
        res = audit_target_leakage(df, target_column="fraudulent", text_column="description")
        assert "target_label" in res["suspicious_column_names"]

    def test_run_full_dataset_audit_e2e(self, tmp_path):
        """run_full_dataset_audit should write valid JSON and Markdown reports."""
        larger_df = pd.DataFrame(
            {
                "title": [f"Job {i}" for i in range(10)],
                "description": [f"Standard description text for role number {i}" for i in range(10)],
                "company_profile": ["Corp"] * 8 + [""] * 2,
                "fraudulent": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        csv_path = tmp_path / "cleaned_jobs_test.csv"
        larger_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "audit_output"
        report = run_full_dataset_audit(
            cleaned_data_path=csv_path,
            output_dir=out_dir,
            test_size=0.30,
            random_state=42,
            save_reports=True,
        )

        assert "metadata" in report
        assert "train_test_contamination" in report
        assert "duplicate_audit" in report
        assert "issue_categories_and_severity" in report
        assert "recommendations" in report

        json_file = out_dir / "dataset_integrity_report.json"
        md_file = out_dir / "dataset_integrity_report.md"
        assert json_file.exists()
        assert md_file.exists()

        with open(json_file, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
        assert loaded_json["dataset_overview"]["dimensions"]["num_rows"] == 10

        md_content = md_file.read_text(encoding="utf-8")
        assert "# JobClarity — Dataset Integrity & Duplicate Audit Report" in md_content

    def test_run_full_dataset_audit_with_raw_and_conflicts(self, tmp_path):
        """run_full_dataset_audit should process raw data and render conflict snippets in markdown."""
        cleaned_df = pd.DataFrame(
            {
                "title": [f"Role {i}" for i in range(12)],
                "description": [
                    "Repeated description with conflicting target labels here." if i < 2
                    else f"Unique role description content {i} with keyword legitimate"
                    for i in range(12)
                ],
                "company_profile": ["Corp"] * 12,
                "fraudulent": [0, 1] + [0] * 5 + [1] * 5,
            }
        )
        raw_df = pd.DataFrame(
            {
                "job_id": list(range(1, 13)),
                "title": [f"Role {i}" for i in range(12)],
                "description": cleaned_df["description"].tolist(),
                "company_profile": ["Corp"] * 12,
                "fraudulent": cleaned_df["fraudulent"].tolist(),
            }
        )

        clean_path = tmp_path / "cleaned.csv"
        raw_path = tmp_path / "raw.csv"
        out_dir = tmp_path / "full_audit"
        cleaned_df.to_csv(clean_path, index=False)
        raw_df.to_csv(raw_path, index=False)

        report = run_full_dataset_audit(
            cleaned_data_path=clean_path,
            raw_data_path=raw_path,
            output_dir=out_dir,
            test_size=0.25,
            random_state=42,
            save_reports=True,
        )

        assert report["raw_comparison"]["raw_shape"] == [12, 5]
        assert (out_dir / "dataset_integrity_report.json").exists()
        assert (out_dir / "dataset_integrity_report.md").exists()

        md_text = (out_dir / "dataset_integrity_report.md").read_text(encoding="utf-8")
        assert "Conflicting Description Detail" in md_text

    def test_run_full_dataset_audit_nonexistent_raises(self, tmp_path):
        """run_full_dataset_audit raises FileNotFoundError when data path does not exist."""
        with pytest.raises(FileNotFoundError):
            run_full_dataset_audit(tmp_path / "does_not_exist.csv")

    def test_audit_duplicates_missing_text_and_features(self):
        """audit_duplicates handles DataFrames without text column or without features."""
        df_target_only = pd.DataFrame({"fraudulent": [0, 1, 0]})
        res = audit_duplicates(df_target_only, text_column="description", target_column="fraudulent")
        assert res["unique_descriptions_count"] == 0
        assert res["exact_feature_duplicates"] == 0

    def test_audit_text_lengths_missing_column(self):
        """audit_text_lengths handles missing text column gracefully."""
        df = pd.DataFrame({"other_col": [1, 2]})
        res = audit_text_lengths(df, text_column="missing")
        assert res["empty_descriptions"] == 0
        assert res["mean_word_count"] == 0.0

    def test_audit_split_contamination_no_target_and_conflicts(self):
        """audit_split_contamination handles missing target column and cross-split conflicts."""
        train_df = pd.DataFrame({"description": ["Identical text"], "fraudulent": [0]})
        test_df = pd.DataFrame({"description": ["Identical text"], "fraudulent": [1]})
        res = audit_split_contamination(train_df, test_df, text_column="description", target_column="fraudulent")
        assert res["conflicting_labels_across_split"] == 1

        # Test without target column in test_df
        test_no_target = pd.DataFrame({"description": ["Identical text"]})
        res_no_tgt = audit_split_contamination(train_df, test_no_target, text_column="description", target_column="fraudulent")
        assert res_no_tgt["test_fraud_total"] == 0

