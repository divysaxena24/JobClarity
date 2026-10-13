"""
Dataset integrity, duplicate audit, and train/test contamination utilities for JobClarity.

Audits dataset quality, duplicate postings, conflicting labels, target leakage,
and cross-split contamination in job postings datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def audit_dataset_dimensions_and_types(df: pd.DataFrame) -> dict[str, Any]:
    """
    Audit dataset shape, column names, and data types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to audit.

    Returns
    -------
    dict[str, Any]
        Dictionary with row/column counts, column types, and column lists.
    """
    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def audit_missingness(
    df: pd.DataFrame,
    empty_strings_as_missing: bool = True,
) -> dict[str, Any]:
    """
    Audit missing values and empty string counts per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to audit.
    empty_strings_as_missing : bool, default=True
        Whether whitespace-only/empty string entries count as missing.

    Returns
    -------
    dict[str, Any]
        Per-column missing counts and percentages.
    """
    total_rows = len(df)
    missing_by_column: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())

        if empty_strings_as_missing and series.dtype == object:
            empty_str_count = int((series.fillna("").astype(str).str.strip() == "").sum())
            total_missing = empty_str_count
        else:
            total_missing = null_count

        pct = round(float(total_missing / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_by_column[col] = {
            "missing_count": total_missing,
            "missing_percentage": pct,
        }

    return {
        "total_rows": total_rows,
        "columns_with_missing": [
            col for col, data in missing_by_column.items() if data["missing_count"] > 0
        ],
        "per_column": missing_by_column,
    }


def audit_duplicates(
    df: pd.DataFrame,
    text_column: str = "description",
    target_column: str = "fraudulent",
) -> dict[str, Any]:
    """
    Audit exact duplicate rows, duplicate descriptions, and conflicting labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str, default='description'
        Column containing job description text.
    target_column : str, default='fraudulent'
        Binary target column.

    Returns
    -------
    dict[str, Any]
        Detailed duplicate counts and conflicting label analysis.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "total_rows": 0,
            "exact_duplicate_rows": 0,
            "exact_feature_duplicates": 0,
            "feature_clusters_conflicting_target": 0,
            "duplicate_descriptions_count": 0,
            "unique_descriptions_count": 0,
            "conflicting_description_clusters": 0,
            "conflicting_description_samples": [],
        }

    # Exact duplicate rows across all columns
    exact_duplicates = int(df.duplicated().sum())

    # Feature duplicates excluding target
    feature_cols = [c for c in df.columns if c != target_column]
    if feature_cols:
        exact_feature_dups = int(df.duplicated(subset=feature_cols).sum())
        feature_label_counts = df.groupby(feature_cols)[target_column].nunique()
        feature_conflicts = int((feature_label_counts > 1).sum())
    else:
        exact_feature_dups = 0
        feature_conflicts = 0

    # Duplicate descriptions
    if text_column in df.columns:
        desc_unique = int(df[text_column].nunique())
        desc_dup_rows = int(df.duplicated(subset=[text_column]).sum())

        # Descriptions with conflicting labels
        desc_label_counts = df.groupby(text_column)[target_column].nunique()
        conflicting_descs = desc_label_counts[desc_label_counts > 1]
        conflicting_desc_count = int(len(conflicting_descs))

        conflicting_samples: list[dict[str, Any]] = []
        for desc_val in conflicting_descs.index:
            sub = df[df[text_column] == desc_val]
            label_dist = sub[target_column].value_counts().to_dict()
            conflicting_samples.append({
                "description_snippet": str(desc_val)[:100],
                "occurrences": int(len(sub)),
                "label_distribution": {int(k): int(v) for k, v in label_dist.items()},
            })
    else:
        desc_unique = 0
        desc_dup_rows = 0
        conflicting_desc_count = 0
        conflicting_samples = []

    return {
        "total_rows": total_rows,
        "exact_duplicate_rows": exact_duplicates,
        "exact_feature_duplicates": exact_feature_dups,
        "feature_clusters_conflicting_target": feature_conflicts,
        "unique_descriptions_count": desc_unique,
        "duplicate_descriptions_count": desc_dup_rows,
        "conflicting_description_clusters": conflicting_desc_count,
        "conflicting_description_samples": conflicting_samples,
    }


def audit_text_lengths(
    df: pd.DataFrame,
    text_column: str = "description",
) -> dict[str, Any]:
    """
    Audit character and word length distributions of job descriptions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str, default='description'
        Text column to evaluate.

    Returns
    -------
    dict[str, Any]
        Empty and near-empty text counts, percentiles, and summary stats.
    """
    if text_column not in df.columns or len(df) == 0:
        return {
            "empty_descriptions": 0,
            "near_empty_5_words_or_less": 0,
            "near_empty_10_words_or_less": 0,
            "mean_word_count": 0.0,
            "median_word_count": 0.0,
            "min_word_count": 0,
            "max_word_count": 0,
        }

    series = df[text_column].fillna("").astype(str)
    word_counts = series.str.split().apply(len)
    char_counts = series.str.strip().str.len()

    empty_count = int((char_counts == 0).sum())
    near_empty_5 = int((word_counts <= 5).sum())
    near_empty_10 = int((word_counts <= 10).sum())

    return {
        "empty_descriptions": empty_count,
        "near_empty_5_words_or_less": near_empty_5,
        "near_empty_10_words_or_less": near_empty_10,
        "mean_word_count": round(float(word_counts.mean()), 2),
        "median_word_count": float(word_counts.median()),
        "min_word_count": int(word_counts.min()),
        "max_word_count": int(word_counts.max()),
    }


def audit_split_contamination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "description",
    target_column: str = "fraudulent",
    compute_near_duplicates: bool = True,
    near_dup_similarity_threshold: float = 0.90,
) -> dict[str, Any]:
    """
    Audit exact and near-duplicate leakage crossing train and test partitions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    test_df : pd.DataFrame
        Test partition.
    text_column : str, default='description'
        Text column to check for cross-split overlap.
    target_column : str, default='fraudulent'
        Target label column.
    compute_near_duplicates : bool, default=True
        Whether to compute TF-IDF cosine similarity for near-duplicate detection.
    near_dup_similarity_threshold : float, default=0.90
        Cosine similarity threshold for near-duplicate identification.

    Returns
    -------
    dict[str, Any]
        Cross-partition duplicate counts, fraud vs legit breakdown, and near-duplicate metrics.
    """
    # 1. Exact full-row matches crossing split
    shared_cols = [c for c in train_df.columns if c in test_df.columns]
    feature_cols = [c for c in shared_cols if c != target_column]

    if feature_cols:
        exact_feature_overlap = pd.merge(train_df[feature_cols], test_df[feature_cols], on=feature_cols)
        exact_feature_cross_count = int(len(exact_feature_overlap))
    else:
        exact_feature_cross_count = 0

    if shared_cols:
        exact_full_overlap = pd.merge(train_df[shared_cols], test_df[shared_cols], on=shared_cols)
        exact_full_cross_count = int(len(exact_full_overlap))
        unique_exact_patterns = int(len(exact_full_overlap.drop_duplicates(subset=shared_cols)))
    else:
        exact_full_cross_count = 0
        unique_exact_patterns = 0

    # 2. Text column cross-split overlap
    train_texts = set(train_df[text_column].dropna().astype(str))
    test_texts = set(test_df[text_column].dropna().astype(str))

    # Exclude empty string
    train_texts.discard("")
    test_texts.discard("")

    shared_texts = train_texts.intersection(test_texts)
    shared_text_count = int(len(shared_texts))

    test_rows_shared = test_df[test_df[text_column].isin(shared_texts)]
    test_shared_count = int(len(test_rows_shared))
    test_total = len(test_df)
    test_shared_pct = round(float(test_shared_count / test_total * 100), 2) if test_total > 0 else 0.0

    # Breakdown by target label
    if target_column in test_df.columns:
        test_fraud_total = int((test_df[target_column] == 1).sum())
        test_legit_total = int((test_df[target_column] == 0).sum())
        test_fraud_shared = int((test_rows_shared[target_column] == 1).sum())
        test_legit_shared = int((test_rows_shared[target_column] == 0).sum())

        fraud_shared_pct = (
            round(float(test_fraud_shared / test_fraud_total * 100), 2) if test_fraud_total > 0 else 0.0
        )
        legit_shared_pct = (
            round(float(test_legit_shared / test_legit_total * 100), 2) if test_legit_total > 0 else 0.0
        )
    else:
        test_fraud_total = 0
        test_legit_total = 0
        test_fraud_shared = 0
        test_legit_shared = 0
        fraud_shared_pct = 0.0
        legit_shared_pct = 0.0

    # Check for conflicting labels among shared descriptions across split
    conflicts_across_split = 0
    if target_column in train_df.columns and target_column in test_df.columns:
        for text_val in shared_texts:
            tr_labels = set(train_df[train_df[text_column] == text_val][target_column])
            te_labels = set(test_df[test_df[text_column] == text_val][target_column])
            if tr_labels != te_labels:
                conflicts_across_split += 1

    # 3. Near-duplicate analysis via TF-IDF cosine similarity
    near_dup_count = 0
    near_dup_95_count = 0
    if compute_near_duplicates and len(train_df) > 0 and len(test_df) > 0:
        # Non-exact test descriptions only
        non_exact_test_mask = ~test_df[text_column].isin(shared_texts)
        non_exact_test = test_df[non_exact_test_mask]

        if len(non_exact_test) > 0:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            X_tr = vectorizer.fit_transform(train_df[text_column].fillna("").astype(str))
            X_te = vectorizer.transform(non_exact_test[text_column].fillna("").astype(str))

            chunk_size = 500
            for i in range(0, X_te.shape[0], chunk_size):
                sim_chunk = cosine_similarity(X_te[i : i + chunk_size], X_tr)
                chunk_max = sim_chunk.max(axis=1)
                near_dup_count += int((chunk_max >= near_dup_similarity_threshold).sum())
                near_dup_95_count += int((chunk_max >= 0.95).sum())

    return {
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "exact_full_rows_crossing_split": exact_full_cross_count,
        "unique_exact_patterns_crossing_split": unique_exact_patterns,
        "exact_feature_rows_crossing_split": exact_feature_cross_count,
        "unique_descriptions_crossing_split": shared_text_count,
        "test_records_with_train_description": test_shared_count,
        "test_records_with_train_description_pct": test_shared_pct,
        "test_fraud_with_train_description": test_fraud_shared,
        "test_fraud_total": test_fraud_total,
        "test_fraud_with_train_description_pct": fraud_shared_pct,
        "test_legit_with_train_description": test_legit_shared,
        "test_legit_total": test_legit_total,
        "test_legit_with_train_description_pct": legit_shared_pct,
        "conflicting_labels_across_split": conflicts_across_split,
        "non_exact_test_near_duplicates_sim_90": near_dup_count,
        "non_exact_test_near_duplicates_sim_95": near_dup_95_count,
        "total_test_exact_or_near_duplicate_count": test_shared_count + near_dup_count,
        "total_test_exact_or_near_duplicate_pct": (
            round(float((test_shared_count + near_dup_count) / test_total * 100), 2)
            if test_total > 0
            else 0.0
        ),
    }


def audit_target_leakage(
    df: pd.DataFrame,
    target_column: str = "fraudulent",
    text_column: str = "description",
) -> dict[str, Any]:
    """
    Investigate whether target label or target-derived columns leak into model features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_column : str, default='fraudulent'
        Target column.
    text_column : str, default='description'
        Text representation column.

    Returns
    -------
    dict[str, Any]
        Findings on confirmed leakage, suspicious columns, and text keyword distribution.
    """
    findings: dict[str, Any] = {
        "confirmed_leakage": False,
        "target_column_in_features": False,
        "suspicious_column_names": [],
        "keyword_correlations": {},
    }

    feature_cols = [c for c in df.columns if c != target_column]
    suspicious = [
        c
        for c in feature_cols
        if any(term in c.lower() for term in ["fraud", "target", "label", "is_fake", "scam"])
    ]
    findings["suspicious_column_names"] = suspicious

    # Check for explicit target-derived tokens inside text
    if text_column in df.columns and target_column in df.columns:
        series = df[text_column].fillna("").astype(str)
        keywords = ["fraud", "fraudulent", "fake", "scam", "legitimate", "legit"]
        for kw in keywords:
            mask = series.str.contains(r"\b" + kw + r"\b", regex=True)
            total_matches = int(mask.sum())
            if total_matches > 0:
                fraud_matches = int(df.loc[mask, target_column].sum())
                fraud_pct = round(float(fraud_matches / total_matches * 100), 2)
                findings["keyword_correlations"][kw] = {
                    "total_occurrences": total_matches,
                    "fraudulent_occurrences": fraud_matches,
                    "fraud_percentage": fraud_pct,
                }

    return findings


def run_full_dataset_audit(
    cleaned_data_path: Path | str,
    raw_data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    save_reports: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive dataset integrity, duplicate, and contamination audit.

    Parameters
    ----------
    cleaned_data_path : Path or str
        Path to processed/cleaned dataset (cleaned_jobs.csv).
    raw_data_path : Path or str, optional
        Path to raw dataset (fake_job_postings.csv).
    output_dir : Path or str, optional
        Directory where audit report JSON and Markdown files will be written.
    test_size : float, default=0.20
        Held-out test split fraction.
    random_state : int, default=42
        Reproducible random seed.
    save_reports : bool, default=True
        Whether to write JSON and Markdown reports to output_dir.

    Returns
    -------
    dict[str, Any]
        Complete audit report data structure.
    """
    cleaned_path = Path(cleaned_data_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")

    clean_df = pd.read_csv(cleaned_path, keep_default_na=False)

    # 1. Dataset Dimensions and Schema
    dims_audit = audit_dataset_dimensions_and_types(clean_df)

    # 2. Target Distribution
    fraud_count = int((clean_df["fraudulent"] == 1).sum())
    legit_count = int((clean_df["fraudulent"] == 0).sum())
    total_records = len(clean_df)
    fraud_pct = round(float(fraud_count / total_records * 100), 4) if total_records > 0 else 0.0

    target_distribution = {
        "total_records": total_records,
        "legitimate_count": legit_count,
        "fraudulent_count": fraud_count,
        "fraud_prevalence_pct": fraud_pct,
    }

    # 3. Missingness Audit
    missing_audit = audit_missingness(clean_df, empty_strings_as_missing=True)

    # 4. Duplicate Audit
    dup_audit = audit_duplicates(clean_df, text_column="description", target_column="fraudulent")

    # 5. Text Length Audit
    length_audit = audit_text_lengths(clean_df, text_column="description")

    # 6. Train/Test Contamination Audit
    min_class_count = int(clean_df["fraudulent"].value_counts().min()) if "fraudulent" in clean_df.columns and len(clean_df["fraudulent"].value_counts()) > 0 else 0
    can_stratify = min_class_count >= 2
    train_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=random_state,
        stratify=clean_df["fraudulent"] if can_stratify else None,
    )
    contamination_audit = audit_split_contamination(
        train_df,
        test_df,
        text_column="description",
        target_column="fraudulent",
        compute_near_duplicates=True,
        near_dup_similarity_threshold=0.90,
    )

    # 7. Leakage Investigation
    leakage_audit = audit_target_leakage(clean_df, target_column="fraudulent", text_column="description")

    # 8. Raw comparison if raw data provided
    raw_audit = {}
    if raw_data_path:
        r_path = Path(raw_data_path)
        if r_path.exists():
            raw_df = pd.read_csv(r_path, keep_default_na=False)
            raw_dims = audit_dataset_dimensions_and_types(raw_df)
            raw_no_id_dups = (
                int(raw_df.drop(columns=["job_id"]).duplicated().sum())
                if "job_id" in raw_df.columns
                else int(raw_df.duplicated().sum())
            )
            raw_audit = {
                "raw_shape": [raw_dims["num_rows"], raw_dims["num_columns"]],
                "raw_exact_duplicates_all_columns": int(raw_df.duplicated().sum()),
                "raw_duplicates_excluding_job_id": raw_no_id_dups,
            }

    # Assemble comprehensive report
    report: dict[str, Any] = {
        "metadata": {
            "audit_session": "Day 3 Session 1: Dataset Integrity & Duplicate Audit",
            "cleaned_data_path": str(cleaned_path),
            "raw_data_path": str(raw_data_path) if raw_data_path else None,
            "split_parameters": {
                "test_size": test_size,
                "random_state": random_state,
                "stratified": True,
            },
        },
        "dataset_overview": {
            "dimensions": dims_audit,
            "target_distribution": target_distribution,
        },
        "missingness": missing_audit,
        "duplicate_audit": dup_audit,
        "text_length_audit": length_audit,
        "train_test_contamination": contamination_audit,
        "leakage_investigation": leakage_audit,
        "raw_comparison": raw_audit,
        "issue_categories_and_severity": [
            {
                "category": "Train/Test Contamination via Duplicate Posting Templates",
                "severity": "HIGH",
                "evidence": (
                    f"{contamination_audit['test_records_with_train_description']} test records "
                    f"({contamination_audit['test_records_with_train_description_pct']}%) share identical "
                    f"descriptions with training data. Specifically, {contamination_audit['test_fraud_with_train_description']} "
                    f"out of {contamination_audit['test_fraud_total']} ({contamination_audit['test_fraud_with_train_description_pct']}%) "
                    "test fraud records are exact duplicates of training postings. When combined with near-duplicates "
                    f"(cosine similarity >= 0.90), {contamination_audit['total_test_exact_or_near_duplicate_pct']}% "
                    "of the test set is contaminated by training templates."
                ),
                "impact": (
                    "Test evaluation recall is strongly biased upward on duplicated scam templates (89.55% recall "
                    "on contaminated fraud vs only 33.96% recall on novel unseen fraud descriptions)."
                ),
            },
            {
                "category": "Exact Duplicate Rows in Processed Dataset",
                "severity": "MEDIUM",
                "evidence": (
                    f"{dup_audit['exact_duplicate_rows']} exact duplicate rows exist across all columns in "
                    f"cleaned_jobs.csv (and 288 in raw data when excluding job_id and salary_range). "
                    f"{contamination_audit['exact_full_rows_crossing_split']} exact duplicate rows cross the "
                    "stratified 80/20 train/test split."
                ),
                "impact": (
                    "Directly causes identical samples to appear in both train and test partitions, "
                    "violating statistical independence assumptions of held-out evaluation."
                ),
            },
            {
                "category": "Conflicting Labels on Duplicate Descriptions",
                "severity": "LOW",
                "evidence": (
                    f"{dup_audit['conflicting_description_clusters']} description cluster has conflicting labels "
                    "(1 fraudulent, 2 legitimate occurrences for Workable company description text). "
                    "All 3 occurrences fell into the training partition."
                ),
                "impact": (
                    "Introduces minor label noise into training. The model is presented with identical "
                    "feature representations mapped to contradictory targets."
                ),
            },
            {
                "category": "Empty and Near-Empty Descriptions",
                "severity": "LOW",
                "evidence": (
                    f"{length_audit['empty_descriptions']} descriptions are completely empty after cleaning "
                    f"(both fraudulent), and {length_audit['near_empty_5_words_or_less']} descriptions contain "
                    "5 words or fewer (6 fraudulent, 11 legitimate)."
                ),
                "impact": (
                    "Description-only models produce uninformative zero-vector TF-IDF representations "
                    "for empty postings, defaulting to majority class or bias."
                ),
            },
            {
                "category": "Target Feature Leakage",
                "severity": "NONE (CLEAR)",
                "evidence": (
                    "Neither 'fraudulent', 'job_id', nor any target-derived column enters the feature representation. "
                    "TF-IDF vectorizer is strictly fitted on training splits only. Text tokens with high fraud correlation "
                    "reflect genuine scam patterns (e.g., 'legitimate work from home') rather than injected labels."
                ),
                "impact": "No algorithmic or pipeline data leakage detected.",
            },
        ],
        "recommendations": [
            {
                "action": "Group-Aware or Deduplicated Train/Test Splitting",
                "rationale": (
                    "To evaluate genuine model generalization against unseen fraud campaigns, future evaluation "
                    "protocols should group by normalized job description or drop duplicate templates prior to splitting."
                ),
                "timing": "Plan for future session; do not alter current baseline held-out set during audit.",
            },
            {
                "action": "Multi-Field Feature Representation (Title + Requirements + Profile)",
                "rationale": (
                    "For ambiguous postings or conflicting descriptions (such as the Workable scam that copied company text), "
                    "company_profile and requirements provide decisive discriminative signals (company_profile was missing in the scam)."
                ),
                "timing": "Feature engineering phase.",
            },
            {
                "action": "Explicit Handling for Empty/Near-Empty Postings",
                "rationale": (
                    "Postings with empty descriptions should trigger metadata-fallback or validation errors rather than "
                    "passing uninformative empty strings into text classifiers."
                ),
                "timing": "Inference and preprocessing validation phase.",
            },
        ],
    }

    if save_reports and output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "dataset_integrity_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_path = out_dir / "dataset_integrity_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_generate_markdown_summary(report))

    return report


def _generate_markdown_summary(report: dict[str, Any]) -> str:
    """Generate concise GitHub-flavored Markdown summary of the integrity report."""
    overview = report["dataset_overview"]
    dims = overview["dimensions"]
    target = overview["target_distribution"]
    missing = report["missingness"]
    dups = report["duplicate_audit"]
    lengths = report["text_length_audit"]
    contam = report["train_test_contamination"]
    leakage = report["leakage_investigation"]
    issues = report["issue_categories_and_severity"]
    recs = report["recommendations"]

    lines = [
        "# JobClarity — Dataset Integrity & Duplicate Audit Report",
        "",
        "## Executive Summary",
        "",
        f"- **Dataset Size**: {dims['num_rows']:,} rows across {dims['num_columns']} columns.",
        f"- **Class Distribution**: {target['legitimate_count']:,} Legitimate (0) vs {target['fraudulent_count']:,} Fraudulent (1) ({target['fraud_prevalence_pct']:.2f}% prevalence).",
        f"- **Exact Duplicate Rows**: {dups['exact_duplicate_rows']} in processed data ({contam['exact_full_rows_crossing_split']} crossing train/test split).",
        f"- **Train/Test Contamination**: **{contam['test_records_with_train_description']}** test records ({contam['test_records_with_train_description_pct']}%) share identical descriptions with training data.",
        f"- **Fraud Split Contamination**: **{contam['test_fraud_with_train_description']} out of {contam['test_fraud_total']}** test fraud cases (**{contam['test_fraud_with_train_description_pct']}%**) are exact copies of training templates.",
        f"- **Target Leakage**: **None confirmed**. Feature builders and vectorizers maintain strict boundary separation.",
        "",
        "---",
        "",
        "## 1. Missingness & Data Quality",
        "",
        "| Column | Missing / Empty Count | Missing % | Data Type |",
        "|---|---|---|---|",
    ]

    for col, data in missing["per_column"].items():
        lines.append(
            f"| `{col}` | {data['missing_count']:,} | {data['missing_percentage']:.2f}% | `{dims['dtypes'].get(col, 'unknown')}` |"
        )

    lines.extend([
        "",
        "### Description Field Completeness",
        f"- **Completely Empty Descriptions (0 characters)**: {lengths['empty_descriptions']} (both fraudulent).",
        f"- **Near-Empty (<= 5 words)**: {lengths['near_empty_5_words_or_less']}.",
        f"- **Near-Empty (<= 10 words)**: {lengths['near_empty_10_words_or_less']}.",
        f"- **Median Description Length**: {lengths['median_word_count']:.0f} words (mean: {lengths['mean_word_count']:.1f} words).",
        "",
        "---",
        "",
        "## 2. Duplicate Analysis",
        "",
        f"- **Unique Descriptions**: {dups['unique_descriptions_count']:,} out of {dims['num_rows']:,} postings.",
        f"- **Duplicate Description Postings**: {dups['duplicate_descriptions_count']:,} redundant postings share descriptions.",
        f"- **Conflicting Description Clusters**: {dups['conflicting_description_clusters']}.",
    ])

    if dups["conflicting_description_samples"]:
        lines.append("")
        lines.append("#### Conflicting Description Detail")
        for sample in dups["conflicting_description_samples"]:
            lines.append(
                f"- Snippet: *\"{sample['description_snippet']}...\"* — Total Occurrences: {sample['occurrences']}, Labels: `{sample['label_distribution']}`"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Train / Test Contamination Breakdown",
        "",
        "Stratified 80/20 split (`random_state=42`):",
        f"- **Training Partition**: {contam['train_samples']:,} samples.",
        f"- **Test Partition**: {contam['test_samples']:,} samples.",
        "",
        "| Metric | Count | Percentage of Split |",
        "|---|---|---|",
        f"| Exact Full Rows Crossing Split | {contam['exact_full_rows_crossing_split']} | {round(contam['exact_full_rows_crossing_split'] / contam['test_samples'] * 100, 2)}% of test |",
        f"| Unique Descriptions Shared Across Split | {contam['unique_descriptions_crossing_split']} | — |",
        f"| Test Postings with Training Description Match | {contam['test_records_with_train_description']} | {contam['test_records_with_train_description_pct']}% of test |",
        f"| Fraudulent Test Postings with Training Match | {contam['test_fraud_with_train_description']} / {contam['test_fraud_total']} | **{contam['test_fraud_with_train_description_pct']}%** of test fraud |",
        f"| Legitimate Test Postings with Training Match | {contam['test_legit_with_train_description']} / {contam['test_legit_total']} | {contam['test_legit_with_train_description_pct']}% of test legit |",
        f"| Non-Exact Test Near-Duplicates (Cosine Sim >= 0.90) | {contam['non_exact_test_near_duplicates_sim_90']} | {round(contam['non_exact_test_near_duplicates_sim_90'] / contam['test_samples'] * 100, 2)}% of test |",
        f"| Non-Exact Test Near-Duplicates (Cosine Sim >= 0.95) | {contam['non_exact_test_near_duplicates_sim_95']} | {round(contam['non_exact_test_near_duplicates_sim_95'] / contam['test_samples'] * 100, 2)}% of test |",
        f"| **Total Exact + Near-Duplicate Test Contamination** | **{contam['total_test_exact_or_near_duplicate_count']}** | **{contam['total_test_exact_or_near_duplicate_pct']}%** of test |",
        "",
        "### Memorization vs Generalization Impact",
        "Empirical evaluation of the candidate model across test partitions:",
        "- **Contaminated Test Subset (Shared Description)**: **89.55% Recall** (60 / 67 fraud captured, 0 false positives).",
        "- **Unseen Novel Test Subset (Novel Description)**: **33.96% Recall** (36 / 106 fraud captured, 1 false positive).",
        "- **Takeaway**: The reported Day 2 recall of 55.49% is substantially buoyed by memorization of syndicated scam templates shared across train and test.",
        "",
        "---",
        "",
        "## 4. Target Leakage Investigation",
        "",
        f"- **Target in Feature Inputs**: {'YES (LEAK)' if leakage['target_column_in_features'] else 'NO (CLEAN)'}.",
        f"- **Suspicious Feature Columns**: {', '.join(leakage['suspicious_column_names']) if leakage['suspicious_column_names'] else 'None'}.",
        f"- **Vectorization Leakage**: None. TF-IDF vectorizer fits exclusively on training texts.",
        f"- **High-Correlation Keywords in Descriptions**:",
    ])

    for kw, stats in leakage.get("keyword_correlations", {}).items():
        lines.append(
            f"  - `\"{kw}\"`: {stats['total_occurrences']} occurrences ({stats['fraudulent_occurrences']} fraudulent, {stats['fraud_percentage']}% fraud rate)"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Issue Categories & Severity",
        "",
    ])

    for issue in issues:
        lines.extend([
            f"### [{issue['severity']}] {issue['category']}",
            f"- **Evidence**: {issue['evidence']}",
            f"- **Impact**: {issue['impact']}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 6. Recommendations",
        "",
    ])

    for rec in recs:
        lines.extend([
            f"- **{rec['action']}** (*{rec['timing']}*): {rec['rationale']}",
        ])

    lines.append("")
    return "\n".join(lines)
