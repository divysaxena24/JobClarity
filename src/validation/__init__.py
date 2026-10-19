"""
Dataset validation and integrity audit package for JobClarity.
"""

from src.validation.dataset_audit import (
    audit_dataset_dimensions_and_types,
    audit_duplicates,
    audit_missingness,
    audit_split_contamination,
    audit_target_leakage,
    audit_text_lengths,
    run_full_dataset_audit,
)
from src.validation.group_evaluator import (
    analyze_near_duplicate_risks,
    compute_description_hash,
    create_group_split,
    evaluate_group_candidate,
    generate_description_groups,
    run_group_aware_evaluation,
)
from src.validation.near_duplicate_grouper import (
    build_near_duplicate_components,
    analyze_components,
    create_near_duplicate_aware_split,
    report_residual_similarities,
    run_near_duplicate_aware_evaluation,
    run_similarity_audit,
    train_and_evaluate_near_dup_aware,
    compute_max_similarities_chunked,
)

__all__ = [
    # dataset_audit
    "audit_dataset_dimensions_and_types",
    "audit_duplicates",
    "audit_missingness",
    "audit_split_contamination",
    "audit_target_leakage",
    "audit_text_lengths",
    "run_full_dataset_audit",
    # group_evaluator
    "analyze_near_duplicate_risks",
    "compute_description_hash",
    "create_group_split",
    "evaluate_group_candidate",
    "generate_description_groups",
    "run_group_aware_evaluation",
    # near_duplicate_grouper
    "build_near_duplicate_components",
    "analyze_components",
    "create_near_duplicate_aware_split",
    "report_residual_similarities",
    "run_near_duplicate_aware_evaluation",
    "run_similarity_audit",
    "train_and_evaluate_near_dup_aware",
    "compute_max_similarities_chunked",
]
