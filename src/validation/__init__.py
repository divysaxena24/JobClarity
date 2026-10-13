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

__all__ = [
    "audit_dataset_dimensions_and_types",
    "audit_duplicates",
    "audit_missingness",
    "audit_split_contamination",
    "audit_target_leakage",
    "audit_text_lengths",
    "run_full_dataset_audit",
    "analyze_near_duplicate_risks",
    "compute_description_hash",
    "create_group_split",
    "evaluate_group_candidate",
    "generate_description_groups",
    "run_group_aware_evaluation",
]
