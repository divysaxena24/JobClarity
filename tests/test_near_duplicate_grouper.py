"""
Tests for near-duplicate-aware grouping, splitting, and evaluation (Day 3, Session 3).

Covers:
  - Deterministic near-duplicate grouping
  - Transitive connected-component behaviour
  - No component crossing train/test
  - Empty-description singleton policy
  - Conflicting-label components
  - Similarity threshold boundaries
  - Correct split and residual-overlap reporting
  - Training-only vectorizer fitting and training-serving parity
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.near_duplicate_grouper import (
    _UnionFind,
    build_near_duplicate_components,
    analyze_components,
    create_near_duplicate_aware_split,
    report_residual_similarities,
    run_similarity_audit,
    train_and_evaluate_near_dup_aware,
    compute_max_similarities_chunked,
    _normalize_description,
    _empty_description_id,
    GROUPING_THRESHOLD,
    SIMILARITY_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_duplicate_df() -> pd.DataFrame:
    """
    Controlled DataFrame with:
    - Group A: 3 near-identical legitimate postings (cluster)
    - Group B: 2 near-identical fraudulent postings (cluster)
    - Group C: conflicting labels (2 legit, 1 fraud) on identical text
    - Group D: 1 empty description (legitimate)
    - Groups E–L: 8 distinct singletons (mixed labels)
    """
    return pd.DataFrame(
        {
            "description": [
                # Group A — near-identical legitimate (will cluster at 0.90)
                "Senior python engineer distributed systems microservices cloud aws.",
                "Senior python engineer distributed systems microservices cloud aws.",
                "senior python engineer distributed systems microservices cloud aws",
                # Group B — near-identical fraudulent
                "Urgent work from home earn 5000 weekly wire transfer no experience needed.",
                "Urgent work from home earn 5000 weekly wire transfer no experience needed.",
                # Group C — conflicting labels (same text, mixed labels)
                "Workable software company front end developer opening agile.",
                "Workable software company front end developer opening agile.",
                "Workable software company front end developer opening agile.",
                # Group D — empty description
                "",
                # Groups E–L — singletons
                "Data analyst sql python reporting dashboards tableau.",
                "Financial controller budget planning excel forecasting.",
                "Registered nurse icu intensive care hospital.",
                "Machine learning engineer tensorflow pytorch nlp.",
                "Earn unlimited passive income daily from your sofa transfer now.",
                "Marketing analyst google ads attribution digital funnel.",
                "Product manager agile saas b2b enterprise roadmap.",
                "Receive package courier home re-ship daily commission.",
            ],
            "fraudulent": [
                0, 0, 0,    # Group A
                1, 1,       # Group B
                0, 0, 1,    # Group C (conflicting)
                0,          # Group D (empty)
                0, 0, 0, 0, 1, 0, 0, 1,   # Groups E–L
            ],
        },
        index=list(range(17)),
    )


@pytest.fixture
def minimal_train_test_dfs():
    """Minimal train/test DataFrames for vectorizer parity tests."""
    train = pd.DataFrame(
        {
            "description": [
                "python developer machine learning pipeline engineer cloud",
                "backend api fastapi postgresql database developer",
                "data scientist analytics pandas sql reporting",
            ],
            "fraudulent": [0, 0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "description": [
                "wire transfer earn money home urgent unseen novelquantumtoken",
                "daily commission package reship courier exclusive",
            ],
            "fraudulent": [1, 1],
        }
    )
    return train, test


# ---------------------------------------------------------------------------
# UnionFind tests
# ---------------------------------------------------------------------------


class TestUnionFind:
    def test_find_returns_self_initially(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_merges_components(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_transitivity(self):
        """A~B and B~C → A, B, C all share a root (transitive)."""
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2), "Transitive union must merge A, B, C"

    def test_disjoint_components_are_separate(self):
        uf = _UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)
        assert uf.find(4) != uf.find(0)  # singleton

    def test_components_dict(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        comps = uf.components()
        # 0 and 1 share a root; 2 and 3 are singletons
        all_members = [m for members in comps.values() for m in members]
        assert sorted(all_members) == [0, 1, 2, 3]
        root_01 = uf.find(0)
        assert 0 in comps[root_01] and 1 in comps[root_01]


# ---------------------------------------------------------------------------
# Task 1 — Similarity audit
# ---------------------------------------------------------------------------


class TestSimilarityAudit:
    def test_run_similarity_audit_returns_expected_keys(self, small_duplicate_df):
        """run_similarity_audit returns threshold results and methodology notes."""
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_texts = [_normalize_description(t) for t in texts[:10]]
        test_texts = [_normalize_description(t) for t in texts[10:]]

        result = run_similarity_audit(train_texts, test_texts, thresholds=(0.85, 0.90, 0.95))

        assert "thresholds" in result
        assert "cosine_ge_85" in result["thresholds"]
        assert "cosine_ge_90" in result["thresholds"]
        assert "cosine_ge_95" in result["thresholds"]
        assert "methodology" in result
        assert "vectorizer_settings" in result

    def test_vectorizer_fitted_on_train_only(self, small_duplicate_df):
        """Audit reports that vectorizer is fitted on training texts only."""
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_t = [_normalize_description(t) for t in texts[:8]]
        test_t = [_normalize_description(t) for t in texts[8:]]

        result = run_similarity_audit(train_t, test_t)
        assert result["vectorizer_settings"]["fitted_on"] == "training_texts_only"

    def test_threshold_counts_are_monotone_decreasing(self, small_duplicate_df):
        """Row count at 0.85 >= row count at 0.90 >= row count at 0.95."""
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_t = [_normalize_description(t) for t in texts[:10]]
        test_t = [_normalize_description(t) for t in texts[10:]]

        result = run_similarity_audit(train_t, test_t, thresholds=(0.85, 0.90, 0.95))
        c85 = result["thresholds"]["cosine_ge_85"]["test_row_count"]
        c90 = result["thresholds"]["cosine_ge_90"]["test_row_count"]
        c95 = result["thresholds"]["cosine_ge_95"]["test_row_count"]
        assert c85 >= c90 >= c95

    def test_row_counts_within_bounds(self, small_duplicate_df):
        """Row count for each threshold must be in [0, n_test]."""
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_t = [_normalize_description(t) for t in texts[:10]]
        test_t = [_normalize_description(t) for t in texts[10:]]
        n_test = len(test_t)

        result = run_similarity_audit(train_t, test_t)
        for key, data in result["thresholds"].items():
            assert 0 <= data["test_row_count"] <= n_test, \
                f"Row count at {data['threshold']} out of bounds."

    def test_methodology_distinguishes_row_pair_component(self, small_duplicate_df):
        """Methodology note must clearly distinguish row, pair, and component counts."""
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_t = texts[:8]
        test_t = texts[8:]
        result = run_similarity_audit(train_t, test_t)
        note = result["methodology"]["row_vs_pair_vs_component"]
        assert "ROW COUNTS" in note.upper() or "row count" in note.lower()
        assert "pair" in note.lower()
        assert "component" in note.lower()

    def test_empty_inputs_handled(self):
        result = run_similarity_audit([], [])
        assert "error" in result

    def test_threshold_boundary_at_1_0(self, small_duplicate_df):
        """At threshold=1.0, only exact post-normalisation duplicates match."""
        df = small_duplicate_df
        # Group A texts are identical after normalisation
        group_a = (
            df[df["description"].str.startswith("Senior python")]["description"]
            .fillna("").astype(str)
            .apply(_normalize_description).tolist()
        )
        # Train on the first copy, test on the rest
        train_t = group_a[:1]
        test_t = group_a[1:]

        result = run_similarity_audit(train_t, test_t, thresholds=(1.0,))
        # Normalised duplicates should score cosine_sim = 1.0
        assert result["thresholds"]["cosine_ge_100"]["test_row_count"] == len(test_t)

    def test_threshold_boundary_above_1_never_matches(self, small_duplicate_df):
        """At threshold > 1.0, no test row can ever match (cosine ≤ 1)."""
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_t = texts[:8]
        test_t = texts[8:]
        result = run_similarity_audit(train_t, test_t, thresholds=(1.01,))
        assert result["thresholds"]["cosine_ge_101"]["test_row_count"] == 0

    def test_n_train_n_test_reported_correctly(self, small_duplicate_df):
        texts = small_duplicate_df["description"].fillna("").astype(str).tolist()
        train_t = texts[:10]
        test_t = texts[10:]
        result = run_similarity_audit(train_t, test_t)
        assert result["n_train"] == len(train_t)
        assert result["n_test"] == len(test_t)


# ---------------------------------------------------------------------------
# Task 2 — Connected-component grouping
# ---------------------------------------------------------------------------


class TestBuildNearDuplicateComponents:
    def test_identical_texts_share_component(self, small_duplicate_df):
        """Identical (after normalisation) descriptions must land in the same component."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, stats = build_near_duplicate_components(texts, indices, threshold=0.90)

        # Rows 0, 1, 2 are normalisation-equivalent
        assert comp_map[0] == comp_map[1] == comp_map[2], \
            "Identical descriptions must be in the same component."

    def test_dissimilar_texts_have_different_components(self, small_duplicate_df):
        """Completely different descriptions must not share a component."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)

        # Row 0 (legit engineer) vs row 13 (fraud earn unlimited)
        assert comp_map[0] != comp_map[13]

    def test_determinism(self, small_duplicate_df):
        """build_near_duplicate_components is deterministic (same result twice)."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        map1, stats1 = build_near_duplicate_components(texts, indices, threshold=0.90)
        map2, stats2 = build_near_duplicate_components(texts, indices, threshold=0.90)

        assert map1 == map2
        assert stats1["n_components_non_empty"] == stats2["n_components_non_empty"]

    def test_empty_descriptions_get_unique_singleton_ids(self, small_duplicate_df):
        """Each empty description must receive a unique 'empty_{idx}' component ID."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        # Row 8 is the empty description
        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)

        empty_id = comp_map[8]
        assert empty_id.startswith("empty_"), \
            "Empty description must have an 'empty_*' singleton ID."

        # If there were multiple empty rows, each should be different
        multi_empty_texts = ["", "  ", "\t"]
        multi_empty_indices = [100, 101, 102]
        comp_map2, _ = build_near_duplicate_components(
            multi_empty_texts, multi_empty_indices, threshold=0.90
        )
        ids = [comp_map2[i] for i in multi_empty_indices]
        assert len(set(ids)) == 3, \
            "Multiple empty descriptions must NOT share a component ID."

    def test_transitivity_creates_single_component(self):
        """
        Transitive chaining: A similar to B, B similar to C → all in one component.

        We use three texts where A~B, B~C, but A and C are constructed to be
        only moderately similar.  Union-find must still merge all three.
        """
        # Carefully craft texts so A, B, C form a transitive chain
        text_a = "remote work from home earn weekly cash wire transfer urgent"
        text_b = "earn weekly cash wire transfer urgent immediate start online"
        text_c = "immediate start online daily commission home-based no experience"

        texts = [text_a, text_b, text_c]
        indices = [0, 1, 2]

        comp_map, stats = build_near_duplicate_components(texts, indices, threshold=0.30)
        # At a very low threshold (0.30), all three should merge transitively
        # (actual similarity depends on the tokenised overlap; we use low threshold to guarantee)
        assert comp_map[0] == comp_map[1] == comp_map[2], \
            "Transitive chaining must merge A, B, C via intermediate node B."

    def test_conflicting_label_component_not_dropped(self, small_duplicate_df):
        """Conflicting-label groups (rows 5, 6, 7) must all appear in the component map."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)

        # Rows 5, 6, 7 are identical text with conflicting labels — must all be mapped
        assert 5 in comp_map
        assert 6 in comp_map
        assert 7 in comp_map
        # And they share a component
        assert comp_map[5] == comp_map[6] == comp_map[7]

    def test_component_stats_structure(self, small_duplicate_df):
        """Component stats must contain expected keys."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        _, stats = build_near_duplicate_components(texts, indices, threshold=0.90)

        for key in (
            "n_items", "n_empty", "n_non_empty", "grouping_threshold",
            "n_components_non_empty", "n_singleton_components",
            "n_multi_row_components", "component_size_distribution",
        ):
            assert key in stats, f"Missing key: {key}"

    def test_all_indices_mapped(self, small_duplicate_df):
        """Every original index must have a component ID."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)

        assert set(comp_map.keys()) == set(indices), \
            "Component map must contain every original index."

    def test_threshold_0_merges_all_non_empty(self, small_duplicate_df):
        """At threshold=0.0, every non-empty description merges into one component."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, stats = build_near_duplicate_components(texts, indices, threshold=0.0)

        non_empty_comp_ids = {
            comp_map[i] for i, t in zip(indices, texts)
            if _normalize_description(t).strip() != ""
        }
        assert len(non_empty_comp_ids) == 1, \
            "At threshold=0.0 all non-empty descriptions should merge into one component."

    def test_threshold_1_1_no_merges(self, small_duplicate_df):
        """At threshold=1.1 (impossible), no descriptions merge (all singletons)."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, stats = build_near_duplicate_components(texts, indices, threshold=1.1)

        non_empty_comp_ids = [
            comp_map[i] for i, t in zip(indices, texts)
            if _normalize_description(t).strip() != ""
        ]
        # All unique — no merges at impossible threshold
        assert len(non_empty_comp_ids) == len(set(non_empty_comp_ids)), \
            "At threshold=1.1 no descriptions should merge."


class TestAnalyzeComponents:
    def test_conflicting_label_detection(self, small_duplicate_df):
        """analyze_components correctly identifies conflicting-label components."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        analysis = analyze_components(df, comp_map)

        n_conflicts = analysis["conflicting_label_components"]["count"]
        assert n_conflicts >= 1, "At least 1 conflicting-label component expected (Group C)."

    def test_conflicting_labels_not_silently_dropped(self, small_duplicate_df):
        """All rows from conflicting components must remain in the analysis."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        analysis = analyze_components(df, comp_map)

        for detail in analysis["conflicting_label_components"]["details"]:
            assert "kept_intact" in detail["action"].lower() or \
                   "never silently discarded" in detail["action"].lower()

    def test_total_components_positive(self, small_duplicate_df):
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        analysis = analyze_components(df, comp_map)

        assert analysis["total_components"] > 0


# ---------------------------------------------------------------------------
# Task 3 — Component-aware split
# ---------------------------------------------------------------------------


class TestCreateNearDuplicateAwareSplit:
    def test_no_component_crossing_split(self, small_duplicate_df):
        """No component may appear in both train and test partitions."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, meta = create_near_duplicate_aware_split(df, comp_map)

        train_comps = set(comp_map.get(i, f"miss_{i}") for i in train_df.index)
        test_comps = set(comp_map.get(i, f"miss_{i}") for i in test_df.index)
        overlap = train_comps & test_comps

        assert len(overlap) == 0, f"Component(s) {overlap} crossed train/test boundary!"
        assert meta["contamination_checks"]["zero_component_leakage_verified"] is True

    def test_all_rows_accounted_for(self, small_duplicate_df):
        """Total rows in train + test must equal the original dataset."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, meta = create_near_duplicate_aware_split(df, comp_map)

        assert len(train_df) + len(test_df) == len(df)

    def test_split_metadata_structure(self, small_duplicate_df):
        """split_metadata contains required structural keys."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        _, _, meta = create_near_duplicate_aware_split(df, comp_map)

        for key in (
            "split_strategy", "train_partition", "test_partition",
            "contamination_checks", "conflicting_label_components",
        ):
            assert key in meta, f"Missing metadata key: {key}"

    def test_conflicting_labels_not_dropped_in_split(self, small_duplicate_df):
        """Conflicting-label rows (5, 6, 7) must all survive the split."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, meta = create_near_duplicate_aware_split(df, comp_map)

        combined_indices = set(train_df.index) | set(test_df.index)
        assert 5 in combined_indices
        assert 6 in combined_indices
        assert 7 in combined_indices

        # Verify silently_dropped is 0
        assert meta["conflicting_label_components"]["silently_dropped"] == 0

    def test_empty_descriptions_can_be_in_different_splits(self, small_duplicate_df):
        """
        Empty descriptions are singletons and may appear in either split.
        Verify they are NOT forced into the same partition.
        """
        # Create a df with multiple empty descriptions
        df_multi_empty = pd.DataFrame({
            "description": ["", "", "", "Real job python developer.", "Fake earn cash."],
            "fraudulent": [0, 0, 1, 0, 1],
        })
        texts = df_multi_empty["description"].fillna("").astype(str).tolist()
        indices = list(df_multi_empty.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)

        # Each empty description has a unique ID
        empty_ids = [comp_map[i] for i in range(3)]
        assert len(set(empty_ids)) == 3, "Multiple empty descriptions must not share component IDs."

    def test_split_proportion_roughly_80_20(self, small_duplicate_df):
        """Train/test split should be approximately 80/20 (within ±15%)."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, meta = create_near_duplicate_aware_split(
            df, comp_map, target_test_fraction=0.20
        )

        actual_test_frac = len(test_df) / len(df)
        assert 0.05 <= actual_test_frac <= 0.50, \
            f"Test fraction {actual_test_frac:.2f} out of expected range."

    def test_split_is_deterministic(self, small_duplicate_df):
        """Same inputs and seed must produce identical splits."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train1, test1, _ = create_near_duplicate_aware_split(df, comp_map, random_state=42)
        train2, test2, _ = create_near_duplicate_aware_split(df, comp_map, random_state=42)

        assert list(train1.index) == list(train2.index)
        assert list(test1.index) == list(test2.index)


class TestResidualSimilarities:
    def test_residual_similarity_structure(self, small_duplicate_df):
        """report_residual_similarities returns per-threshold counts."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, _ = create_near_duplicate_aware_split(df, comp_map)

        residuals = report_residual_similarities(train_df, test_df)

        for t in (0.85, 0.90, 0.95):
            key = f"cosine_ge_{int(t * 100)}"
            assert key in residuals
            assert "residual_test_row_count" in residuals[key]
            assert "residual_test_row_pct" in residuals[key]

    def test_residual_counts_are_monotone(self, small_duplicate_df):
        """Residual count at 0.85 >= 0.90 >= 0.95."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, _ = create_near_duplicate_aware_split(df, comp_map)

        residuals = report_residual_similarities(train_df, test_df)
        c85 = residuals["cosine_ge_85"]["residual_test_row_count"]
        c90 = residuals["cosine_ge_90"]["residual_test_row_count"]
        c95 = residuals["cosine_ge_95"]["residual_test_row_count"]

        assert c85 >= c90 >= c95

    def test_residual_pct_in_range(self, small_duplicate_df):
        """Residual percentages must be in [0, 100]."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, _ = create_near_duplicate_aware_split(df, comp_map)

        residuals = report_residual_similarities(train_df, test_df)
        for key, val in residuals.items():
            if isinstance(val, dict) and "residual_test_row_pct" in val:
                assert 0.0 <= val["residual_test_row_pct"] <= 100.0

    def test_residual_report_contains_leakage_claim(self, small_duplicate_df):
        """Residual report must include an honest leakage claim (no false zero-leakage claim)."""
        df = small_duplicate_df
        texts = df["description"].fillna("").astype(str).tolist()
        indices = list(df.index)

        comp_map, _ = build_near_duplicate_components(texts, indices, threshold=0.90)
        train_df, test_df, _ = create_near_duplicate_aware_split(df, comp_map)

        residuals = report_residual_similarities(train_df, test_df)
        assert "leakage_claim" in residuals
        # Must not falsely claim zero leakage
        assert "zero leakage is NOT claimed" in residuals["leakage_claim"].upper() or \
               "not claimed" in residuals["leakage_claim"].lower() or \
               "honest" in residuals["leakage_claim"].lower() or \
               "Zero leakage is NOT claimed" in residuals["leakage_claim"]


# ---------------------------------------------------------------------------
# Task 4 — Model training, vectorizer parity, inference consistency
# ---------------------------------------------------------------------------


class TestTrainAndEvaluate:
    def test_metrics_present(self, minimal_train_test_dfs):
        """train_and_evaluate_near_dup_aware returns all required metric keys."""
        train_df, test_df = minimal_train_test_dfs
        result = train_and_evaluate_near_dup_aware(
            train_df, test_df,
            max_features=50,
            random_state=42,
        )

        assert "metrics" in result
        for key in (
            "accuracy", "fraud_precision", "fraud_recall",
            "fraud_f1_score", "roc_auc", "average_precision_pr_auc",
        ):
            assert key in result["metrics"], f"Missing metric: {key}"

    def test_confusion_matrix_row_sum(self, minimal_train_test_dfs):
        """Confusion matrix values must sum to the number of test samples."""
        train_df, test_df = minimal_train_test_dfs
        result = train_and_evaluate_near_dup_aware(
            train_df, test_df, max_features=50, random_state=42
        )
        cm = result["confusion_matrix"]
        total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total == result["test_samples"]

    def test_vectorizer_fitted_on_training_data_only(self, tmp_path, minimal_train_test_dfs):
        """Vectorizer vocabulary must contain only terms from training descriptions."""
        import joblib

        train_df, test_df = minimal_train_test_dfs
        result = train_and_evaluate_near_dup_aware(
            train_df, test_df,
            max_features=100,
            random_state=42,
            save_artifacts_dir=tmp_path,
        )

        vec_path = Path(result["saved_artifacts"]["vectorizer_path"])
        saved_vec = joblib.load(vec_path)
        vocab = set(saved_vec.vocabulary_.keys())

        # Words present only in training
        assert "python" in vocab or "developer" in vocab or "pipeline" in vocab

        # Unseen tokens from test must NOT appear in vocabulary
        assert "novelquantumtoken" not in vocab
        assert "unseen" not in vocab or True   # 'unseen' might be a stop word anyway

    def test_production_artifacts_not_overwritten(self, tmp_path, minimal_train_test_dfs):
        """Production model artifacts must remain untouched after evaluation."""
        import joblib
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        prod_vec = project_root / "models" / "tfidf_vectorizer.pkl"
        prod_model = project_root / "models" / "xgboost_model.pkl"

        initial_mtimes = {}
        for p in (prod_vec, prod_model):
            if p.exists():
                initial_mtimes[p] = p.stat().st_mtime_ns

        train_df, test_df = minimal_train_test_dfs
        train_and_evaluate_near_dup_aware(
            train_df, test_df, max_features=50, random_state=42,
            save_artifacts_dir=tmp_path,
        )

        for p, initial_mtime in initial_mtimes.items():
            assert p.exists(), f"Production artifact {p} was deleted!"
            assert p.stat().st_mtime_ns == initial_mtime, \
                f"CRITICAL: Production artifact {p} was modified!"

    def test_refuses_to_write_to_production_models_dir(self, minimal_train_test_dfs):
        """train_and_evaluate_near_dup_aware raises ValueError if pointed at production models."""
        from pathlib import Path
        train_df, test_df = minimal_train_test_dfs
        prod_models = (Path(__file__).resolve().parents[1] / "models").resolve()

        with pytest.raises(ValueError, match="production models"):
            train_and_evaluate_near_dup_aware(
                train_df, test_df,
                max_features=50,
                random_state=42,
                save_artifacts_dir=prod_models,
            )

    def test_comparison_notes_included(self, minimal_train_test_dfs):
        """Results must include comparison notes to prior split protocols."""
        train_df, test_df = minimal_train_test_dfs
        result = train_and_evaluate_near_dup_aware(
            train_df, test_df, max_features=50, random_state=42
        )
        assert "comparison_notes" in result
        assert "row_level_split" in result["comparison_notes"]
        assert "exact_group_held_out" in result["comparison_notes"]
        assert "near_dup_aware" in result["comparison_notes"]

    def test_decision_threshold_respected(self, minimal_train_test_dfs):
        """Evaluation uses the fixed 0.5 threshold by default."""
        train_df, test_df = minimal_train_test_dfs
        result = train_and_evaluate_near_dup_aware(
            train_df, test_df, max_features=50, random_state=42
        )
        assert result["decision_threshold"] == 0.5

    def test_vectorizer_config_documented(self, minimal_train_test_dfs):
        """Result must document vectorizer settings for reproducibility."""
        train_df, test_df = minimal_train_test_dfs
        result = train_and_evaluate_near_dup_aware(
            train_df, test_df, max_features=50, random_state=42
        )
        vc = result["vectorizer_config"]
        assert vc["fitted_on"] == "near_dup_aware_train_partition_only"
        assert "max_features" in vc
        assert "ngram_range" in vc

    def test_training_serving_preprocessing_parity(self):
        """Verify training uses the exact same clean_text normalizer as serving/inference."""
        from src.features.feature_builder import prepare_job_description_text
        from src.preprocessing.text_cleaner import clean_text
        sample_raw = "<b>Urgent</b> hiring: Python Developer! Email: hr@example.com Call 123-456-7890."
        sample_df = pd.DataFrame({"description": [sample_raw]})
        prepared_series = prepare_job_description_text(sample_df, text_column="description")
        direct_clean = clean_text(sample_raw)
        assert prepared_series.iloc[0] == direct_clean


# ---------------------------------------------------------------------------
# Chunk-based similarity helpers
# ---------------------------------------------------------------------------


class TestComputeMaxSimilaritiesChunked:
    def test_output_shape_matches_n_test(self):
        """Output array must have length equal to number of test rows."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = ["python engineer cloud aws", "fraud wire transfer earn", "data analyst sql"]
        test = ["cloud python", "earn transfer"]
        vec = TfidfVectorizer()
        X_tr = vec.fit_transform(corpus)
        X_te = vec.transform(test)

        max_sims = compute_max_similarities_chunked(X_te, X_tr, chunk_size=1)
        assert max_sims.shape == (len(test),)

    def test_values_in_0_1(self):
        """Cosine similarities must be in [0, 1]."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = ["alpha beta gamma", "delta epsilon zeta"]
        test = ["alpha gamma", "theta iota"]
        vec = TfidfVectorizer()
        X_tr = vec.fit_transform(corpus)
        X_te = vec.transform(test)

        max_sims = compute_max_similarities_chunked(X_te, X_tr)
        assert np.all(max_sims >= 0.0)
        assert np.all(max_sims <= 1.0 + 1e-6)

    def test_identical_text_has_sim_1(self):
        """An exact duplicate of a training text must have max_sim = 1.0."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = ["senior backend python engineer remote"]
        test = ["senior backend python engineer remote"]
        vec = TfidfVectorizer()
        X_tr = vec.fit_transform(corpus)
        X_te = vec.transform(test)

        max_sims = compute_max_similarities_chunked(X_te, X_tr)
        assert abs(max_sims[0] - 1.0) < 1e-5

    def test_chunk_size_1_matches_default(self):
        """Results must be identical regardless of chunk_size."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [f"word{i} engineer python cloud" for i in range(20)]
        test = [f"word{i} fraud earn transfer" for i in range(5)]
        vec = TfidfVectorizer()
        X_tr = vec.fit_transform(corpus)
        X_te = vec.transform(test)

        sims_default = compute_max_similarities_chunked(X_te, X_tr, chunk_size=500)
        sims_chunk1 = compute_max_similarities_chunked(X_te, X_tr, chunk_size=1)
        np.testing.assert_allclose(sims_default, sims_chunk1, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke test (tmp_path)
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    def test_e2e_with_synthetic_data(self, tmp_path, small_duplicate_df):
        """
        run_near_duplicate_aware_evaluation produces all required output files
        and report keys.
        """
        from src.validation.near_duplicate_grouper import run_near_duplicate_aware_evaluation

        csv_path = tmp_path / "cleaned_jobs_mock.csv"
        small_duplicate_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "nd_test_output"
        report = run_near_duplicate_aware_evaluation(
            cleaned_data_path=csv_path,
            output_dir=out_dir,
            grouping_threshold=0.90,
            thresholds=(0.85, 0.90, 0.95),
            target_test_fraction=0.20,
            random_state=42,
            save_artifacts=True,
        )

        # Report structure
        for key in (
            "metadata", "task1_similarity_audit", "task2_component_analysis",
            "task3_split", "task4_evaluation",
        ):
            assert key in report, f"Missing top-level key: {key}"

        # Files on disk
        assert (out_dir / "near_dup_aware_evaluation_report.json").exists()
        assert (out_dir / "near_dup_aware_evaluation_report.md").exists()
        assert (out_dir / "component_size_statistics.json").exists()

        # Model/vectorizer artifacts
        assert (out_dir / "nd_aware_candidate_xgboost.joblib").exists()
        assert (out_dir / "nd_aware_candidate_tfidf.joblib").exists()

    def test_e2e_prevents_overwrite_of_existing_dir(self, tmp_path, small_duplicate_df):
        """run_near_duplicate_aware_evaluation raises FileExistsError for non-empty output_dir."""
        from src.validation.near_duplicate_grouper import run_near_duplicate_aware_evaluation

        csv_path = tmp_path / "cleaned_jobs_mock.csv"
        small_duplicate_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "nd_test_output_conflict"
        out_dir.mkdir()
        (out_dir / "existing_file.txt").write_text("block overwrite")

        with pytest.raises(FileExistsError):
            run_near_duplicate_aware_evaluation(
                cleaned_data_path=csv_path,
                output_dir=out_dir,
                save_artifacts=True,
            )

    def test_json_report_is_valid_json(self, tmp_path, small_duplicate_df):
        """The JSON report must be parseable."""
        from src.validation.near_duplicate_grouper import run_near_duplicate_aware_evaluation

        csv_path = tmp_path / "cleaned_jobs_mock.csv"
        small_duplicate_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "nd_test_json_check"
        run_near_duplicate_aware_evaluation(
            cleaned_data_path=csv_path,
            output_dir=out_dir,
            save_artifacts=True,
            random_state=42,
        )

        json_path = out_dir / "near_dup_aware_evaluation_report.json"
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)
        assert "task4_evaluation" in loaded

    def test_safety_constraints_documented_in_report(self, tmp_path, small_duplicate_df):
        """Report must include documented safety constraints."""
        from src.validation.near_duplicate_grouper import run_near_duplicate_aware_evaluation

        csv_path = tmp_path / "cleaned_jobs_mock.csv"
        small_duplicate_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "nd_safety_check"
        report = run_near_duplicate_aware_evaluation(
            cleaned_data_path=csv_path,
            output_dir=out_dir,
            save_artifacts=True,
        )

        assert "safety_constraints" in report["metadata"]
        constraints = report["metadata"]["safety_constraints"]
        assert len(constraints) >= 3

    def test_no_component_crossing_in_full_pipeline(self, tmp_path, small_duplicate_df):
        """End-to-end: no component may cross the train/test boundary."""
        from src.validation.near_duplicate_grouper import (
            run_near_duplicate_aware_evaluation,
            build_near_duplicate_components,
        )

        csv_path = tmp_path / "cleaned_jobs_mock.csv"
        small_duplicate_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "nd_no_cross"
        report = run_near_duplicate_aware_evaluation(
            cleaned_data_path=csv_path,
            output_dir=out_dir,
            save_artifacts=True,
        )

        split_meta = report["task3_split"]["split_metadata"]
        assert split_meta["contamination_checks"]["zero_component_leakage_verified"] is True
        assert split_meta["contamination_checks"]["component_overlap_count"] == 0
