"""
Decision-threshold optimization module for JobClarity.

Generates out-of-fold (OOF) fraud probabilities using only the training partition.
TF-IDF and the XGBoost model are fitted independently inside each CV fold so that
no validation-set information ever leaks into the vectorizer vocabulary or model weights.

Design principles
-----------------
- The held-out 20 % test set is never touched here.
- No single threshold is silently labelled "recommended production threshold".
  The function returns the full trade-off table and the set of Pareto-optimal
  operating points.  If the caller supplies an explicit, named selection
  criterion, the function additionally computes the single threshold that best
  satisfies that criterion and records the rationale in the report.
- Selection criteria are configurable via ``ThresholdSelectionCriterion``.
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from xgboost import XGBClassifier

from src.features.feature_builder import prepare_job_description_text
from src.features.tfidf_vectorizer import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_text,
)
from src.models.evaluate import load_dataset
from src.models.hyperparameter_tuner import (
    CandidateConfig,
    compute_fold_scale_pos_weight,
)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Configurable selection criterion
# ---------------------------------------------------------------------------

class ThresholdSelectionCriterion(str, Enum):
    """
    Named, explicitly-documented criteria for selecting a single operating threshold.

    Pass ``criterion=None`` (the default) to skip threshold selection entirely
    and return only the full trade-off table and Pareto-optimal points.

    If a criterion IS supplied, the selected threshold and its rationale are
    recorded in the report under ``selected_threshold``.  No selection is
    made silently — the criterion name and its derivation are always surfaced.

    Attributes
    ----------
    MAX_F1 :
        Highest fraud F1 across all thresholds.
    MAX_PRECISION_AT_RECALL_50 :
        Highest fraud precision among thresholds where fraud recall >= 0.50.
        Useful when missing half the fraud is acceptable but false alarms are costly.
    MAX_PRECISION_AT_RECALL_60 :
        Highest fraud precision among thresholds where fraud recall >= 0.60.
    MAX_PRECISION_AT_RECALL_70 :
        Highest fraud precision among thresholds where fraud recall >= 0.70.
    MAX_PRECISION_AT_RECALL_80 :
        Highest fraud precision among thresholds where fraud recall >= 0.80.
    MAX_F1_AT_RECALL_50 :
        Highest fraud F1 among thresholds where fraud recall >= 0.50.
    MAX_F1_AT_RECALL_60 :
        Highest fraud F1 among thresholds where fraud recall >= 0.60.
    MAX_F1_AT_RECALL_70 :
        Highest fraud F1 among thresholds where fraud recall >= 0.70.
    MAX_F1_AT_RECALL_80 :
        Highest fraud F1 among thresholds where fraud recall >= 0.80.
    """

    MAX_F1 = "max_f1"
    MAX_PRECISION_AT_RECALL_50 = "max_precision_at_recall_50"
    MAX_PRECISION_AT_RECALL_60 = "max_precision_at_recall_60"
    MAX_PRECISION_AT_RECALL_70 = "max_precision_at_recall_70"
    MAX_PRECISION_AT_RECALL_80 = "max_precision_at_recall_80"
    MAX_F1_AT_RECALL_50 = "max_f1_at_recall_50"
    MAX_F1_AT_RECALL_60 = "max_f1_at_recall_60"
    MAX_F1_AT_RECALL_70 = "max_f1_at_recall_70"
    MAX_F1_AT_RECALL_80 = "max_f1_at_recall_80"


# ---------------------------------------------------------------------------
# OOF probability generation
# ---------------------------------------------------------------------------

def generate_oof_probabilities(
    X_train_text: pd.Series,
    y_train: np.ndarray,
    candidate: CandidateConfig,
    n_splits: int = 5,
    random_state: int = 42,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold fraud probabilities for *candidate* using only training data.

    For every fold:
      1. TF-IDF is fitted exclusively on that fold's training text.
      2. The model is trained on that fold's training features.
      3. Fraud probabilities are collected for the held-out validation portion.

    No information from any validation fold ever flows into another fold's
    vectorizer vocabulary or model parameters.

    Parameters
    ----------
    X_train_text : pd.Series
        Cleaned job-description text for all training samples.
    y_train : np.ndarray
        Binary labels (0 = legitimate, 1 = fraudulent) for training samples.
    candidate : CandidateConfig
        XGBoost hyperparameter configuration to use.
    n_splits : int, default=5
        Number of stratified CV folds.
    random_state : int, default=42
        Seed for StratifiedKFold and XGBoost.
    max_features : int, default=10_000
        TF-IDF maximum vocabulary size.
    ngram_range : tuple, default=(1, 2)
        TF-IDF n-gram range.

    Returns
    -------
    oof_probs : np.ndarray, shape (n_train_samples,)
        Fraud probability for every training sample, collected from the fold
        in which that sample was used for validation.
    oof_true : np.ndarray, shape (n_train_samples,)
        Ground-truth labels in the same order as *oof_probs*.
    """
    X_text = (
        X_train_text.reset_index(drop=True)
        if isinstance(X_train_text, pd.Series)
        else pd.Series(X_train_text).reset_index(drop=True)
    )
    y_arr = np.asarray(y_train)
    n = len(y_arr)

    oof_probs = np.full(n, np.nan)
    oof_true = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(
        f"[OOF] Generating OOF probabilities: {n_splits}-fold Stratified CV "
        f"| candidate={candidate.candidate_id}"
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_text, y_arr)):
        fold_num = fold_idx + 1
        fold_X_train_raw = X_text.iloc[train_idx]
        fold_y_train = y_arr[train_idx]
        fold_X_val_raw = X_text.iloc[val_idx]
        fold_y_val = y_arr[val_idx]

        # Fit TF-IDF strictly on this fold's training text
        fold_vec = create_tfidf_vectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        fold_vec, X_fold_train = fit_tfidf_vectorizer(fold_vec, fold_X_train_raw)
        X_fold_val = transform_text(fold_vec, fold_X_val_raw)

        # Resolve class weight from fold training labels only
        if candidate.scale_pos_weight == "balanced":
            resolved_weight = compute_fold_scale_pos_weight(fold_y_train)
        else:
            resolved_weight = float(candidate.scale_pos_weight)

        # Fit model on fold training features
        model = XGBClassifier(
            n_estimators=candidate.n_estimators,
            max_depth=candidate.max_depth,
            learning_rate=candidate.learning_rate,
            subsample=candidate.subsample,
            colsample_bytree=candidate.colsample_bytree,
            scale_pos_weight=resolved_weight,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_fold_train, fold_y_train)

        # Collect OOF probabilities for validation fold
        fold_probs = model.predict_proba(X_fold_val)[:, 1]
        oof_probs[val_idx] = fold_probs
        oof_true[val_idx] = fold_y_val

        n_fraud_val = int((fold_y_val == 1).sum())
        fold_pr_auc = float(average_precision_score(fold_y_val, fold_probs))
        print(
            f"  [Fold {fold_num}/{n_splits}] "
            f"val samples: {len(val_idx)} (fraud: {n_fraud_val}) | "
            f"PR-AUC: {fold_pr_auc:.4f}"
        )

    assert not np.isnan(oof_probs).any(), "Some OOF slots were not filled — check CV loop."
    return oof_probs, oof_true.astype(int)


# ---------------------------------------------------------------------------
# Per-threshold metric computation
# ---------------------------------------------------------------------------

def evaluate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Compute classification metrics for a given decision threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted fraud probabilities.
    threshold : float
        Decision boundary; samples with prob >= threshold are predicted fraud.

    Returns
    -------
    dict
        Metrics including precision, recall, F1, FP, FN, TP, TN, confusion matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    return {
        "threshold": round(float(threshold), 4),
        "fraud_precision": round(prec, 4),
        "fraud_recall": round(rec, 4),
        "fraud_f1": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Pareto-optimal operating points
# ---------------------------------------------------------------------------

def compute_pareto_optimal_points(
    threshold_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Identify the Pareto-optimal operating points on the precision-recall plane.

    A point is Pareto-optimal if no other threshold simultaneously dominates it
    on both fraud precision and fraud recall.  These points form the frontier of
    achievable precision-recall trade-offs and are the natural starting point for
    production threshold selection.

    Parameters
    ----------
    threshold_results : list of dict
        Output of ``evaluate_threshold`` for each threshold in the grid.

    Returns
    -------
    list of dict
        Subset of *threshold_results* that are Pareto-optimal, sorted by
        ascending fraud recall (equivalently, descending threshold).
    """
    pareto: list[dict[str, Any]] = []

    for candidate in threshold_results:
        cp = candidate["fraud_precision"]
        cr = candidate["fraud_recall"]

        # A candidate is dominated if there exists another point with
        # precision >= cp AND recall >= cr (with at least one strict).
        dominated = False
        for other in threshold_results:
            if other is candidate:
                continue
            op = other["fraud_precision"]
            or_ = other["fraud_recall"]
            if op >= cp and or_ >= cr and (op > cp or or_ > cr):
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    pareto.sort(key=lambda r: r["fraud_recall"])
    return pareto


# ---------------------------------------------------------------------------
# Named criterion application
# ---------------------------------------------------------------------------

def _apply_criterion(
    threshold_results: list[dict[str, Any]],
    criterion: ThresholdSelectionCriterion,
) -> dict[str, Any]:
    """
    Apply a named selection criterion to the threshold grid.

    Returns a dict with keys: ``threshold_row``, ``criterion``, ``rationale``.
    Raises ``ValueError`` if no threshold satisfies the criterion's constraints.
    """
    recall_floors = {
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_50: 0.50,
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_60: 0.60,
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_70: 0.70,
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_80: 0.80,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_50: 0.50,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_60: 0.60,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_70: 0.70,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_80: 0.80,
    }

    if criterion == ThresholdSelectionCriterion.MAX_F1:
        best = max(threshold_results, key=lambda r: (r["fraud_f1"], r["fraud_precision"]))
        rationale = (
            f"Criterion '{criterion.value}': threshold {best['threshold']} maximises "
            f"fraud F1 ({best['fraud_f1']:.4f}) across the full grid."
        )

    elif criterion in (
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_50,
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_60,
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_70,
        ThresholdSelectionCriterion.MAX_PRECISION_AT_RECALL_80,
    ):
        floor = recall_floors[criterion]
        eligible = [r for r in threshold_results if r["fraud_recall"] >= floor]
        if not eligible:
            raise ValueError(
                f"Criterion '{criterion.value}': no threshold achieves recall >= {floor}. "
                "Consider a lower recall floor or inspecting the PR curve."
            )
        best = max(eligible, key=lambda r: (r["fraud_precision"], r["fraud_f1"]))
        rationale = (
            f"Criterion '{criterion.value}': threshold {best['threshold']} maximises "
            f"fraud precision ({best['fraud_precision']:.4f}) among thresholds "
            f"where recall >= {floor:.2f} (fraud recall = {best['fraud_recall']:.4f}, "
            f"F1 = {best['fraud_f1']:.4f})."
        )

    elif criterion in (
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_50,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_60,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_70,
        ThresholdSelectionCriterion.MAX_F1_AT_RECALL_80,
    ):
        floor = recall_floors[criterion]
        eligible = [r for r in threshold_results if r["fraud_recall"] >= floor]
        if not eligible:
            raise ValueError(
                f"Criterion '{criterion.value}': no threshold achieves recall >= {floor}. "
                "Consider a lower recall floor or inspecting the PR curve."
            )
        best = max(eligible, key=lambda r: (r["fraud_f1"], r["fraud_precision"]))
        rationale = (
            f"Criterion '{criterion.value}': threshold {best['threshold']} maximises "
            f"fraud F1 ({best['fraud_f1']:.4f}) among thresholds "
            f"where recall >= {floor:.2f} (fraud precision = {best['fraud_precision']:.4f}, "
            f"fraud recall = {best['fraud_recall']:.4f})."
        )

    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")

    return {
        "threshold_row": best,
        "criterion": criterion.value,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Main optimization pipeline
# ---------------------------------------------------------------------------

def optimize_decision_threshold(
    data_path: Path | str = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    output_dir: Path | str = PROJECT_ROOT / "artifacts" / "evaluation" / "day2" / "threshold_optimization",
    test_size: float = 0.20,
    n_splits: int = 5,
    random_state: int = 42,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    text_column: str = "description",
    threshold_grid: list[float] | None = None,
    candidate: CandidateConfig | None = None,
    criterion: ThresholdSelectionCriterion | str | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Run OOF-based decision-threshold analysis on the training partition.

    Workflow
    --------
    1. Reproducibly split dataset into the same 80/20 train/test partition used
       in Sessions 3 and 4.  The 20 % test set is isolated and **never touched**.
    2. Generate OOF fraud probabilities using ``generate_oof_probabilities`` —
       TF-IDF and XGBoost are refitted inside every fold; no leakage occurs.
    3. Sweep a threshold grid and record precision, recall, F1, TP, FP, FN, TN,
       and the full 2×2 confusion matrix for every threshold.
    4. Compute the precision-recall curve and overall OOF PR-AUC.
    5. Identify **Pareto-optimal operating points** on the precision-recall plane
       (points not dominated on both metrics simultaneously).
    6. If *criterion* is supplied, compute the single threshold that best
       satisfies that named criterion and record it with an explicit rationale
       under ``selected_threshold`` in the report.  No threshold is silently
       chosen when *criterion* is ``None`` (the default).

    Threshold selection
    -------------------
    No threshold is promoted to "production" inside this function.  The full
    trade-off table and Pareto frontier are always returned.  Callers who need
    a single operating point should pass one of the named
    ``ThresholdSelectionCriterion`` values and justify the choice in their
    experiment notes.

    The held-out 20 % test set is **not evaluated** here.

    Parameters
    ----------
    data_path : Path or str
    output_dir : Path or str
    test_size : float, default=0.20
    n_splits : int, default=5
    random_state : int, default=42
    max_features : int, default=10_000
    ngram_range : tuple, default=(1, 2)
    text_column : str, default='description'
    threshold_grid : list of float, optional
        Custom threshold grid.  Defaults to 0.05 … 0.95 in steps of 0.05.
    candidate : CandidateConfig, optional
        Defaults to Session-3 best (cand_0_baseline, unweighted).
    criterion : ThresholdSelectionCriterion or str or None, default=None
        Named selection criterion.  ``None`` means no single threshold is
        selected; only the full table and Pareto frontier are returned.
    save_artifacts : bool, default=True

    Returns
    -------
    dict with keys:
        ``report``            – full JSON-serialisable report dict
        ``oof_probs``         – np.ndarray of OOF fraud probabilities
        ``oof_true``          – np.ndarray of matching ground-truth labels
        ``threshold_results`` – list of per-threshold metric dicts
        ``pareto_points``     – Pareto-optimal subset of threshold_results
        ``selected_threshold``– dict (if criterion supplied) or None
        ``artifacts_saved``   – dict of written file paths
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------ guards
    prod_models_dir = (PROJECT_ROOT / "models").resolve()
    candidate_dir = (PROJECT_ROOT / "artifacts" / "candidate").resolve()

    if output_dir.resolve() == prod_models_dir:
        raise ValueError(
            f"Refusing to write threshold optimization report into production models "
            f"directory: {prod_models_dir}. Use a separate output directory."
        )
    if output_dir.resolve() == candidate_dir:
        raise ValueError(
            f"Refusing to write into baseline candidate directory: {candidate_dir}. "
            "Use a separate output directory such as artifacts/evaluation/day2/threshold_optimization/."
        )

    # Normalise criterion
    if criterion is not None and not isinstance(criterion, ThresholdSelectionCriterion):
        criterion = ThresholdSelectionCriterion(criterion)

    print("=" * 75)
    print("JobClarity — Session 5: Decision-Threshold Analysis")
    print("=" * 75)

    # ---------------------------------------------------------------- defaults
    if threshold_grid is None:
        grid = sorted(set(round(t, 2) for t in np.arange(0.05, 1.00, 0.05)))
    else:
        grid = sorted(set(round(float(t), 4) for t in threshold_grid))

    if candidate is None:
        candidate = CandidateConfig(
            candidate_id="cand_0_baseline",
            name="Baseline Default (Unweighted)",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            scale_pos_weight=1.0,
        )

    # ------------------------------------------------------------------- data
    print(f"[1/4] Loading dataset: {data_path}")
    df = load_dataset(data_path)
    if "fraudulent" not in df.columns:
        raise KeyError("Target column 'fraudulent' not found in dataset.")

    y_all = df["fraudulent"].astype(int)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    y_train = train_df["fraudulent"].astype(int).to_numpy()
    y_test_info = {
        "samples": len(test_df),
        "legitimate_0": int((test_df["fraudulent"] == 0).sum()),
        "fraudulent_1": int((test_df["fraudulent"] == 1).sum()),
        "note": (
            "Test set isolated — not evaluated in this session. "
            "The Session 4 held-out test evaluation is complete and must not be repeated here."
        ),
    }

    X_train_text = prepare_job_description_text(train_df, text_column=text_column)

    train_fraud = int((y_train == 1).sum())
    train_legit = int((y_train == 0).sum())
    print(
        f"      Train: {len(y_train)} samples "
        f"(legit: {train_legit}, fraud: {train_fraud})"
    )
    print(f"      Test:  {len(test_df)} samples [ISOLATED — not evaluated here]")

    # --------------------------------------------------------- OOF probabilities
    print(f"\n[2/4] Generating OOF probabilities ({n_splits}-fold stratified CV)...")
    t0 = time.time()
    oof_probs, oof_true = generate_oof_probabilities(
        X_train_text=X_train_text,
        y_train=y_train,
        candidate=candidate,
        n_splits=n_splits,
        random_state=random_state,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    oof_duration = time.time() - t0

    oof_pr_auc = float(average_precision_score(oof_true, oof_probs))
    print(f"      OOF PR-AUC: {oof_pr_auc:.4f}  (duration: {oof_duration:.1f}s)")

    # --------------------------------------------------- precision-recall curve
    pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(oof_true, oof_probs)
    pr_curve_data = {
        "precisions": [round(float(p), 4) for p in pr_precisions.tolist()],
        "recalls": [round(float(r), 4) for r in pr_recalls.tolist()],
        "thresholds": [round(float(t), 4) for t in pr_thresholds.tolist()],
    }

    # ------------------------------------------------------- threshold sweep
    print(f"\n[3/4] Sweeping threshold grid ({len(grid)} thresholds)...")
    print(
        f"  {'thr':>5s}  {'precision':>9s}  {'recall':>6s}  {'f1':>6s}  "
        f"{'TP':>4s}  {'FP':>5s}  {'FN':>4s}  {'TN':>5s}"
    )
    print("  " + "-" * 58)

    threshold_results: list[dict[str, Any]] = []
    for thr in grid:
        row = evaluate_threshold(oof_true, oof_probs, thr)
        threshold_results.append(row)
        print(
            f"  {thr:>5.2f}  {row['fraud_precision']:>9.4f}  "
            f"{row['fraud_recall']:>6.4f}  {row['fraud_f1']:>6.4f}  "
            f"{row['true_positives']:>4d}  {row['false_positives']:>5d}  "
            f"{row['false_negatives']:>4d}  {row['true_negatives']:>5d}"
        )

    # ----------------------------------------------------- Pareto frontier
    pareto_points = compute_pareto_optimal_points(threshold_results)
    print(f"\n  Pareto-optimal operating points ({len(pareto_points)}):")
    print(
        f"  {'thr':>5s}  {'precision':>9s}  {'recall':>6s}  {'f1':>6s}"
    )
    print("  " + "-" * 35)
    for pt in pareto_points:
        print(
            f"  {pt['threshold']:>5.2f}  {pt['fraud_precision']:>9.4f}  "
            f"{pt['fraud_recall']:>6.4f}  {pt['fraud_f1']:>6.4f}"
        )

    # ----------------------------------------------------- optional selection
    selected_threshold: dict[str, Any] | None = None
    if criterion is not None:
        sel = _apply_criterion(threshold_results, criterion)
        selected_threshold = {
            "threshold": sel["threshold_row"]["threshold"],
            "criterion": sel["criterion"],
            "rationale": sel["rationale"],
            "metrics": {
                k: sel["threshold_row"][k]
                for k in (
                    "fraud_precision", "fraud_recall", "fraud_f1",
                    "true_positives", "false_positives",
                    "false_negatives", "true_negatives", "confusion_matrix",
                )
            },
            "warning": (
                "This threshold was selected using training-partition OOF predictions only. "
                "It has not been evaluated on the held-out test set and must not be treated "
                "as a production-validated threshold without an additional independent evaluation."
            ),
        }
        print(f"\n  Selected threshold (criterion='{criterion.value}'): "
              f"{selected_threshold['threshold']}")
        print(f"  {sel['rationale']}")
    else:
        print(
            "\n  No selection criterion supplied. "
            "See 'pareto_optimal_points' in the report for operating-point candidates."
        )

    # -------------------------------------------------------------- report
    report: dict[str, Any] = {
        "metadata": {
            "pipeline": "day2_session5_threshold_analysis",
            "session": "Day 2 Session 5",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_path": str(data_path),
            "text_representation": {
                "source_column": text_column,
                "representation": "job_description_only",
                "cleaner": "src.preprocessing.text_cleaner.clean_text",
            },
            "split_parameters": {
                "test_size": test_size,
                "random_state": random_state,
                "stratified": True,
                "note": (
                    "Identical split to Sessions 2–4. "
                    "Test set is isolated and NOT evaluated here."
                ),
            },
            "oof_cv_strategy": {
                "method": "StratifiedKFold",
                "n_splits": n_splits,
                "shuffle": True,
                "random_state": random_state,
                "leakage_free": True,
                "note": (
                    "TF-IDF and XGBoost are fitted independently inside each fold. "
                    "No validation-fold information influences the vectorizer or model."
                ),
            },
            "candidate": candidate.to_dict(),
            "threshold_selection_policy": {
                "approach": "full_tradeoff_table_and_pareto_frontier",
                "criterion_supplied": criterion.value if criterion else None,
                "note": (
                    "No threshold is silently labelled as 'production threshold'. "
                    "The full trade-off table and Pareto-optimal points are always "
                    "returned. A single threshold is computed only when an explicit, "
                    "named ThresholdSelectionCriterion is supplied by the caller."
                ),
            },
            "experimental_constraints": {
                "test_set_status": (
                    "The Session 4 held-out test set has already been evaluated. "
                    "It is NOT used here for threshold selection or model tuning."
                ),
                "production_artifacts": "Not modified.",
            },
            "held_out_test_set": y_test_info,
        },
        "dataset_info": {
            "total_records": len(df),
            "train_samples": len(y_train),
            "test_samples": len(test_df),
            "train_distribution": {
                "legitimate_0": train_legit,
                "fraudulent_1": train_fraud,
                "fraudulent_ratio": round(float(train_fraud / len(y_train)), 4),
            },
        },
        "oof_summary": {
            "n_splits": n_splits,
            "oof_pr_auc": round(oof_pr_auc, 4),
            "oof_duration_seconds": round(oof_duration, 2),
        },
        "threshold_grid_results": threshold_results,
        "pr_curve": pr_curve_data,
        "pareto_optimal_points": pareto_points,
        "selected_threshold": selected_threshold,
    }

    # ---------------------------------------------------------- save artifacts
    artifacts_saved: dict[str, str] = {}
    if save_artifacts:
        print(f"\n[4/4] Saving artifacts to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "threshold_analysis_report.json"
        table_path = output_dir / "threshold_grid_summary.csv"
        pareto_path = output_dir / "pareto_optimal_points.csv"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Full grid CSV
        table_rows = [
            {
                "threshold": r["threshold"],
                "fraud_precision": r["fraud_precision"],
                "fraud_recall": r["fraud_recall"],
                "fraud_f1": r["fraud_f1"],
                "true_positives": r["true_positives"],
                "false_positives": r["false_positives"],
                "false_negatives": r["false_negatives"],
                "true_negatives": r["true_negatives"],
                "pareto_optimal": r in pareto_points,
            }
            for r in threshold_results
        ]
        pd.DataFrame(table_rows).to_csv(table_path, index=False)

        # Pareto-only CSV
        pareto_rows = [
            {
                "threshold": r["threshold"],
                "fraud_precision": r["fraud_precision"],
                "fraud_recall": r["fraud_recall"],
                "fraud_f1": r["fraud_f1"],
                "true_positives": r["true_positives"],
                "false_positives": r["false_positives"],
                "false_negatives": r["false_negatives"],
                "true_negatives": r["true_negatives"],
            }
            for r in pareto_points
        ]
        pd.DataFrame(pareto_rows).to_csv(pareto_path, index=False)

        artifacts_saved = {
            "report": str(report_path),
            "grid_summary_csv": str(table_path),
            "pareto_optimal_csv": str(pareto_path),
        }
        print(f"  Saved: {report_path.name}")
        print(f"  Saved: {table_path.name}")
        print(f"  Saved: {pareto_path.name}")

    print("=" * 75)
    print(
        f"Threshold Analysis Complete — "
        f"{len(pareto_points)} Pareto-optimal operating points identified."
    )
    print("=" * 75)

    return {
        "report": report,
        "oof_probs": oof_probs,
        "oof_true": oof_true,
        "threshold_results": threshold_results,
        "pareto_points": pareto_points,
        "selected_threshold": selected_threshold,
        "artifacts_saved": artifacts_saved,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Session 5: Decision-threshold analysis using OOF predictions. "
            "Reports the full precision-recall trade-off and Pareto-optimal "
            "operating points. No single threshold is selected unless "
            "--criterion is supplied."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation" / "day2" / "threshold_optimization",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=10_000)
    parser.add_argument("--text-column", type=str, default="description")
    parser.add_argument(
        "--criterion",
        type=str,
        default=None,
        choices=[c.value for c in ThresholdSelectionCriterion],
        help=(
            "Optional named selection criterion. When supplied, the threshold "
            "that best satisfies the criterion is recorded in the report under "
            "'selected_threshold'. Omit to return only the full trade-off table "
            "and Pareto-optimal points."
        ),
    )

    args = parser.parse_args()
    optimize_decision_threshold(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_features=args.max_features,
        text_column=args.text_column,
        criterion=args.criterion,
    )


if __name__ == "__main__":
    main()
