"""
Hyperparameter tuning module for JobClarity using leakage-free stratified cross-validation.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    target_column: str = "fraudulent",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset reproducibly using stratified train_test_split.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' missing from DataFrame.")

    y = df[target_column]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return train_df, test_df


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CandidateConfig:
    """Configuration for an XGBoost tuning candidate."""

    candidate_id: str
    name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    scale_pos_weight: float | str  # numeric float (e.g. 1.0) or 'balanced'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CANDIDATES: list[CandidateConfig] = [
    CandidateConfig(
        candidate_id="cand_0_baseline",
        name="Baseline Default (Unweighted)",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
    ),
    CandidateConfig(
        candidate_id="cand_1_baseline_weighted",
        name="Baseline Default (Class-Weighted)",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight="balanced",
    ),
    CandidateConfig(
        candidate_id="cand_2_shallow_subsampled",
        name="Shallow Subsampled (Unweighted)",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
    ),
    CandidateConfig(
        candidate_id="cand_3_shallow_subsampled_weighted",
        name="Shallow Subsampled (Class-Weighted)",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight="balanced",
    ),
    CandidateConfig(
        candidate_id="cand_4_medium_depth_conservative_weighted",
        name="Medium Depth Regularized (Class-Weighted)",
        n_estimators=120,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight="balanced",
    ),
]


def compute_fold_scale_pos_weight(y_train_fold: np.ndarray | pd.Series) -> float:
    """
    Compute fold-specific scale_pos_weight as negative_samples / positive_samples.
    Strictly isolated to the provided training fold labels.

    Parameters
    ----------
    y_train_fold : array-like
        Binary labels of the training fold.

    Returns
    -------
    float
        Class weight ratio.
    """
    y_arr = np.asarray(y_train_fold)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)


def evaluate_candidate_on_fold(
    candidate: CandidateConfig,
    X_train_fold,
    y_train_fold: np.ndarray,
    X_val_fold,
    y_val_fold: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fit and evaluate a candidate model on a pre-transformed fold.

    Parameters
    ----------
    candidate : CandidateConfig
        Candidate hyperparameter configuration.
    X_train_fold : scipy.sparse matrix
        TF-IDF features fitted solely on training fold.
    y_train_fold : np.ndarray
        Labels for training fold.
    X_val_fold : scipy.sparse matrix
        TF-IDF features transformed for validation fold.
    y_val_fold : np.ndarray
        Labels for validation fold.
    random_state : int, default=42
        Seed for XGBoost.

    Returns
    -------
    dict
        Evaluation metrics on the validation fold.
    """
    # Resolve class weight for this fold
    if candidate.scale_pos_weight == "balanced":
        resolved_weight = compute_fold_scale_pos_weight(y_train_fold)
    else:
        resolved_weight = float(candidate.scale_pos_weight)

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

    model.fit(X_train_fold, y_train_fold)

    val_probs = model.predict_proba(X_val_fold)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_val_fold, val_probs)),
        "roc_auc": float(roc_auc_score(y_val_fold, val_probs)),
        "fraud_precision": float(
            precision_score(y_val_fold, val_preds, pos_label=1, zero_division=0)
        ),
        "fraud_recall": float(
            recall_score(y_val_fold, val_preds, pos_label=1, zero_division=0)
        ),
        "fraud_f1": float(
            f1_score(y_val_fold, val_preds, pos_label=1, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_val_fold, val_preds)),
        "resolved_scale_pos_weight": round(resolved_weight, 4),
    }

    return metrics


def run_stratified_cv_tuning(
    X_train_text: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
    candidates: list[CandidateConfig] | None = None,
    n_splits: int = 3,
    random_state: int = 42,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[dict[str, Any]]:
    """
    Execute fold-safe stratified cross-validation across all tuning candidates.

    Guarantees:
    1. StratifiedKFold applied strictly to the training split.
    2. Within each fold, TF-IDF vectorizer is fitted strictly on that fold's training text.
    3. Validation text is transformed using the fitted fold vectorizer only.
    4. Fold scale_pos_weight is computed strictly from that fold's training labels.
    5. Evaluates PR-AUC (Average Precision), ROC-AUC, F1, Precision, Recall, and Accuracy.

    Parameters
    ----------
    X_train_text : pd.Series or list of str
        Cleaned training text descriptions.
    y_train : pd.Series or np.ndarray
        Training labels.
    candidates : list of CandidateConfig, optional
        Candidates to evaluate. Defaults to DEFAULT_CANDIDATES.
    n_splits : int, default=3
        Number of stratified CV folds.
    random_state : int, default=42
        Seed for CV split and models.
    max_features : int, default=10000
        TF-IDF max features.
    ngram_range : tuple, default=(1, 2)
        TF-IDF n-gram range.

    Returns
    -------
    list of dict
        Candidate evaluation results ranked by mean PR-AUC.
    """
    candidates = candidates or DEFAULT_CANDIDATES
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X_text_series = (
        pd.Series(X_train_text).reset_index(drop=True)
        if not isinstance(X_train_text, pd.Series)
        else X_train_text.reset_index(drop=True)
    )
    y_arr = np.asarray(y_train)

    # Initialize results container for each candidate
    candidate_records: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        candidate_records[cand.candidate_id] = {
            "candidate_id": cand.candidate_id,
            "name": cand.name,
            "config": cand.to_dict(),
            "fold_results": [],
            "total_duration_seconds": 0.0,
        }

    print(
        f"[CV] Starting {n_splits}-fold Stratified CV for {len(candidates)} candidates..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_text_series, y_arr)):
        fold_num = fold_idx + 1
        print(
            f"  [Fold {fold_num}/{n_splits}] Train samples: {len(train_idx)}, Val samples: {len(val_idx)}"
        )

        # 1. Fold texts and labels
        fold_X_train_raw = X_text_series.iloc[train_idx]
        fold_y_train = y_arr[train_idx]
        fold_X_val_raw = X_text_series.iloc[val_idx]
        fold_y_val = y_arr[val_idx]

        # 2. Fit TF-IDF strictly on training fold text
        fold_vec_start = time.time()
        fold_vec = create_tfidf_vectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        fold_vec, X_fold_train = fit_tfidf_vectorizer(fold_vec, fold_X_train_raw)

        # 3. Transform validation fold text
        X_fold_val = transform_text(fold_vec, fold_X_val_raw)
        fold_vec_time = time.time() - fold_vec_start

        # 4. Evaluate all candidates on this fold's representations
        for cand in candidates:
            cand_t0 = time.time()
            fold_metrics = evaluate_candidate_on_fold(
                candidate=cand,
                X_train_fold=X_fold_train,
                y_train_fold=fold_y_train,
                X_val_fold=X_fold_val,
                y_val_fold=fold_y_val,
                random_state=random_state,
            )
            cand_duration = time.time() - cand_t0

            fold_metrics["fold"] = fold_num
            fold_metrics["fit_duration_seconds"] = round(cand_duration, 3)
            candidate_records[cand.candidate_id]["fold_results"].append(fold_metrics)
            candidate_records[cand.candidate_id]["total_duration_seconds"] += (
                cand_duration + (fold_vec_time / len(candidates))
            )

            print(
                f"    -> {cand.candidate_id:25s} | PR-AUC: {fold_metrics['pr_auc']:.4f} | "
                f"F1: {fold_metrics['fraud_f1']:.4f} | Recall: {fold_metrics['fraud_recall']:.4f} "
                f"({cand_duration:.1f}s)"
            )

    # 5. Aggregate metrics across folds
    aggregated_results: list[dict[str, Any]] = []
    metric_keys = [
        "pr_auc",
        "roc_auc",
        "fraud_f1",
        "fraud_precision",
        "fraud_recall",
        "accuracy",
    ]

    for cand_id, rec in candidate_records.items():
        folds = rec["fold_results"]
        mean_metrics = {}
        std_metrics = {}

        for k in metric_keys:
            vals = [f[k] for f in folds]
            mean_metrics[f"mean_{k}"] = round(float(np.mean(vals)), 4)
            std_metrics[f"std_{k}"] = round(float(np.std(vals)), 4)

        summary_entry = {
            "candidate_id": cand_id,
            "name": rec["name"],
            "config": rec["config"],
            "fold_results": folds,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "cv_duration_seconds": round(rec["total_duration_seconds"], 2),
        }
        aggregated_results.append(summary_entry)

    # Rank by mean_pr_auc (descending), using mean_fraud_f1 as tiebreaker
    aggregated_results.sort(
        key=lambda x: (
            x["mean_metrics"]["mean_pr_auc"],
            x["mean_metrics"]["mean_fraud_f1"],
        ),
        reverse=True,
    )

    for rank, entry in enumerate(aggregated_results, 1):
        entry["rank"] = rank

    return aggregated_results


def tune_xgboost_pipeline(
    data_path: Path | str = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    output_dir: Path | str = PROJECT_ROOT / "artifacts" / "candidate" / "tuned",
    test_size: float = 0.20,
    n_splits: int = 3,
    random_state: int = 42,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    text_column: str = "description",
    candidates: list[CandidateConfig] | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Execute stratified cross-validation hyperparameter tuning and train the winning candidate.

    Guarantees:
    1. Uses the existing description-only text representation.
    2. The 20% held-out test split is preserved completely untouched.
    3. Stratified cross-validation is performed exclusively on the 80% training split.
    4. Evaluates a time-bounded grid of candidates exploring n_estimators, max_depth,
       learning_rate, subsample, colsample_bytree, and scale_pos_weight.
    5. Selects the optimal model based on mean cross-validation PR-AUC.
    6. Fit TF-IDF within each training fold without validation fold leakage.
    7. Computes class weights strictly from training fold labels.
    8. Refits the winning configuration on the full 80% training split.
    9. Saves tuned artifacts to a separate directory without overwriting production or baseline candidate artifacts.
    10. Does NOT evaluate on the final held-out test split.

    Parameters
    ----------
    data_path : Path or str
        Path to cleaned jobs dataset CSV.
    output_dir : Path or str
        Destination directory for tuned candidate artifacts.
    test_size : float, default=0.20
        Fraction reserved for untouched test set.
    n_splits : int, default=3
        Number of stratified CV folds.
    random_state : int, default=42
        Deterministic random seed.
    max_features : int, default=10000
        TF-IDF maximum features.
    ngram_range : tuple, default=(1, 2)
        TF-IDF n-gram range.
    text_column : str, default='description'
        Target text column name.
    candidates : list of CandidateConfig, optional
        Custom candidates list. Defaults to DEFAULT_CANDIDATES.
    save_artifacts : bool, default=True
        Whether to save tuned artifacts to disk.

    Returns
    -------
    dict
        Comprehensive tuning report and trained model metadata.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    prod_models_dir = (PROJECT_ROOT / "models").resolve()
    baseline_candidate_dir = (PROJECT_ROOT / "artifacts" / "candidate").resolve()

    # Directory safety guards
    if output_dir.resolve() == prod_models_dir:
        raise ValueError(
            f"Refusing to write directly into production models directory: {prod_models_dir}. "
            "Tuned artifacts must be saved to a separate candidate directory."
        )

    if output_dir.resolve() == baseline_candidate_dir:
        raise ValueError(
            f"Refusing to write directly into baseline candidate directory: {baseline_candidate_dir}. "
            "Tuned artifacts must be isolated in a subfolder such as artifacts/candidate/tuned/."
        )

    print("=" * 75)
    print("JobClarity Stratified Cross-Validation XGBoost Tuning Pipeline")
    print("=" * 75)

    # 1. Load dataset
    print(f"[1/5] Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    if "fraudulent" not in df.columns:
        raise KeyError("Target column 'fraudulent' not found in dataset.")

    # 2. Stratified train/test split (Hold-out test split isolated and preserved untouched)
    print(
        f"[2/5] Creating stratified split (train: {1.0 - test_size:.0%}, held-out test: {test_size:.0%})"
    )
    train_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=random_state,
        target_column="fraudulent",
    )

    y_train = train_df["fraudulent"].astype(int)
    y_test = test_df["fraudulent"].astype(int)

    train_samples = len(train_df)
    test_samples = len(test_df)
    train_legit = int((y_train == 0).sum())
    train_fraud = int((y_train == 1).sum())
    test_legit = int((y_test == 0).sum())
    test_fraud = int((y_test == 1).sum())

    print(
        f"      Train samples (80%): {train_samples} (Legit: {train_legit}, Fraud: {train_fraud})"
    )
    print(
        f"      Held-out test (20%): {test_samples} (Legit: {test_legit}, Fraud: {test_fraud}) [UNTOUCHED]"
    )

    # Prepare description-only training text
    X_train_text = prepare_job_description_text(train_df, text_column=text_column)

    # 3. Stratified Cross-Validation on the 80% training split
    print(
        f"[3/5] Performing {n_splits}-fold Stratified CV hyperparameter evaluation on training split..."
    )
    tuning_start_time = time.time()
    cv_results = run_stratified_cv_tuning(
        X_train_text=X_train_text,
        y_train=y_train,
        candidates=candidates,
        n_splits=n_splits,
        random_state=random_state,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    tuning_duration = time.time() - tuning_start_time

    # Best candidate selection based on mean PR-AUC
    best_candidate_rec = cv_results[0]
    best_config_dict = best_candidate_rec["config"]
    best_candidate = CandidateConfig(**best_config_dict)

    print("\n" + "-" * 75)
    print("Cross-Validation Candidates Summary Ranking:")
    print("-" * 75)
    for res in cv_results:
        m = res["mean_metrics"]
        s = res["std_metrics"]
        print(
            f"Rank {res['rank']}: {res['candidate_id']:30s} | "
            f"Mean PR-AUC: {m['mean_pr_auc']:.4f} (+/- {s['std_pr_auc']:.4f}) | "
            f"Mean F1: {m['mean_fraud_f1']:.4f} | Recall: {m['mean_fraud_recall']:.4f} | "
            f"Precision: {m['mean_fraud_precision']:.4f} | Time: {res['cv_duration_seconds']:.1f}s"
        )
    print("-" * 75)
    print(
        f"Selected Best Candidate: {best_candidate.name} ({best_candidate.candidate_id})"
    )
    print(f"Selection Criterion: Mean Cross-Validation PR-AUC (Average Precision)")
    print("-" * 75)

    # 4. Refit best candidate on the entire 80% training split
    print("[4/5] Refitting selected candidate on the entire 80% training split...")
    refit_vec = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    refit_vec, X_train_full = fit_tfidf_vectorizer(refit_vec, X_train_text)

    # Compute final scale_pos_weight strictly on full training split
    if best_candidate.scale_pos_weight == "balanced":
        final_scale_pos_weight = compute_fold_scale_pos_weight(y_train)
    else:
        final_scale_pos_weight = float(best_candidate.scale_pos_weight)

    tuned_model = XGBClassifier(
        n_estimators=best_candidate.n_estimators,
        max_depth=best_candidate.max_depth,
        learning_rate=best_candidate.learning_rate,
        subsample=best_candidate.subsample,
        colsample_bytree=best_candidate.colsample_bytree,
        scale_pos_weight=final_scale_pos_weight,
        random_state=random_state,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    tuned_model.fit(X_train_full, y_train)

    # 5. Save candidate artifacts and comprehensive metadata
    artifacts_saved: dict[str, str] = {}
    if save_artifacts:
        print(f"[5/5] Saving tuned candidate artifacts to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        vec_path = output_dir / "tfidf_vectorizer.pkl"
        model_path = output_dir / "xgboost_model_tuned.pkl"
        report_path = output_dir / "cv_tuning_report.json"
        metadata_path = output_dir / "tuning_metadata.json"

        joblib.dump(refit_vec, vec_path)
        joblib.dump(tuned_model, model_path)

        # Full tuning report
        report_data = {
            "metadata": {
                "pipeline": "leakage_free_stratified_cv_tuning",
                "session": "Day 2 Session 3",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_path": str(data_path),
                "text_representation": {
                    "source_column": text_column,
                    "representation": "job_description_only",
                    "training_serving_consistency": True,
                    "cleaner": "src.preprocessing.text_cleaner.clean_text",
                },
                "cv_strategy": {
                    "method": "StratifiedKFold",
                    "n_splits": n_splits,
                    "shuffle": True,
                    "random_state": random_state,
                    "isolated_to_train_split": True,
                },
                "selection_criterion": "mean_pr_auc (average precision on positive class)",
                "held_out_test_set": {
                    "status": "preserved_untouched",
                    "samples": test_samples,
                    "legitimate_0": test_legit,
                    "fraudulent_1": test_fraud,
                    "evaluated_in_tuning": False,
                    "reservation": "Reserved for Session 4 final test evaluation",
                },
            },
            "dataset_info": {
                "total_records": len(df),
                "train_samples": train_samples,
                "held_out_test_samples": test_samples,
                "train_distribution": {
                    "legitimate_0": train_legit,
                    "fraudulent_1": train_fraud,
                    "fraudulent_ratio": round(float(train_fraud / train_samples), 4),
                },
            },
            "tuning_summary": {
                "candidates_evaluated": len(cv_results),
                "total_cv_duration_seconds": round(tuning_duration, 2),
                "ranked_candidates": cv_results,
            },
            "best_candidate": {
                "rank": 1,
                "candidate_id": best_candidate.candidate_id,
                "name": best_candidate.name,
                "hyperparameters": {
                    **best_candidate.to_dict(),
                    "resolved_final_scale_pos_weight": round(
                        final_scale_pos_weight, 4
                    ),
                },
                "mean_validation_scores": best_candidate_rec["mean_metrics"],
                "std_validation_scores": best_candidate_rec["std_metrics"],
                "selection_rationale": (
                    f"Candidate '{best_candidate.candidate_id}' achieved highest mean CV PR-AUC "
                    f"({best_candidate_rec['mean_metrics']['mean_pr_auc']:.4f}) across {n_splits} stratified folds."
                ),
            },
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Compact tuning metadata
        metadata_compact = {
            "best_candidate_id": best_candidate.candidate_id,
            "best_candidate_name": best_candidate.name,
            "hyperparameters": {
                **best_candidate.to_dict(),
                "resolved_scale_pos_weight": round(final_scale_pos_weight, 4),
            },
            "cv_mean_pr_auc": best_candidate_rec["mean_metrics"]["mean_pr_auc"],
            "cv_mean_fraud_f1": best_candidate_rec["mean_metrics"]["mean_fraud_f1"],
            "cv_mean_fraud_recall": best_candidate_rec["mean_metrics"][
                "mean_fraud_recall"
            ],
            "cv_mean_fraud_precision": best_candidate_rec["mean_metrics"][
                "mean_fraud_precision"
            ],
            "train_samples": train_samples,
            "test_samples": test_samples,
            "test_set_untouched": True,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_compact, f, indent=2)

        artifacts_saved = {
            "vectorizer": str(vec_path),
            "model": str(model_path),
            "cv_report": str(report_path),
            "tuning_metadata": str(metadata_path),
        }
        print(f"Artifacts successfully written to: {output_dir}")

    print("=" * 75)
    print("Tuning Pipeline Completed Successfully")
    print("=" * 75)

    return {
        "best_candidate": best_candidate,
        "cv_results": cv_results,
        "refit_vectorizer": refit_vec,
        "refit_model": tuned_model,
        "artifacts_saved": artifacts_saved,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stratified cross-validation hyperparameter tuning for JobClarity XGBoost."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
        help="Path to cleaned jobs dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "candidate" / "tuned",
        help="Path to output tuned candidate artifacts directory",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of dataset reserved for held-out test split (untouched)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of stratified cross-validation folds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum TF-IDF vocabulary features",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="description",
        help="Text column to use (default: description)",
    )

    args = parser.parse_args()

    tune_xgboost_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_features=args.max_features,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
