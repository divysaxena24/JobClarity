"""
Model training utilities for JobClarity.
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.feature_builder import create_combined_text
from src.features.tfidf_vectorizer import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_text,
)
from src.models.evaluate import (
    evaluate_model,
    load_dataset,
    plot_confusion_matrix,
)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_model(
    random_state: int = 42,
):
    """
    Create a baseline XGBoost classifier.
    """

    model = XGBClassifier(
        random_state=random_state,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train the model.
    """

    model.fit(X_train, y_train)

    return model


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    target_column: str = "fraudulent",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset reproducibly using stratified train_test_split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing text and target labels.
    test_size : float, default=0.20
        Fraction of data reserved for final held-out test split.
    random_state : int, default=42
        Fixed random seed for deterministic reproducibility.
    target_column : str, default="fraudulent"
        Binary label column name (0 = legitimate, 1 = fraudulent).

    Returns
    -------
    tuple
        (train_df, test_df)
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


def train_leakage_free_baseline(
    data_path: Path | str = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    output_dir: Path | str = PROJECT_ROOT / "artifacts" / "candidate",
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    save_artifacts: bool = True,
    text_column: str = "description",
) -> dict:
    """
    Execute a minimal, reproducible, leakage-free training/evaluation pipeline.

    Guarantees:
    1. Dataset is split using train_test_split with stratify=y, fixed seed, and 20% test set.
    2. Text representation matches production inference: uses job description text cleaned with clean_text.
    3. TF-IDF vectorizer is fitted ONLY on training text.
    4. Test text is transformed using the already-fitted vectorizer (never fit/fit_transform on test data).
    5. Baseline XGBoost classifier trained on training data only.
    6. Label mapping remains 0 = legitimate and 1 = fraudulent.
    7. Records dataset sizes and class distributions for training and test sets.
    8. Final test set is kept out of all tuning and model selection decisions.
    9. Candidate model and metadata saved to candidate/artifacts directory, never overwriting production models.

    Parameters
    ----------
    data_path : Path or str
        Path to input CSV dataset.
    output_dir : Path or str
        Target directory for baseline candidate artifacts and metadata.
    test_size : float, default=0.20
        Test split fraction.
    random_state : int, default=42
        Random seed for reproducible split and model training.
    max_features : int, default=10000
        Maximum TF-IDF vocabulary features.
    ngram_range : tuple, default=(1, 2)
        N-gram range for TF-IDF vectorizer.
    save_artifacts : bool, default=True
        Whether to save candidate artifacts to output_dir.
    text_column : str, default='description'
        Column name to extract for text representation.

    Returns
    -------
    dict
        Pipeline execution report and metrics.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    prod_models_dir = (PROJECT_ROOT / "models").resolve()

    # Guard: Never overwrite production model files
    if output_dir.resolve() == prod_models_dir:
        raise ValueError(
            f"Refusing to write directly into production models directory: {prod_models_dir}. "
            "Candidate models and evaluation metadata must be saved to a separate artifacts directory."
        )

    print("=" * 70)
    print("JobClarity Leakage-Free Baseline Training Pipeline")
    print("=" * 70)

    # 1. Load dataset
    print(f"[1/6] Loading dataset from: {data_path}")
    df = load_dataset(data_path)

    if "fraudulent" not in df.columns:
        raise KeyError("Required target column 'fraudulent' not found in dataset.")

    # Validate label mapping: 0 = legitimate, 1 = fraudulent
    y_raw = df["fraudulent"]
    unique_labels = set(y_raw.unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Target 'fraudulent' contains invalid labels: {unique_labels}. Expected binary {{0, 1}}."
        )

    # 2. Stratified train/test split on raw texts before vectorization
    print(f"[2/6] Performing stratified split (test_size={test_size}, random_state={random_state})")
    train_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=random_state,
        target_column="fraudulent",
    )

    from src.features.feature_builder import prepare_job_description_text
    X_train_text = prepare_job_description_text(train_df, text_column=text_column)
    y_train = train_df["fraudulent"].astype(int)
    X_test_text = prepare_job_description_text(test_df, text_column=text_column)
    y_test = test_df["fraudulent"].astype(int)


    # 3. Record dataset sizes and class distributions
    train_samples = int(len(y_train))
    test_samples = int(len(y_test))
    total_samples = int(len(df))

    train_legit = int((y_train == 0).sum())
    train_fraud = int((y_train == 1).sum())
    test_legit = int((y_test == 0).sum())
    test_fraud = int((y_test == 1).sum())

    dataset_info = {
        "total_records": total_samples,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "fraud_prevalence_pct": round(float((df["fraudulent"] == 1).mean() * 100), 2),
        "split_parameters": {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "stratified": True,
        },
        "train_distribution": {
            "legitimate_0": train_legit,
            "fraudulent_1": train_fraud,
            "legitimate_ratio": round(float(train_legit / train_samples), 4),
            "fraudulent_ratio": round(float(train_fraud / train_samples), 4),
        },
        "test_distribution": {
            "legitimate_0": test_legit,
            "fraudulent_1": test_fraud,
            "legitimate_ratio": round(float(test_legit / test_samples), 4),
            "fraudulent_ratio": round(float(test_fraud / test_samples), 4),
        },
    }

    print(f"      Total records: {total_samples}")
    print(f"      Train samples: {train_samples} (Legit: {train_legit}, Fraud: {train_fraud})")
    print(f"      Test samples:  {test_samples} (Legit: {test_legit}, Fraud: {test_fraud})")

    # 4. Fit TF-IDF Vectorizer ONLY on training text
    print("[3/6] Fitting TF-IDF vectorizer strictly on training text...")
    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    vectorizer, X_train = fit_tfidf_vectorizer(vectorizer, X_train_text)

    # 5. Transform test text using already-fitted vectorizer
    print("[4/6] Transforming held-out test text (no fit on test data)...")
    X_test = transform_text(vectorizer, X_test_text)
    print(f"      Train feature matrix: {X_train.shape}, Test feature matrix: {X_test.shape}")

    # 6. Train baseline XGBoost classifier on training data only
    print("[5/6] Training baseline XGBoost classifier on training data only...")
    model = create_model(random_state=random_state)
    model = train_model(model, X_train, y_train)

    # 7. Evaluate on held-out test set
    print("[6/6] Evaluating candidate baseline on final held-out test split...")
    eval_results = evaluate_model(
        model,
        X_test,
        y_test,
        model_name="Baseline Candidate XGBoost (Leakage-Free)",
    )

    for k, v in eval_results["metrics"].items():
        print(f"      {k:25s}: {v}")

    # 8. Save candidate artifacts and metadata
    artifacts_saved = {}
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

        vec_path = output_dir / "tfidf_vectorizer.pkl"
        model_path = output_dir / "xgboost_model.pkl"
        report_path = output_dir / "baseline_evaluation_report.json"
        metadata_path = output_dir / "split_metadata.json"
        cm_path = output_dir / "baseline_confusion_matrix.png"

        joblib.dump(vectorizer, vec_path)
        joblib.dump(model, model_path)

        report = {
            "metadata": {
                "pipeline": "leakage_free_baseline_training",
                "leakage_free": True,
                "data_path": str(data_path),
                "text_representation": {
                    "source_column": text_column,
                    "representation": "job_description_only",
                    "training_serving_consistency": True,
                    "cleaner": "src.preprocessing.text_cleaner.clean_text",
                },
                "vectorizer_fitted_on": "train_split_only",
                "model_trained_on": "train_split_only",
                "test_evaluation_only": True,
                "label_mapping": {"0": "legitimate", "1": "fraudulent"},
            },
            "dataset_info": dataset_info,
            "evaluations": {
                "baseline_candidate": eval_results,
            },
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)

        plot_confusion_matrix(
            np.array(eval_results["confusion_matrix"]["matrix"]),
            title="Baseline Candidate XGBoost - Confusion Matrix",
            output_path=cm_path,
        )

        artifacts_saved = {
            "vectorizer": str(vec_path),
            "model": str(model_path),
            "report": str(report_path),
            "metadata": str(metadata_path),
            "confusion_matrix": str(cm_path),
        }
        print(f"\nArtifacts successfully saved to: {output_dir}")

    print("=" * 70)
    print("Baseline Training Pipeline Completed Successfully")
    print("=" * 70)

    return {
        "dataset_info": dataset_info,
        "evaluation": eval_results,
        "vectorizer": vectorizer,
        "model": model,
        "artifacts_saved": artifacts_saved,
    }


# Backwards-compatible alias
run_training_pipeline = train_leakage_free_baseline

# Re-export hyperparameter tuning utilities
from src.models.hyperparameter_tuner import (
    CandidateConfig,
    DEFAULT_CANDIDATES,
    compute_fold_scale_pos_weight,
    run_stratified_cv_tuning,
    tune_xgboost_pipeline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train or tune leakage-free JobClarity models."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run stratified cross-validation hyperparameter tuning pipeline instead of baseline.",
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
        default=None,
        help="Path to output candidate artifacts directory",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of dataset to reserve for test split",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of stratified CV folds when tuning",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Fixed random seed for reproducible split",
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
        help="Column to use for text representation (default: description)",
    )
    args = parser.parse_args()

    if args.tune:
        out_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "candidate" / "tuned")
        tune_xgboost_pipeline(
            data_path=args.data_path,
            output_dir=out_dir,
            test_size=args.test_size,
            n_splits=args.n_splits,
            random_state=args.random_state,
            max_features=args.max_features,
            text_column=args.text_column,
        )
    else:
        out_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "candidate")
        train_leakage_free_baseline(
            data_path=args.data_path,
            output_dir=out_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
            text_column=args.text_column,
        )


if __name__ == "__main__":
    main()
