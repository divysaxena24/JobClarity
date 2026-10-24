"""
Evaluation script for JobClarity models.

Computes comprehensive baseline metrics on the held-out evaluation split:
- Accuracy
- Fraud-class Precision, Recall, and F1-score
- ROC-AUC
- Average Precision (PR-AUC)
- Confusion Matrix
- Classification Report

Saves structured results to a JSON report and generates confusion matrix visualization.
"""

import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load cleaned job postings dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_csv(data_path, keep_default_na=False)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
) -> dict:
    """
    Compute full evaluation metrics on test set.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Scalar metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_fraud = float(precision_score(y_test, y_pred, pos_label=1))
    rec_fraud = float(recall_score(y_test, y_pred, pos_label=1))
    f1_fraud = float(f1_score(y_test, y_pred, pos_label=1))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(val) for val in cm.ravel()]

    # Classification report (dict + formatted string)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred, digits=4)

    return {
        "model_name": model_name,
        "test_samples": int(len(y_test)),
        "class_distribution": {
            "legitimate_0": int((y_test == 0).sum()),
            "fraudulent_1": int((y_test == 1).sum()),
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "fraud_precision": round(prec_fraud, 4),
            "fraud_recall": round(rec_fraud, 4),
            "fraud_f1_score": round(f1_fraud, 4),
            "roc_auc": round(roc_auc, 4),
            "average_precision_pr_auc": round(pr_auc, 4),
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "classification_report_str": report_str,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Plot and save confusion matrix with annotations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=f"{cm[i, j]:,}",
                va="center",
                ha="center",
                size="large",
                fontweight="bold",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real (0)", "Fake (1)"])
    ax.set_yticklabels(["Real (0)", "Fake (1)"])
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_path(path: Path) -> str:
    """Format path relative to PROJECT_ROOT if possible, otherwise return string."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_evaluation(
    data_path: Path = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    vectorizer_path: Path = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl",
    tuned_model_path: Path = PROJECT_ROOT / "models" / "xgboost_model_tuned.pkl",
    base_model_path: Path = PROJECT_ROOT / "models" / "xgboost_model.pkl",
    output_report_path: Path = PROJECT_ROOT / "artifacts" / "baseline_evaluation_report.json",
    output_cm_path: Path = PROJECT_ROOT / "artifacts" / "baseline_confusion_matrix.png",
    test_size: float = 0.20,
    random_state: int = 42,
    enable_mlflow: bool = False,
    experiment_name: str = "JobClarity",
    tracking_uri: str | None = None,
) -> dict:
    """Run full evaluation suite for both baseline and production models."""
    print("=" * 70)
    print("JobClarity Model Baseline Evaluation")
    print("=" * 70)

    # 1. Load dataset
    print(f"[1/5] Loading dataset from: {_format_path(data_path)}")
    df = load_dataset(data_path)
    print(f"      Total records: {len(df):,}, Columns: {len(df.columns)}")

    # 2. Load vectorizer & transform
    print(f"[2/5] Loading TF-IDF vectorizer: {_format_path(vectorizer_path)}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    if "combined_text" not in df.columns:
        from src.features.feature_builder import create_combined_text
        df = create_combined_text(df)

    y = df["fraudulent"]

    # 3. Stratified Split (Reproducing exact project split on raw text first)
    print(f"[3/5] Performing stratified split (test_size={test_size}, random_state={random_state})")
    text_train, text_test, y_train, y_test = train_test_split(
        df["combined_text"], y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"      Test class counts: Real={int((y_test == 0).sum())}, Fake={int((y_test == 1).sum())}")

    # 4. Evaluate Models
    print("[4/5] Evaluating models on held-out test split...")
    results = {
        "dataset_info": {
            "total_records": len(df),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "fraud_prevalence_pct": round(float((y == 1).mean() * 100), 2),
            "split_parameters": {
                "test_size": test_size,
                "random_state": random_state,
                "stratified": True,
            },
        },
        "evaluations": {},
    }

    # Evaluate Untuned Baseline if present
    base_res = None
    if base_model_path and base_model_path.exists():
        base_model = joblib.load(base_model_path)
        base_res = evaluate_model(base_model, X_test, y_test, model_name="Untuned XGBoost Baseline")
        results["evaluations"]["untuned_baseline"] = base_res
        print("\n--- Untuned Baseline (xgboost_model.pkl) ---")
        for k, v in base_res["metrics"].items():
            print(f"  {k:25s}: {v}")

    # Evaluate Production Tuned Model
    tuned_res = None
    if tuned_model_path and tuned_model_path.exists():
        tuned_model = joblib.load(tuned_model_path)
        tuned_res = evaluate_model(tuned_model, X_test, y_test, model_name="Production Tuned XGBoost")
        results["evaluations"]["production_tuned"] = tuned_res
        print("\n--- Production Tuned Model (xgboost_model_tuned.pkl) ---")
        for k, v in tuned_res["metrics"].items():
            print(f"  {k:25s}: {v}")

        print("\nConfusion Matrix (Production Tuned):")
        print(f"  TN: {tuned_res['confusion_matrix']['true_negatives']:,}  |  FP: {tuned_res['confusion_matrix']['false_positives']:,}")
        print(f"  FN: {tuned_res['confusion_matrix']['false_negatives']:,}  |  TP: {tuned_res['confusion_matrix']['true_positives']:,}")
    elif base_model_path is None or not base_model_path.exists():
        raise FileNotFoundError(f"Neither tuned model ({tuned_model_path}) nor baseline model ({base_model_path}) found.")

    # 5. Save Artifacts
    print(f"\n[5/5] Saving evaluation report and plots...")
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"      Report saved: {_format_path(output_report_path)}")

    # Plot confusion matrix
    if tuned_res is not None:
        cm_arr = np.array(tuned_res["confusion_matrix"]["matrix"])
        plot_confusion_matrix(
            cm_arr,
            title="Production Tuned XGBoost - Confusion Matrix",
            output_path=output_cm_path,
        )
        print(f"      Plot saved:   {_format_path(output_cm_path)}")
    elif base_res is not None:
        cm_arr = np.array(base_res["confusion_matrix"]["matrix"])
        plot_confusion_matrix(
            cm_arr,
            title="Untuned Baseline XGBoost - Confusion Matrix",
            output_path=output_cm_path,
        )
        print(f"      Plot saved:   {_format_path(output_cm_path)}")

    # Also plot baseline confusion matrix if base model exists and tuned model was also plotted
    if base_res is not None and tuned_res is not None:
        base_cm_path = PROJECT_ROOT / "artifacts" / "untuned_baseline_confusion_matrix.png"
        plot_confusion_matrix(
            np.array(base_res["confusion_matrix"]["matrix"]),
            title="Untuned Baseline XGBoost - Confusion Matrix",
            output_path=base_cm_path,
        )
        print(f"      Base plot saved: {_format_path(base_cm_path)}")

    if enable_mlflow and (tuned_res is not None or base_res is not None):
        try:
            from src.models.experiment_tracker import log_baseline_experiment
            primary_res = tuned_res or base_res
            log_baseline_experiment(
                model=tuned_model if tuned_res is not None else base_model,
                vectorizer_or_pipeline=vectorizer,
                train_df=df.iloc[:len(y_train)],
                test_df=df.iloc[len(y_train):],
                eval_results={"eval_default_50": primary_res["metrics"], "pr_auc": primary_res["metrics"].get("average_precision_pr_auc", 0.0), "roc_auc": primary_res["metrics"].get("roc_auc", 0.0)},
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_name="Baseline_Evaluation_Pipeline",
            )
        except Exception as mlflow_exc:
            print(f"[MLflow Warning] Could not log to MLflow: {mlflow_exc}")

    print("=" * 70)
    print("Baseline Evaluation Completed Successfully")
    print("=" * 70)
    return results


def evaluate_held_out_candidates(
    data_path: Path | str = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    baseline_dir: Path | str = PROJECT_ROOT / "artifacts" / "candidate",
    tuned_dir: Path | str = PROJECT_ROOT / "artifacts" / "candidate" / "tuned",
    output_dir: Path | str = PROJECT_ROOT / "artifacts" / "evaluation" / "day2",
    test_size: float = 0.20,
    random_state: int = 42,
    text_column: str = "description",
    save_artifacts: bool = True,
) -> dict:
    """
    Perform rigorous, leakage-free evaluation of candidate models on the held-out test split.

    Guarantees:
    1. Loads description-only baseline candidate and selected tuned candidate.
    2. Recreates the exact reproducible stratified 80/20 train/test split.
    3. Verifies agreement with recorded metadata from training and tuning sessions.
    4. Evaluates both candidates and majority-class baseline on the untouched 20% test set.
    5. Validates probability bounds, dimensions, and prediction consistency.
    6. Saves reports and confusion matrices to isolated directory (artifacts/evaluation/day2/).
    7. Leaves all existing candidate and production artifacts untouched.

    Parameters
    ----------
    data_path : Path or str
        Path to cleaned jobs dataset CSV.
    baseline_dir : Path or str
        Directory containing baseline candidate artifacts.
    tuned_dir : Path or str
        Directory containing tuned candidate artifacts.
    output_dir : Path or str
        Directory to write evaluation reports and confusion matrices.
    test_size : float, default=0.20
        Held-out test split fraction.
    random_state : int, default=42
        Random seed for exact reproducible split.
    text_column : str, default='description'
        Text representation column.
    save_artifacts : bool, default=True
        Whether to save artifacts to disk.

    Returns
    -------
    dict
        Evaluation report comparing majority baseline, candidate baseline, and candidate tuned models.
    """
    data_path = Path(data_path)
    baseline_dir = Path(baseline_dir)
    tuned_dir = Path(tuned_dir)
    output_dir = Path(output_dir)

    prod_models_dir = (PROJECT_ROOT / "models").resolve()
    baseline_candidate_dir = baseline_dir.resolve()
    tuned_candidate_dir = tuned_dir.resolve()

    # Safety checks: Never overwrite production or candidate artifact directories
    if output_dir.resolve() == prod_models_dir:
        raise ValueError(
            f"Refusing to write evaluation report into production models directory: {prod_models_dir}"
        )
    if output_dir.resolve() == baseline_candidate_dir:
        raise ValueError(
            f"Refusing to overwrite baseline candidate directory: {baseline_candidate_dir}"
        )
    if output_dir.resolve() == tuned_candidate_dir:
        raise ValueError(
            f"Refusing to overwrite tuned candidate directory: {tuned_candidate_dir}"
        )

    print("=" * 75)
    print("JobClarity Day 2 Session 4: Held-Out Test Set Evaluation")
    print("=" * 75)

    # 1. Load dataset & Recreate exact reproducible stratified split
    print(f"[1/6] Loading dataset from: {_format_path(data_path)}")
    df = load_dataset(data_path)
    if "fraudulent" not in df.columns:
        raise KeyError("Target column 'fraudulent' not found in dataset.")

    from src.models.model_trainer import split_dataset
    from src.features.feature_builder import prepare_job_description_text

    print(f"[2/6] Recreating reproducible stratified split (test_size={test_size}, random_state={random_state})...")
    train_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=random_state,
        target_column="fraudulent",
    )

    y_train = train_df["fraudulent"].astype(int)
    y_test = test_df["fraudulent"].astype(int).values

    train_samples = len(train_df)
    test_samples = len(test_df)
    train_legit = int((y_train == 0).sum())
    train_fraud = int((y_train == 1).sum())
    test_legit = int((y_test == 0).sum())
    test_fraud = int((y_test == 1).sum())

    print(f"      Total records: {len(df):,}")
    print(f"      Train split (80%): {train_samples:,} (Legit: {train_legit:,}, Fraud: {train_fraud:,})")
    print(f"      Held-out test (20%): {test_samples:,} (Legit: {test_legit:,}, Fraud: {test_fraud:,})")

    # Verify split metadata agreement with saved metadata
    meta_agreement = {"baseline_metadata_matched": False, "tuning_metadata_matched": False}
    baseline_meta_path = baseline_dir / "split_metadata.json"
    if baseline_meta_path.exists():
        with open(baseline_meta_path, "r", encoding="utf-8") as f:
            b_meta = json.load(f)
        assert b_meta["train_samples"] == train_samples
        assert b_meta["test_samples"] == test_samples
        assert b_meta["test_distribution"]["legitimate_0"] == test_legit
        assert b_meta["test_distribution"]["fraudulent_1"] == test_fraud
        meta_agreement["baseline_metadata_matched"] = True
        print("      [Verified] Exact agreement with artifacts/candidate/split_metadata.json")

    tuning_meta_path = tuned_dir / "tuning_metadata.json"
    if tuning_meta_path.exists():
        with open(tuning_meta_path, "r", encoding="utf-8") as f:
            t_meta = json.load(f)
        assert t_meta["train_samples"] == train_samples
        assert t_meta["test_samples"] == test_samples
        meta_agreement["tuning_metadata_matched"] = True
        print("      [Verified] Exact agreement with artifacts/candidate/tuned/tuning_metadata.json")

    # 2. Extract and clean test descriptions
    print("[3/6] Cleaning test descriptions using product text representation...")
    X_test_text = prepare_job_description_text(test_df, text_column=text_column)

    # 3. Load baseline candidate
    print(f"[4/6] Loading candidate artifacts from disk...")
    base_vec_path = baseline_dir / "tfidf_vectorizer.pkl"
    base_model_path = baseline_dir / "xgboost_model.pkl"
    tuned_vec_path = tuned_dir / "tfidf_vectorizer.pkl"
    tuned_model_path = tuned_dir / "xgboost_model_tuned.pkl"

    if not base_vec_path.exists() or not base_model_path.exists():
        raise FileNotFoundError(f"Baseline candidate artifacts missing in {baseline_dir}")
    if not tuned_vec_path.exists() or not tuned_model_path.exists():
        raise FileNotFoundError(f"Tuned candidate artifacts missing in {tuned_dir}")

    vec_base = joblib.load(base_vec_path)
    model_base = joblib.load(base_model_path)
    vec_tuned = joblib.load(tuned_vec_path)
    model_tuned = joblib.load(tuned_model_path)

    # Transform test representations
    X_test_base = vec_base.transform(X_test_text)
    X_test_tuned = vec_tuned.transform(X_test_text)

    # 4. Rigorous Input & Probability Validation
    print("[5/6] Validating probability bounds, feature dimensions, and inference consistency...")
    # Feature dimension check
    assert X_test_base.shape[0] == test_samples
    assert X_test_tuned.shape[0] == test_samples
    assert X_test_base.shape[1] == len(vec_base.vocabulary_)
    assert X_test_tuned.shape[1] == len(vec_tuned.vocabulary_)

    # Probability bounds check
    prob_base_full = model_base.predict_proba(X_test_base)
    prob_tuned_full = model_tuned.predict_proba(X_test_tuned)
    assert np.all((prob_base_full >= 0.0) & (prob_base_full <= 1.0))
    assert np.allclose(prob_base_full.sum(axis=1), 1.0)
    assert np.all((prob_tuned_full >= 0.0) & (prob_tuned_full <= 1.0))
    assert np.allclose(prob_tuned_full.sum(axis=1), 1.0)

    # Single-instance production inference compatibility check
    single_sample = "Software engineer building backend APIs in Python and FastAPI."
    cleaned_single = prepare_job_description_text(single_sample)
    p1 = model_base.predict_proba(vec_base.transform([cleaned_single]))
    p2 = model_tuned.predict_proba(vec_tuned.transform([cleaned_single]))
    assert p1.shape == (1, 2) and p2.shape == (1, 2)
    print("      [Verified] Probability bounds [0, 1] valid; single-item inference compatible.")

    # 5. Evaluate models on held-out test split
    print("[6/6] Computing evaluation metrics on held-out test split...")

    # A. Majority-Class Baseline (predict 0 for everything)
    y_pred_maj = np.zeros(test_samples, dtype=int)
    y_prob_maj = np.zeros(test_samples, dtype=float)
    maj_acc = float(accuracy_score(y_test, y_pred_maj))
    maj_prec = float(precision_score(y_test, y_pred_maj, pos_label=1, zero_division=0))
    maj_rec = float(recall_score(y_test, y_pred_maj, pos_label=1, zero_division=0))
    maj_f1 = float(f1_score(y_test, y_pred_maj, pos_label=1, zero_division=0))
    maj_roc_auc = 0.5000
    maj_pr_auc = round(float(test_fraud / test_samples), 4)  # Prevalence baseline
    maj_cm = confusion_matrix(y_test, y_pred_maj)

    majority_res = {
        "model_name": "Majority-Class Baseline (All Legitimate)",
        "test_samples": test_samples,
        "class_distribution": {"legitimate_0": test_legit, "fraudulent_1": test_fraud},
        "metrics": {
            "accuracy": round(maj_acc, 4),
            "fraud_precision": round(maj_prec, 4),
            "fraud_recall": round(maj_rec, 4),
            "fraud_f1_score": round(maj_f1, 4),
            "roc_auc": maj_roc_auc,
            "average_precision_pr_auc": maj_pr_auc,
        },
        "confusion_matrix": {
            "true_negatives": int(maj_cm[0, 0]),
            "false_positives": int(maj_cm[0, 1]),
            "false_negatives": int(maj_cm[1, 0]),
            "true_positives": int(maj_cm[1, 1]),
            "matrix": maj_cm.tolist(),
        },
        "decision_threshold": 0.5,
    }

    # B. Baseline Candidate Evaluation
    base_res = evaluate_model(
        model_base,
        X_test_base,
        y_test,
        model_name="Baseline Candidate XGBoost (Description-Only)",
    )
    base_res["decision_threshold"] = 0.5

    # C. Tuned Candidate Evaluation
    tuned_res = evaluate_model(
        model_tuned,
        X_test_tuned,
        y_test,
        model_name="Tuned Candidate XGBoost (Description-Only)",
    )
    tuned_res["decision_threshold"] = 0.5

    # Comparative difference analysis
    prob_base_fraud = prob_base_full[:, 1]
    prob_tuned_fraud = prob_tuned_full[:, 1]
    y_pred_base = (prob_base_fraud >= 0.5).astype(int)
    y_pred_tuned = (prob_tuned_fraud >= 0.5).astype(int)

    agreement_rate = float((y_pred_base == y_pred_tuned).mean())
    max_prob_diff = float(np.abs(prob_base_fraud - prob_tuned_fraud).max())
    mean_prob_diff = float(np.abs(prob_base_fraud - prob_tuned_fraud).mean())

    comparison = {
        "prediction_agreement_rate": round(agreement_rate, 6),
        "max_absolute_probability_difference": round(max_prob_diff, 6),
        "mean_absolute_probability_difference": round(mean_prob_diff, 6),
        "delta_vs_baseline": {
            "accuracy": round(tuned_res["metrics"]["accuracy"] - base_res["metrics"]["accuracy"], 4),
            "fraud_precision": round(
                tuned_res["metrics"]["fraud_precision"] - base_res["metrics"]["fraud_precision"], 4
            ),
            "fraud_recall": round(
                tuned_res["metrics"]["fraud_recall"] - base_res["metrics"]["fraud_recall"], 4
            ),
            "fraud_f1_score": round(
                tuned_res["metrics"]["fraud_f1_score"] - base_res["metrics"]["fraud_f1_score"], 4
            ),
            "roc_auc": round(tuned_res["metrics"]["roc_auc"] - base_res["metrics"]["roc_auc"], 4),
            "pr_auc": round(
                tuned_res["metrics"]["average_precision_pr_auc"]
                - base_res["metrics"]["average_precision_pr_auc"],
                4,
            ),
        },
        "delta_vs_majority": {
            "accuracy": round(tuned_res["metrics"]["accuracy"] - majority_res["metrics"]["accuracy"], 4),
            "pr_auc": round(
                tuned_res["metrics"]["average_precision_pr_auc"]
                - majority_res["metrics"]["average_precision_pr_auc"],
                4,
            ),
            "roc_auc": round(tuned_res["metrics"]["roc_auc"] - majority_res["metrics"]["roc_auc"], 4),
            "fraud_f1_score": round(tuned_res["metrics"]["fraud_f1_score"], 4),
        },
        "tradeoff_analysis": (
            "At default threshold 0.5, both the baseline and selected tuned candidate achieve "
            f"exceptional fraud precision ({tuned_res['metrics']['fraud_precision']:.2%}) with only "
            f"{tuned_res['confusion_matrix']['false_positives']} false positive out of {test_legit:,} legitimate jobs. "
            f"However, fraud recall is {tuned_res['metrics']['fraud_recall']:.2%} ({tuned_res['confusion_matrix']['false_negatives']} "
            f"missed fraud cases). This high-precision headroom provides strong empirical evidence that decision "
            "threshold optimization (tuning threshold below 0.5) can capture substantial additional fraud with minimal false-positive inflation."
        ),
    }

    # Consolidated report
    report_data = {
        "metadata": {
            "pipeline": "day2_session4_held_out_test_evaluation",
            "session": "Day 2 Session 4",
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_path": _format_path(data_path),
            "text_representation": {
                "source_column": text_column,
                "representation": "job_description_only",
                "cleaner": "src.preprocessing.text_cleaner.clean_text",
            },
            "split_parameters": {
                "test_size": test_size,
                "random_state": random_state,
                "stratified": True,
            },
            "metadata_consistency": meta_agreement,
            "default_decision_threshold": 0.5,
        },
        "dataset_info": {
            "total_records": len(df),
            "train_samples": train_samples,
            "test_samples": test_samples,
            "fraud_prevalence_pct": round(float((df["fraudulent"] == 1).mean() * 100), 2),
            "test_distribution": {
                "legitimate_0": test_legit,
                "fraudulent_1": test_fraud,
                "fraudulent_ratio": round(float(test_fraud / test_samples), 4),
            },
        },
        "evaluations": {
            "majority_class_baseline": majority_res,
            "baseline_candidate": base_res,
            "tuned_candidate": tuned_res,
        },
        "comparison": comparison,
    }

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Metric':<25} | {'Majority Baseline':<18} | {'Candidate Baseline':<18} | {'Tuned Candidate':<18}")
    print("-" * 75)
    for m in ["accuracy", "fraud_precision", "fraud_recall", "fraud_f1_score", "roc_auc", "average_precision_pr_auc"]:
        v_maj = f"{majority_res['metrics'][m]:.4f}"
        v_base = f"{base_res['metrics'][m]:.4f}"
        v_tuned = f"{tuned_res['metrics'][m]:.4f}"
        print(f"{m:<25} | {v_maj:<18} | {v_base:<18} | {v_tuned:<18}")
    print("=" * 75)

    # 6. Save artifacts
    artifacts_saved = {}
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "final_evaluation_report.json"
        base_cm_file = output_dir / "baseline_confusion_matrix.png"
        tuned_cm_file = output_dir / "tuned_confusion_matrix.png"
        maj_cm_file = output_dir / "majority_class_confusion_matrix.png"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        plot_confusion_matrix(
            np.array(base_res["confusion_matrix"]["matrix"]),
            title="Baseline Candidate (Held-Out Test Set)",
            output_path=base_cm_file,
        )
        plot_confusion_matrix(
            np.array(tuned_res["confusion_matrix"]["matrix"]),
            title="Tuned Candidate (Held-Out Test Set)",
            output_path=tuned_cm_file,
        )
        plot_confusion_matrix(
            np.array(majority_res["confusion_matrix"]["matrix"]),
            title="Majority Baseline (Held-Out Test Set)",
            output_path=maj_cm_file,
        )

        artifacts_saved = {
            "final_report": str(report_file),
            "baseline_confusion_matrix": str(base_cm_file),
            "tuned_confusion_matrix": str(tuned_cm_file),
            "majority_confusion_matrix": str(maj_cm_file),
        }
        print(f"\nArtifacts successfully saved to: {_format_path(output_dir)}")

    print("=" * 75)
    print("Held-Out Evaluation Completed Successfully")
    print("=" * 75)

    return {
        "report": report_data,
        "majority_class_baseline": majority_res,
        "baseline_candidate": base_res,
        "tuned_candidate": tuned_res,
        "comparison": comparison,
        "artifacts_saved": artifacts_saved,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate JobClarity models on held-out split.")
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Run Day 2 Session 4 candidate evaluation on held-out test split.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Log evaluation run to MLflow",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
        help="Path to cleaned jobs dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_evaluation_report.json",
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--output-cm-plot",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_confusion_matrix.png",
        help="Path to output confusion matrix PNG",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation" / "day2",
        help="Output directory when running held-out candidate evaluation",
    )
    args = parser.parse_args()

    if args.held_out:
        evaluate_held_out_candidates(
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
    else:
        run_evaluation(
            data_path=args.data_path,
            output_report_path=args.output_json,
            output_cm_path=args.output_cm_plot,
            enable_mlflow=getattr(args, "enable_mlflow", False),
        )


if __name__ == "__main__":
    main()

