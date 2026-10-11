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


def run_evaluation(
    data_path: Path = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv",
    vectorizer_path: Path = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl",
    tuned_model_path: Path = PROJECT_ROOT / "models" / "xgboost_model_tuned.pkl",
    base_model_path: Path = PROJECT_ROOT / "models" / "xgboost_model.pkl",
    output_report_path: Path = PROJECT_ROOT / "artifacts" / "baseline_evaluation_report.json",
    output_cm_path: Path = PROJECT_ROOT / "artifacts" / "baseline_confusion_matrix.png",
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Run full evaluation suite for both baseline and production models."""
    print("=" * 70)
    print("JobClarity Model Baseline Evaluation")
    print("=" * 70)

    # 1. Load dataset
    print(f"[1/5] Loading dataset from: {data_path.relative_to(PROJECT_ROOT)}")
    df = load_dataset(data_path)
    print(f"      Total records: {len(df):,}, Columns: {len(df.columns)}")

    # 2. Load vectorizer & transform
    print(f"[2/5] Loading TF-IDF vectorizer: {vectorizer_path.relative_to(PROJECT_ROOT)}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    if "combined_text" not in df.columns:
        from src.features.feature_builder import create_combined_text
        df = create_combined_text(df)

    X = vectorizer.transform(df["combined_text"])
    y = df["fraudulent"]

    # 3. Stratified Split (Reproducing exact project split)
    print(f"[3/5] Performing stratified split (test_size={test_size}, random_state={random_state})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
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
    if base_model_path.exists():
        base_model = joblib.load(base_model_path)
        base_res = evaluate_model(base_model, X_test, y_test, model_name="Untuned XGBoost Baseline")
        results["evaluations"]["untuned_baseline"] = base_res
        print("\n--- Untuned Baseline (xgboost_model.pkl) ---")
        for k, v in base_res["metrics"].items():
            print(f"  {k:25s}: {v}")

    # Evaluate Production Tuned Model
    if not tuned_model_path.exists():
        raise FileNotFoundError(f"Tuned model artifact not found at {tuned_model_path}")
    tuned_model = joblib.load(tuned_model_path)
    tuned_res = evaluate_model(tuned_model, X_test, y_test, model_name="Production Tuned XGBoost")
    results["evaluations"]["production_tuned"] = tuned_res
    print("\n--- Production Tuned Model (xgboost_model_tuned.pkl) ---")
    for k, v in tuned_res["metrics"].items():
        print(f"  {k:25s}: {v}")

    print("\nConfusion Matrix (Production Tuned):")
    print(f"  TN: {tuned_res['confusion_matrix']['true_negatives']:,}  |  FP: {tuned_res['confusion_matrix']['false_positives']:,}")
    print(f"  FN: {tuned_res['confusion_matrix']['false_negatives']:,}  |  TP: {tuned_res['confusion_matrix']['true_positives']:,}")

    # 5. Save Artifacts
    print(f"\n[5/5] Saving evaluation report and plots...")
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"      Report saved: {output_report_path.relative_to(PROJECT_ROOT)}")

    # Plot confusion matrix for production model
    cm_arr = np.array(tuned_res["confusion_matrix"]["matrix"])
    plot_confusion_matrix(
        cm_arr,
        title="Production Tuned XGBoost - Confusion Matrix",
        output_path=output_cm_path,
    )
    print(f"      Plot saved:   {output_cm_path.relative_to(PROJECT_ROOT)}")

    # Also plot baseline confusion matrix if base model exists
    if base_model_path.exists():
        base_cm_path = PROJECT_ROOT / "artifacts" / "untuned_baseline_confusion_matrix.png"
        plot_confusion_matrix(
            np.array(base_res["confusion_matrix"]["matrix"]),
            title="Untuned Baseline XGBoost - Confusion Matrix",
            output_path=base_cm_path,
        )
        print(f"      Base plot saved: {base_cm_path.relative_to(PROJECT_ROOT)}")

    print("=" * 70)
    print("Baseline Evaluation Completed Successfully")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate JobClarity models on held-out split.")
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
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        output_report_path=args.output_json,
        output_cm_path=args.output_cm_plot,
    )


if __name__ == "__main__":
    main()
