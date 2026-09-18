"""Baseline model comparison across candidate architectures."""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT         = Path(__file__).resolve().parent.parent.parent
DATA_SPLIT   = ROOT / "data" / "processed" / "temporal_split"
REPORTS_DIR  = ROOT / "reports" / "model_evaluation"
FIGURES_DIR  = ROOT / "reports" / "figures" / "model_comparison"
MODELS_DIR   = ROOT / "models" / "candidates"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.utils.progress_tracker import ProgressTracker


def run_comparison():
    tracker = ProgressTracker()
    tracker.start_stage("Baseline Model Comparison", "Initializing multi-model baseline comparison...")

    print("=" * 65)
    print("  IEEE-CIS MULTI-MODEL BASELINE COMPARISON")
    print("=" * 65)

    print("  Loading temporal split datasets...")
    train_df = pd.read_parquet(DATA_SPLIT / "train_split.parquet")
    val_df   = pd.read_parquet(DATA_SPLIT / "val_split.parquet")

    drop_cols = ["TransactionID", "isFraud"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"].values.astype(np.int32)
    X_val   = val_df[feature_cols]
    y_val   = val_df["isFraud"].values.astype(np.int32)

    print(f"  Training set:   {len(X_train):,} samples, {len(feature_cols)} features (Fraud: {y_train.sum():,})")
    print(f"  Validation set: {len(X_val):,} samples, {len(feature_cols)} features (Fraud: {y_val.sum():,})")

    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="aucpr"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=63,
            random_state=42
        )
    }

    results = []
    pr_curves = {}
    roc_curves_dict = {}
    conf_matrices = {}

    for model_name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*50}")
        tracker.start_model(model_name, f"Training {model_name} on {len(X_train):,} samples...")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        t1 = time.time()
        val_probs = model.predict_proba(X_val)[:, 1]
        infer_time = time.time() - t1
        print(f"  Inference time: {infer_time:.2f}s ({len(X_val)/infer_time:,.0f} samples/sec)")

        pr_auc = float(average_precision_score(y_val, val_probs))
        roc_auc = float(roc_auc_score(y_val, val_probs))

        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        pr_curves[model_name] = (recall, precision, pr_auc)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        best_f1 = float(f1_scores[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])

        fpr, tpr, _ = roc_curve(y_val, val_probs)
        roc_curves_dict[model_name] = (fpr, tpr, roc_auc)

        val_preds = (val_probs >= best_threshold).astype(int)
        cm = confusion_matrix(y_val, val_preds).tolist()
        conf_matrices[model_name] = cm

        print(f"  Results for {model_name}:")
        print(f"    PR-AUC (Primary): {pr_auc:.4f}")
        print(f"    ROC-AUC:          {roc_auc:.4f}")
        print(f"    Optimal Threshold:{best_threshold:.4f}")
        print(f"    F1 Score:         {best_f1:.4f}")
        print(f"    Recall:           {best_recall*100:.2f}%")
        print(f"    Precision:        {best_precision*100:.2f}%")

        metrics = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "f1": round(best_f1, 4),
            "recall": round(best_recall, 4),
            "precision": round(best_precision, 4),
            "optimal_threshold": round(best_threshold, 4),
            "training_time_s": round(train_time, 2),
            "inference_time_s": round(infer_time, 4)
        }

        results.append({
            "model": model_name,
            **metrics,
            "features_used": len(feature_cols),
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        })

        ckpt_path = MODELS_DIR / f"{model_name.lower()}_baseline.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(model, f)

        tracker.complete_model(model_name, metrics, train_time, infer_time)

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with open(REPORTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved comparison reports to: {REPORTS_DIR}")

    print("\n  Generating comparison visualizations...")
    save_comparison_plots(res_df, pr_curves, roc_curves_dict, conf_matrices)

    tracker.complete_stage("Baseline Model Comparison", "Multi-model baseline comparison completed successfully.")
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON COMPLETE")
    print("=" * 65)


def save_comparison_plots(df, pr_curves, roc_curves, conf_matrices):
    colors = {"LightGBM": "#2ea043", "XGBoost": "#58a6ff", "HistGradientBoosting": "#d29922"}

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["model"], df["pr_auc"], color=[colors.get(m, "#58a6ff") for m in df["model"]], width=0.5)
    plt.ylabel("PR-AUC (Average Precision)", fontsize=11, fontweight="bold")
    plt.title("IEEE-CIS Fraud Detection — Model Comparison (PR-AUC)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, max(df["pr_auc"]) * 1.15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_auc_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["model"], df["roc_auc"], color=[colors.get(m, "#58a6ff") for m in df["model"]], width=0.5)
    plt.ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    plt.title("IEEE-CIS Fraud Detection — Model Comparison (ROC-AUC)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_auc_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["precision"], width, label="Precision", color="#58a6ff")
    plt.bar(x, df["recall"], width, label="Recall", color="#2ea043")
    plt.bar(x + width, df["f1"], width, label="F1-Score", color="#d29922")
    plt.xticks(x, df["model"], fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Precision, Recall, and F1 at Optimal Threshold", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_f1_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["model"], df["training_time_s"], color="#a371f7", width=0.5)
    plt.ylabel("Training Time (seconds)", fontsize=11, fontweight="bold")
    plt.title("Model Training Efficiency Comparison", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}s", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_time_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for m, (r, p, score) in pr_curves.items():
        plt.plot(r, p, label=f"{m} (PR-AUC = {score:.4f})", color=colors.get(m, "blue"), linewidth=2)
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves (Temporal Validation)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for m, (fpr, tpr, score) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{m} (ROC-AUC = {score:.4f})", color=colors.get(m, "blue"), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("ROC Curves (Temporal Validation)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close()

    print(f"  All 6 high-resolution comparison plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_comparison()
