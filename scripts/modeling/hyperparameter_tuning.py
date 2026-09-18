"""Hyperparameter optimization targeting PR-AUC."""

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

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT         = Path(__file__).resolve().parent.parent.parent
DATA_SPLIT   = ROOT / "data" / "processed" / "temporal_split"
REPORTS_DIR  = ROOT / "reports" / "model_evaluation"
FIGURES_DIR  = ROOT / "reports" / "figures" / "tuning"
MODELS_DIR   = ROOT / "models" / "tuned"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.utils.progress_tracker import ProgressTracker

SEARCH_SPACES = {
    "LightGBM": {
        "n_estimators": [300],
        "learning_rate": [0.03, 0.05, 0.08],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_samples": [20, 50, 100],
        "scale_pos_weight": [1.0, 3.0, 5.0]
    },
    "XGBoost": {
        "n_estimators": [300],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [5, 6, 8],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 5, 10],
        "scale_pos_weight": [1.0, 3.0, 5.0]
    },
    "HistGradientBoosting": {
        "max_iter": [300],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_leaf_nodes": [31, 63, 127],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0]
    }
}


def sample_params(space, rng):
    return {k: rng.choice(v) for k, v in space.items()}


def run_tuning(target_model="LightGBM", n_trials=10, random_seed=42):
    tracker = ProgressTracker()
    tracker.start_stage("Hyperparameter Tuning", f"Configuring {n_trials} tuning trials for {target_model}...")

    print("=" * 65)
    print(f"  HYPERPARAMETER TUNING: {target_model} ({n_trials} Trials)")
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

    rng = np.random.RandomState(random_seed)
    space = SEARCH_SPACES[target_model]

    tracker.start_tuning(target_model, n_trials)

    trial_records = []
    best_score = -1.0
    best_params = None
    best_model_obj = None

    for t_idx in range(1, n_trials + 1):
        params = sample_params(space, rng)
        print(f"\n  [Trial {t_idx}/{n_trials}] Testing params: {params}")

        t0 = time.time()
        if target_model == "LightGBM":
            model = lgb.LGBMClassifier(
                **params,
                random_state=random_seed + t_idx,
                n_jobs=-1,
                verbosity=-1
            )
        elif target_model == "XGBoost":
            model = xgb.XGBClassifier(
                **params,
                random_state=random_seed + t_idx,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="aucpr"
            )
        else:
            model = HistGradientBoostingClassifier(
                **params,
                random_state=random_seed + t_idx
            )

        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        val_probs = model.predict_proba(X_val)[:, 1]
        pr_auc = float(average_precision_score(y_val, val_probs))
        roc_auc = float(roc_auc_score(y_val, val_probs))

        is_best = pr_auc > best_score
        if is_best:
            best_score = pr_auc
            best_params = params
            best_model_obj = model
            print(f"    >>> NEW BEST PR-AUC: {pr_auc:.4f} <<<")
        else:
            print(f"    PR-AUC: {pr_auc:.4f} (Current Best: {best_score:.4f})")

        rec = {
            "trial": t_idx,
            "model": target_model,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "fit_time_s": round(fit_time, 2),
            "params": params,
            "is_best": is_best
        }
        trial_records.append(rec)

        tracker.update_trial(target_model, t_idx, n_trials, pr_auc, is_best)

    best_ckpt = MODELS_DIR / f"{target_model.lower()}_best_tuned.pkl"
    with open(best_ckpt, "wb") as f:
        pickle.dump(best_model_obj, f)

    res_df = pd.DataFrame(trial_records)
    res_df.to_csv(REPORTS_DIR / f"{target_model.lower()}_tuning_results.csv", index=False)
    with open(REPORTS_DIR / f"{target_model.lower()}_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(trial_records, f, indent=2)

    plt.figure(figsize=(9, 5))
    scores = [r["pr_auc"] for r in trial_records]
    plt.plot(range(1, n_trials + 1), scores, marker="o", color="#58a6ff", label="Trial PR-AUC")
    cum_max = np.maximum.accumulate(scores)
    plt.plot(range(1, n_trials + 1), cum_max, linestyle="--", color="#2ea043", label="Best PR-AUC Progress")
    plt.xlabel("Trial Number", fontsize=11, fontweight="bold")
    plt.ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
    plt.title(f"{target_model} Hyperparameter Tuning Progression", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{target_model.lower()}_tuning.png", dpi=300)
    plt.close()

    tracker.complete_tuning(target_model, best_score, best_params)
    print(f"\n  Tuning finished. Best Score: {best_score:.4f}")
    print(f"  Best Parameters: {best_params}")


if __name__ == "__main__":
    run_tuning()
