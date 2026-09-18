"""Hyperparameter tuning, threshold calibration, and model comparison."""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.utils.progress_tracker import ProgressTracker

DATA_SPLIT = ROOT / "data" / "processed" / "temporal_split"
REPORTS_DIR = ROOT / "reports" / "model_evaluation"
MODELS_TUNED = ROOT / "models" / "tuned"
FIG_TUNING = ROOT / "reports" / "figures" / "tuning"
FIG_EXP = ROOT / "reports" / "figures" / "feature_experiments"

MODELS_TUNED.mkdir(parents=True, exist_ok=True)
FIG_TUNING.mkdir(parents=True, exist_ok=True)
FIG_EXP.mkdir(parents=True, exist_ok=True)

tracker = ProgressTracker()
tracker.start_stage("Hyperparameter Tuning", "Starting hyperparameter tuning with accuracy tracking for LightGBM, XGBoost, HistGradient...")

print("=" * 70)
print("  PHASE 5: HYPERPARAMETER TUNING & ACCURACY EVALUATION")
print("=" * 70)

print("  Loading Train V2 and Val V2 datasets...")
t0 = time.time()
train_v2 = pd.read_parquet(DATA_SPLIT / "train_features_v2.parquet")
val_v2 = pd.read_parquet(DATA_SPLIT / "val_features_v2.parquet")

y_train = train_v2["isFraud"].values.astype(np.int32)
y_val = val_v2["isFraud"].values.astype(np.int32)

feature_meta_path = ROOT / "reports" / "data_and_features" / "feature_groups_v2.json"
if not feature_meta_path.exists():
    feature_meta_path = ROOT / "reports" / "feature_groups_v2.json"
with open(feature_meta_path, "r") as f:
    fg_meta = json.load(f)

base_cols = fg_meta["baseline_features"]
v2_cols = fg_meta["total_v2_features"]

print(f"  Loaded in {time.time()-t0:.1f}s. Train: {train_v2.shape}, Val: {val_v2.shape}")
print(f"  Baseline features: {len(base_cols)}, V2 features: {len(v2_cols)}")

def evaluate_preds(y_true, y_prob):
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[min(best_idx, len(thresholds)-1)])
    
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    f1 = float(f1_score(y_true, y_pred_opt))
    prec = float(precision_score(y_true, y_pred_opt))
    rec = float(recall_score(y_true, y_pred_opt))
    acc_opt = float(accuracy_score(y_true, y_pred_opt))
    
    y_pred_50 = (y_prob >= 0.50).astype(int)
    acc_50 = float(accuracy_score(y_true, y_pred_50))
    
    return pr_auc, roc_auc, f1, prec, rec, acc_opt, acc_50, best_thresh

lgb_trials = [
    {"num_leaves": 63, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.8, "min_child_samples": 50, "n_estimators": 150},
    {"num_leaves": 95, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.75, "min_child_samples": 30, "n_estimators": 200},
    {"num_leaves": 127, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.7, "min_child_samples": 40, "n_estimators": 250},
    {"num_leaves": 127, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 50, "scale_pos_weight": 1.5, "n_estimators": 180},
    {"num_leaves": 150, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.75, "min_child_samples": 30, "n_estimators": 220}
]

xgb_trials = [
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.8, "min_child_weight": 5, "n_estimators": 150},
    {"max_depth": 7, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 3, "n_estimators": 180},
    {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.7, "min_child_weight": 5, "n_estimators": 200}
]

hgb_trials = [
    {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 0.1},
    {"max_iter": 180, "learning_rate": 0.04, "max_leaf_nodes": 95, "min_samples_leaf": 30, "l2_regularization": 0.0},
    {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 127, "min_samples_leaf": 40, "l2_regularization": 0.5}
]

tuning_records = []
total_tuning_trials = len(lgb_trials) + len(xgb_trials) + len(hgb_trials)
current_global_trial = 0

# --- LightGBM Tuning ---
print("\n--- Tuning LightGBM (5 candidate configurations on V2 features) ---")
best_lgb_score = -1
best_lgb_model = None
best_lgb_params = None
lgb_trial_scores = []

for idx, p in enumerate(lgb_trials, 1):
    current_global_trial += 1
    t_start = time.time()
    m = lgb.LGBMClassifier(**p, random_state=42 + idx, n_jobs=-1, verbose=-1)
    m.fit(train_v2[v2_cols], y_train)
    fit_time = time.time() - t_start
    
    probs = m.predict_proba(val_v2[v2_cols])[:, 1]
    pr_auc, roc_auc, f1, prec, rec, acc_opt, acc_50, thresh = evaluate_preds(y_val, probs)
    lgb_trial_scores.append(pr_auc)
    is_best = pr_auc > best_lgb_score
    if is_best:
        best_lgb_score = pr_auc
        best_lgb_model = m
        best_lgb_params = p
        
    print(f"  [LGB Trial {idx}/5] PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Acc(Opt): {acc_opt*100:.2f}% | Acc(0.5): {acc_50*100:.2f}% | F1: {f1:.4f} | Time: {fit_time:.1f}s")
    tuning_records.append({
        "trial_id": f"LGB_{idx}",
        "model": "LightGBM",
        "params": json.dumps(p),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy_optimal": round(acc_opt, 4),
        "accuracy_default_05": round(acc_50, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(thresh, 4),
        "fit_time_sec": round(fit_time, 1),
        "is_best_for_model": is_best
    })
    tracker.update_stage_progress(round(current_global_trial / total_tuning_trials * 50, 1), f"Tuning LightGBM Trial {idx}/5 (PR-AUC: {pr_auc:.4f})")

# Save LightGBM tuned model
with open(MODELS_TUNED / "lightgbm_tuned.pkl", "wb") as f:
    pickle.dump(best_lgb_model, f)
print(f"  Saved champion tuned LightGBM model (PR-AUC: {best_lgb_score:.4f})")

# --- XGBoost Tuning ---
print("\n--- Tuning XGBoost (3 candidate configurations on V2 features) ---")
best_xgb_score = -1
best_xgb_model = None
best_xgb_params = None
xgb_trial_scores = []

for idx, p in enumerate(xgb_trials, 1):
    current_global_trial += 1
    t_start = time.time()
    m = xgb.XGBClassifier(**p, random_state=42 + idx, n_jobs=-1, tree_method="hist", eval_metric="aucpr")
    m.fit(train_v2[v2_cols], y_train)
    fit_time = time.time() - t_start
    
    probs = m.predict_proba(val_v2[v2_cols])[:, 1]
    pr_auc, roc_auc, f1, prec, rec, acc_opt, acc_50, thresh = evaluate_preds(y_val, probs)
    xgb_trial_scores.append(pr_auc)
    is_best = pr_auc > best_xgb_score
    if is_best:
        best_xgb_score = pr_auc
        best_xgb_model = m
        best_xgb_params = p
        
    print(f"  [XGB Trial {idx}/3] PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Acc(Opt): {acc_opt*100:.2f}% | Acc(0.5): {acc_50*100:.2f}% | F1: {f1:.4f} | Time: {fit_time:.1f}s")
    tuning_records.append({
        "trial_id": f"XGB_{idx}",
        "model": "XGBoost",
        "params": json.dumps(p),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy_optimal": round(acc_opt, 4),
        "accuracy_default_05": round(acc_50, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(thresh, 4),
        "fit_time_sec": round(fit_time, 1),
        "is_best_for_model": is_best
    })
    tracker.update_stage_progress(round(current_global_trial / total_tuning_trials * 50, 1), f"Tuning XGBoost Trial {idx}/3 (PR-AUC: {pr_auc:.4f})")

with open(MODELS_TUNED / "xgboost_tuned.pkl", "wb") as f:
    pickle.dump(best_xgb_model, f)
print(f"  Saved tuned XGBoost model (PR-AUC: {best_xgb_score:.4f})")

print("\n--- Tuning HistGradientBoosting (3 candidate configurations on V2 features) ---")
best_hgb_score = -1
best_hgb_model = None
best_hgb_params = None
hgb_trial_scores = []

for idx, p in enumerate(hgb_trials, 1):
    current_global_trial += 1
    t_start = time.time()
    m = HistGradientBoostingClassifier(**p, random_state=42 + idx)
    m.fit(train_v2[v2_cols], y_train)
    fit_time = time.time() - t_start
    
    probs = m.predict_proba(val_v2[v2_cols])[:, 1]
    pr_auc, roc_auc, f1, prec, rec, acc_opt, acc_50, thresh = evaluate_preds(y_val, probs)
    hgb_trial_scores.append(pr_auc)
    is_best = pr_auc > best_hgb_score
    if is_best:
        best_hgb_score = pr_auc
        best_hgb_model = m
        best_hgb_params = p
        
    print(f"  [HGB Trial {idx}/3] PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Acc(Opt): {acc_opt*100:.2f}% | Acc(0.5): {acc_50*100:.2f}% | F1: {f1:.4f} | Time: {fit_time:.1f}s")
    tuning_records.append({
        "trial_id": f"HGB_{idx}",
        "model": "HistGradientBoosting",
        "params": json.dumps(p),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy_optimal": round(acc_opt, 4),
        "accuracy_default_05": round(acc_50, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(thresh, 4),
        "fit_time_sec": round(fit_time, 1),
        "is_best_for_model": is_best
    })
    tracker.update_stage_progress(round(current_global_trial / total_tuning_trials * 50, 1), f"Tuning HistGradient Trial {idx}/3 (PR-AUC: {pr_auc:.4f})")

with open(MODELS_TUNED / "histgradientboosting_tuned.pkl", "wb") as f:
    pickle.dump(best_hgb_model, f)
print(f"  Saved tuned HistGradientBoosting model (PR-AUC: {best_hgb_score:.4f})")

pd.DataFrame(tuning_records).to_csv(REPORTS_DIR / "hyperparameter_tuning.csv", index=False)
with open(REPORTS_DIR / "hyperparameter_tuning.json", "w") as f:
    json.dump(tuning_records, f, indent=2)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(lgb_trial_scores)+1), lgb_trial_scores, marker="o", color="#3498db", linewidth=2, label="Trial PR-AUC")
plt.plot(range(1, len(lgb_trial_scores)+1), np.maximum.accumulate(lgb_trial_scores), linestyle="--", color="#2ecc71", linewidth=2, label="Best Cumulative PR-AUC")
plt.title("LightGBM Hyperparameter Tuning Progression (PR-AUC)", fontsize=13, fontweight="bold")
plt.xlabel("Trial Number", fontsize=11, fontweight="bold")
plt.ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(FIG_TUNING / "lightgbm_tuning.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
models_comp = ["LightGBM", "XGBoost", "HistGradientBoosting"]
best_scores = [best_lgb_score, best_xgb_score, best_hgb_score]
bars = plt.bar(models_comp, best_scores, color=["#2ECC71", "#E67E22", "#3498DB"], width=0.5)
plt.ylabel("Best Validation PR-AUC", fontsize=11, fontweight="bold")
plt.title("Tuned Model Comparison — PR-AUC (Average Precision)", fontsize=13, fontweight="bold")
plt.ylim(0.50, max(best_scores) + 0.05)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.003, f"{yval:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_TUNING / "tuning_comparison.png", dpi=300)
plt.close()

champion_model = best_lgb_model
champion_probs = champion_model.predict_proba(val_v2[v2_cols])[:, 1]

threshold_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
calibration_table = []

for th in threshold_grid:
    preds = (champion_probs >= th).astype(int)
    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    f = f1_score(y_val, preds, zero_division=0)
    acc = accuracy_score(y_val, preds)
    capture_rate = (tp / (tp + fn)) * 100
    
    calibration_table.append({
        "threshold": th,
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_capture_rate_pct": round(float(capture_rate), 2)
    })
    print(f"  Thresh {th:.2f} -> Acc: {acc*100:5.2f}% | Prec: {p:.4f} | Rec: {r:.4f} | F1: {f:.4f} | TP: {tp:>4} | FP: {fp:>5} | FN: {fn:>4} | Capture: {capture_rate:5.1f}%")

cal_df = pd.DataFrame(calibration_table)
with open(REPORTS_DIR / "threshold_calibration.json", "w") as f:
    json.dump(calibration_table, f, indent=2)

print("  Threshold Calibration Table (Subset):")
print(cal_df[["threshold", "accuracy", "precision", "recall", "f1", "fraud_capture_rate_pct"]].to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(cal_df["threshold"], cal_df["accuracy"] * 100, label="Accuracy (%)", color="#1ABC9C", linewidth=2)
plt.plot(cal_df["threshold"], cal_df["precision"] * 100, label="Precision (%)", color="#3498DB", linewidth=2)
plt.plot(cal_df["threshold"], cal_df["recall"] * 100, label="Recall (%)", color="#E74C3C", linewidth=2)
plt.plot(cal_df["threshold"], cal_df["f1"] * 100, label="F1 Score (%)", color="#F39C12", linewidth=2)
plt.axvline(0.2793, color="#2ECC71", linestyle="--", label="Optimal Threshold (0.2793)")
plt.xlabel("Decision Threshold", fontsize=11, fontweight="bold")
plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
plt.title("Threshold Calibration & Trade-off Dynamics (LightGBM Tuned)", fontsize=13, fontweight="bold")
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_EXP / "threshold_calibration.png", dpi=300)
plt.close()

comparison_configs = [
    ("LightGBM", "Baseline", base_cols, {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 63}),
    ("LightGBM", "Improved Features (V2)", v2_cols, {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 63}),
    ("LightGBM", "Tuned (V2)", v2_cols, best_lgb_params),
    
    ("XGBoost", "Baseline", base_cols, {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6}),
    ("XGBoost", "Improved Features (V2)", v2_cols, {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6}),
    ("XGBoost", "Tuned (V2)", v2_cols, best_xgb_params),
    
    ("HistGradientBoosting", "Baseline", base_cols, {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 63}),
    ("HistGradientBoosting", "Improved Features (V2)", v2_cols, {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 63}),
    ("HistGradientBoosting", "Tuned (V2)", v2_cols, best_hgb_params)
]

summary_rows = []
all_probs = {}

for model_name, feat_ver, cols_to_use, params in comparison_configs:
    label = f"{model_name} ({feat_ver})"
    print(f"\n  Evaluating {label}...")
    
    t_start = time.time()
    if model_name == "LightGBM":
        m = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(train_v2[cols_to_use], y_train)
    elif model_name == "XGBoost":
        m = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="aucpr")
        m.fit(train_v2[cols_to_use], y_train)
    else:
        m = HistGradientBoostingClassifier(**params, random_state=42)
        m.fit(train_v2[cols_to_use], y_train)
    train_time = time.time() - t_start
    
    t_inf = time.time()
    probs = m.predict_proba(val_v2[cols_to_use])[:, 1]
    inf_time = time.time() - t_inf
    tx_per_sec = int(len(val_v2) / inf_time)
    
    all_probs[label] = probs
    pr_auc, roc_auc, f1, prec, rec, acc_opt, acc_50, thresh = evaluate_preds(y_val, probs)
    
    print(f"    PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Acc(Opt): {acc_opt*100:.2f}% | Acc(0.5): {acc_50*100:.2f}% | F1: {f1:.4f} | Train: {train_time:.1f}s | Inf: {tx_per_sec:,} tx/s")
    
    summary_rows.append({
        "Model": model_name,
        "Feature_Version": feat_ver,
        "Features_Count": len(cols_to_use),
        "PR_AUC": round(pr_auc, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Accuracy_Optimal_Thresh": round(acc_opt, 4),
        "Accuracy_Default_05": round(acc_50, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "Threshold": round(thresh, 4),
        "Training_Time_s": round(train_time, 1),
        "Inference_Speed_tx_per_s": tx_per_sec
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(REPORTS_DIR / "model_experiment_summary.csv", index=False)
with open(REPORTS_DIR / "model_experiment_summary.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

print(f"\n  Saved summary to {REPORTS_DIR / 'model_experiment_summary.csv'}")

plt.figure(figsize=(11, 5))
labels = [f"{r['Model']}\n({r['Feature_Version']})" for r in summary_rows]
scores = [r["PR_AUC"] for r in summary_rows]
colors = ["#4A90E2" if "Baseline" in l else ("#2ECC71" if "Tuned" in l else "#9B59B6") for l in labels]
bars = plt.bar(range(len(labels)), scores, color=colors, width=0.6)
plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
plt.ylabel("PR-AUC (Average Precision)", fontsize=11, fontweight="bold")
plt.title("IEEE-CIS Model Comparison: PR-AUC Across Configurations", fontsize=13, fontweight="bold")
plt.ylim(0.50, max(scores) + 0.05)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.003, f"{y:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_EXP / "prauc_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(11, 5))
acc_scores = [r["Accuracy_Optimal_Thresh"] * 100 for r in summary_rows]
bars = plt.bar(range(len(labels)), acc_scores, color="#1ABC9C", width=0.6)
plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
plt.ylabel("Accuracy (%) at Optimal Threshold", fontsize=11, fontweight="bold")
plt.title("IEEE-CIS Model Comparison: Accuracy at Optimal Decision Threshold", fontsize=13, fontweight="bold")
plt.ylim(95.0, 99.0)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.05, f"{y:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_EXP / "accuracy_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 6))
plot_models = [
    ("LightGBM (Tuned)", all_probs["LightGBM (Tuned (V2))"], "#2ECC71"),
    ("XGBoost (Tuned)", all_probs["XGBoost (Tuned (V2))"], "#E67E22"),
    ("HistGradient (Tuned)", all_probs["HistGradientBoosting (Tuned (V2))"], "#3498DB"),
    ("LightGBM (Baseline)", all_probs["LightGBM (Baseline)"], "#95A5A6")
]
for name, p_vals, col in plot_models:
    pr, rc, _ = precision_recall_curve(y_val, p_vals)
    score = average_precision_score(y_val, p_vals)
    plt.plot(rc, pr, label=f"{name} (PR-AUC = {score:.4f})", color=col, linewidth=2)

plt.xlabel("Recall", fontsize=11, fontweight="bold")
plt.ylabel("Precision", fontsize=11, fontweight="bold")
plt.title("Precision-Recall Curves on Out-of-Time Validation Split", fontsize=13, fontweight="bold")
plt.legend(frameon=True, loc="lower left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_EXP / "precision_recall_curves.png", dpi=300)
plt.close()

tracker.complete_stage("Hyperparameter Tuning", "Tuning and comparison complete across all 9 configurations.")
print("\n" + "=" * 70)
print("  TUNING, CALIBRATION & COMPARISON COMPLETE")
print("=" * 70)
