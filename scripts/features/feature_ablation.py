"""Feature group ablation study."""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.utils.progress_tracker import ProgressTracker

DATA_SPLIT = ROOT / "data" / "processed" / "temporal_split"
REPORTS_DIR = ROOT / "reports" / "data_and_features"
FIGURES_DIR = ROOT / "reports" / "figures" / "feature_experiments"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

tracker = ProgressTracker()
tracker.update_stage_progress(52.0, "Phase 4: Running feature group ablation study...")

print("=" * 70)
print("  PHASE 4: FEATURE GROUP ABLATION STUDY")
print("=" * 70)

print("  Loading Train V2 and Val V2 datasets...")
t0 = time.time()
train_df = pd.read_parquet(DATA_SPLIT / "train_features_v2.parquet")
val_df = pd.read_parquet(DATA_SPLIT / "val_features_v2.parquet")

y_train = train_df["isFraud"].values
y_val = val_df["isFraud"].values

with open(REPORTS_DIR / "feature_groups_v2.json", "r") as f:
    fg_meta = json.load(f)

base_cols = fg_meta["baseline_features"]
new_groups = fg_meta["new_feature_groups"]

print(f"  Data loaded in {time.time()-t0:.1f}s. Train: {train_df.shape}, Val: {val_df.shape}")

experiments = [
    ("Baseline", base_cols),
    ("+ Group A (Transaction & Cyclical)", base_cols + new_groups["Group_A_Transaction"]),
    ("+ Group B (Velocity & Burst)", base_cols + new_groups["Group_B_Velocity"]),
    ("+ Group C (Historical Amount)", base_cols + new_groups["Group_C_Amount_Behavior"]),
    ("+ Group D (Entity Combinations)", base_cols + new_groups["Group_D_Entity_Combinations"]),
    ("+ Group E (Missingness Aggregates)", base_cols + new_groups["Group_E_Missingness_Aggregates"]),
    ("+ Group F (Frequency & Rarity)", base_cols + new_groups["Group_F_Frequency_Rarity"]),
    ("+ Group G (Consistency Proxies)", base_cols + new_groups["Group_G_Consistency"]),
    ("ALL Combined (V2 Full)", fg_meta["total_v2_features"]),
]

results = []
baseline_pr_auc = None

print(f"\n  Executing {len(experiments)} controlled ablation experiments...")
print(f"  Model: LightGBM (n_estimators=100, lr=0.05, num_leaves=63, random_state=42)")
print("-" * 70)

for i, (name, feature_list) in enumerate(experiments, 1):
    print(f"\n  [{i}/{len(experiments)}] Training {name} ({len(feature_list)} features)...")
    t_start = time.time()
    
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    X_tr = train_df[feature_list]
    X_va = val_df[feature_list]
    
    model.fit(X_tr, y_train)
    t_train = time.time() - t_start
    
    y_prob = model.predict_proba(X_va)[:, 1]
    
    pr_auc = float(average_precision_score(y_val, y_prob))
    roc_auc = float(roc_auc_score(y_val, y_prob))
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[min(best_idx, len(thresholds)-1)])
    
    y_pred = (y_prob >= best_thresh).astype(int)
    f1 = float(f1_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred))
    rec = float(recall_score(y_val, y_pred))
    
    if i == 1:
        baseline_pr_auc = pr_auc
        delta_str = "0.0000 (ref)"
    else:
        delta = pr_auc - baseline_pr_auc
        delta_str = f"{delta:+.4f}"
        
    print(f"    PR-AUC: {pr_auc:.4f} ({delta_str}) | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} (P: {prec:.4f}, R: {rec:.4f}) | Train: {t_train:.1f}s")
    
    results.append({
        "experiment": name,
        "feature_count": len(feature_list),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(best_thresh, 4),
        "train_time_sec": round(t_train, 1),
        "delta_pr_auc_vs_baseline": round(pr_auc - baseline_pr_auc, 4)
    })

# Save results to CSV and JSON
df_results = pd.DataFrame(results)
csv_path = REPORTS_DIR / "feature_ablation.csv"
json_path = REPORTS_DIR / "feature_ablation.json"
df_results.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved ablation results to {csv_path} and {json_path}")

# Plot Ablation Study Chart
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#4A90E2" if r["experiment"] == "Baseline" else ("#2ECC71" if r["delta_pr_auc_vs_baseline"] > 0 else "#E74C3C") for r in results]
colors[-1] = "#9B59B6"  # Highlight ALL Combined

bars = ax.barh([r["experiment"] for r in results], [r["pr_auc"] for r in results], color=colors, height=0.6)
ax.set_xlabel("PR-AUC (Average Precision on Validation Split)", fontsize=12, fontweight="bold")
ax.set_title("IEEE-CIS Fraud Detection — Feature Group Ablation Study", fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0.50, max(r["pr_auc"] for r in results) + 0.05)

# Annotate bars with PR-AUC values and delta
for bar, r in zip(bars, results):
    w = bar.get_width()
    delta_text = f" (ref)" if r["delta_pr_auc_vs_baseline"] == 0 else f" ({r['delta_pr_auc_vs_baseline']:+.4f})"
    ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.4f}{delta_text}", va="center", ha="left", fontsize=10, fontweight="bold")

plt.tight_layout()
fig_path = FIGURES_DIR / "ablation_study.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"  Saved ablation plot to {fig_path}")

tracker.update_stage_progress(60.0, "Phase 4 Ablation Study complete.")
print("\n  Phase 4 Feature Group Ablation Study COMPLETE.")
