import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "reports" / "model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_comprehensive_evaluation_image():
    headers = [
        "Model Architecture",
        "Configuration",
        "Features",
        "Peak Tuning\nAccuracy",
        "Full Validation\nAccuracy (0.50)",
        "Calibrated Triage\nAccuracy (0.2793)",
        "Evaluation Status"
    ]
    
    rows = [
        ["LightGBM", "Tuned (V2 Champion)", "506", "98.24%", "97.56%", "97.34%", "Overall Champion"],
        ["HistGradientBoosting", "Tuned (V2)", "506", "97.92%", "97.54%", "97.16%", "Strong Runner-up"],
        ["XGBoost", "Tuned (V2)", "506", "97.68%", "97.47%", "97.25%", "Fastest Throughput"],
        ["LightGBM", "Improved (V2)", "506", "97.40%", "97.40%", "97.15%", "Improved V2 Baseline"],
        ["LightGBM", "Baseline", "480", "97.41%", "97.41%", "97.23%", "Initial Baseline"],
        ["HistGradientBoosting", "Baseline", "480", "97.40%", "97.40%", "97.12%", "Initial Baseline"],
        ["XGBoost", "Baseline", "480", "97.35%", "97.35%", "96.84%", "Initial Baseline"]
    ]

    col_widths = [0.18, 0.18, 0.08, 0.14, 0.15, 0.15, 0.15]

    fig, ax = plt.subplots(figsize=(15, 8.8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')
    
    fig.text(0.5, 0.94, "IEEE-CIS Fraud Detection — Model Evaluation Accuracy Benchmark",
             ha='center', va='center', fontsize=18, fontweight='bold', color='#0f172a', family='sans-serif')
    fig.text(0.5, 0.90, "Standardized Accuracy Evaluation across Temporal Validation Split (118,108 Transactions)",
             ha='center', va='center', fontsize=12, color='#475569', family='sans-serif')

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.28, 0.96, 0.58]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#ffffff', weight='bold', size=11)
        cell.set_height(0.09)
        cell.set_edgecolor('#334155')
        cell.set_linewidth(1.5)

    for row_idx, row in enumerate(rows, start=1):
        is_champion = (row_idx == 1)
        bg_color = '#ecfdf5' if is_champion else ('#f8fafc' if row_idx % 2 == 1 else '#ffffff')
        edge_color = '#10b981' if is_champion else '#e2e8f0'
        text_weight = 'bold' if is_champion else 'normal'
        text_color = '#065f46' if is_champion else '#1e293b'

        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(bg_color)
            cell.set_height(0.07)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1.5 if is_champion else 1.0)
            
            if is_champion and col_idx in [3, 4, 5]:
                cell.set_text_props(color='#047857', weight='bold', size=12)
            elif is_champion and col_idx == 6:
                cell.set_text_props(color='#047857', weight='bold', size=11)
            elif col_idx in [3, 4]:
                cell.set_text_props(color='#0f172a', weight='bold', size=10.5)
            else:
                cell.set_text_props(color=text_color, weight=text_weight, size=10.5)

    fig.text(0.04, 0.22, "Key Evaluation Highlights & Verified Operational Benchmarks:", 
             fontsize=12, fontweight='bold', color='#0f172a')
    
    bullet_points = [
        ("Peak Tuning Accuracy (98.24%):", "Achieved during Optuna hyperparameter search (Trial LGB_5) on validation folds."),
        ("Full Validation Benchmark (97.56%):", "Strict out-of-time temporal validation across all 118,108 transactions at default 0.50 (115,225 correct)."),
        ("Operational Triage Accuracy (97.34%):", "At calibrated threshold 0.2793, catches 25.6% more fraud incidents (1,956 vs 1,557) with 97.34% accuracy."),
        ("Champion Model Checkpoint:", "LightGBM Tuned with 506 engineered features saved at models/tuned/lightgbm_tuned.pkl.")
    ]
    
    for i, (title, desc) in enumerate(bullet_points):
        y_pos = 0.175 - (i * 0.038)
        fig.text(0.04, y_pos, f"{title}  {desc}", fontsize=10.5, color='#1e293b')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "accuracy_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(OUTPUT_DIR / "model_evaluation_accuracy_benchmark.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_comparison_image():
    headers = [
        "Model Architecture",
        "Features",
        "Peak Tuning\nAccuracy",
        "Full Validation\nAccuracy (0.50)",
        "Calibrated Triage\nAccuracy (0.2793)",
        "Operational Verdict"
    ]
    
    rows = [
        ["LightGBM (Tuned)", "506", "98.24%", "97.56%", "97.34%", "Overall Champion"],
        ["HistGradientBoosting (Tuned)", "506", "97.92%", "97.54%", "97.16%", "Strong Runner-up"],
        ["XGBoost (Tuned)", "506", "97.68%", "97.47%", "97.25%", "Fastest Inference"]
    ]

    col_widths = [0.24, 0.10, 0.16, 0.17, 0.17, 0.16]

    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')
    
    fig.text(0.5, 0.92, "IEEE-CIS Fraud Detection — Tuned Model Comparison Leaderboard",
             ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a', family='sans-serif')
    fig.text(0.5, 0.85, "Comparison of Tuned Gradient Boosted Architectures (V2 Feature Set: 506 Features)",
             ha='center', va='center', fontsize=11, color='#64748b', family='sans-serif')

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.35, 0.96, 0.45]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#ffffff', weight='bold', size=11)
        cell.set_height(0.12)
        cell.set_edgecolor('#334155')

    for row_idx, row in enumerate(rows, start=1):
        is_champion = (row_idx == 1)
        bg_color = '#ecfdf5' if is_champion else ('#f8fafc' if row_idx % 2 == 1 else '#ffffff')
        edge_color = '#10b981' if is_champion else '#cbd5e1'
        text_color = '#065f46' if is_champion else '#1e293b'

        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(bg_color)
            cell.set_height(0.10)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1.5 if is_champion else 1.0)
            if is_champion and col_idx in [2, 3, 4]:
                cell.set_text_props(color='#047857', weight='bold', size=12)
            elif is_champion:
                cell.set_text_props(color='#047857', weight='bold', size=11)
            else:
                cell.set_text_props(color=text_color, weight='normal', size=11)

    fig.text(0.5, 0.20, "Champion Model: LightGBM Tuned with 98.24% Peak Tuning Accuracy & 97.56% Full Validation Accuracy",
             ha='center', va='center', fontsize=11, fontweight='bold', color='#047857')
    fig.text(0.5, 0.12, "Model Checkpoint: models/tuned/lightgbm_tuned.pkl | Temporal Evaluation Window: 118,108 Transactions",
             ha='center', va='center', fontsize=10, color='#64748b')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_tuning_image():
    headers = [
        "Trial ID",
        "Model",
        "Key Hyperparameters",
        "Peak Tuning\nAccuracy",
        "Full Validation\nAccuracy (0.50)",
        "Calibrated Triage\nAccuracy (0.2793)",
        "Role / Verdict"
    ]
    
    rows = [
        ["LGB_5", "LightGBM", "num_leaves=150, lr=0.04, n_est=220, colsample=0.75", "98.24%", "97.56%", "97.34%", "Overall Champion"],
        ["LGB_4", "LightGBM", "num_leaves=127, lr=0.05, n_est=180, scale_pos=1.5", "98.20%", "97.48%", "97.18%", "Candidate"],
        ["LGB_3", "LightGBM", "num_leaves=127, lr=0.03, n_est=250", "98.12%", "97.45%", "97.15%", "Candidate"],
        ["HGB_3", "HistGradient", "max_iter=150, lr=0.05, max_leaf_nodes=127, l2=0.5", "97.92%", "97.54%", "97.16%", "Strong Runner-up"],
        ["XGB_3", "XGBoost", "max_depth=8, lr=0.03, n_est=200, colsample=0.70", "97.68%", "97.47%", "97.25%", "Throughput Champion"]
    ]

    col_widths = [0.08, 0.12, 0.35, 0.11, 0.11, 0.11, 0.12]

    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')
    
    fig.text(0.5, 0.93, "IEEE-CIS Fraud Detection — Hyperparameter Tuning Trials Leaderboard",
             ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a', family='sans-serif')
    fig.text(0.5, 0.86, "Optuna Hyperparameter Search Results across Top Architectural Candidates",
             ha='center', va='center', fontsize=11, color='#64748b', family='sans-serif')

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.28, 0.96, 0.52]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#ffffff', weight='bold', size=11)
        cell.set_height(0.12)
        cell.set_edgecolor('#334155')

    for row_idx, row in enumerate(rows, start=1):
        is_champion = (row_idx == 1)
        bg_color = '#ecfdf5' if is_champion else ('#f8fafc' if row_idx % 2 == 1 else '#ffffff')
        edge_color = '#10b981' if is_champion else '#cbd5e1'
        text_color = '#065f46' if is_champion else '#1e293b'

        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(bg_color)
            cell.set_height(0.08)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1.5 if is_champion else 1.0)
            if is_champion and col_idx in [3, 4, 5]:
                cell.set_text_props(color='#047857', weight='bold', size=11.5)
            elif is_champion:
                cell.set_text_props(color='#047857', weight='bold', size=10.5)
            else:
                cell.set_text_props(color=text_color, weight='normal', size=10.5)
                
            if col_idx == 2:
                cell.set_text_props(ha='left')

    fig.text(0.5, 0.16, "Trial LGB_5 selected as Pre-TigerGraph ML Champion (Peak Accuracy: 98.24% | Val Accuracy: 97.56%)",
             ha='center', va='center', fontsize=11, fontweight='bold', color='#047857')
    fig.text(0.5, 0.09, "Model Artifact: models/tuned/lightgbm_tuned.pkl | Evaluated on IEEE-CIS Temporal Validation Split",
             ha='center', va='center', fontsize=10, color='#64748b')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "hyperparameter_tuning_trials.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_learning_rate_image():
    headers = [
        "Model",
        "Variant / Trial",
        "Learning\nRate",
        "Estimators /\nIterations",
        "Operational Status & Benchmark"
    ]

    rows = [
        ["LightGBM", "Tuned Champion (LGB_5)", "0.04", "220", "Overall Champion (98.24% Peak / 97.56% Val Acc)"],
        ["HistGradientBoosting", "Tuned (HGB_3)", "0.05", "150", "Runner-up (97.92% Peak / 97.54% Val Acc)"],
        ["XGBoost", "Tuned (XGB_3)", "0.03", "200", "Fastest Inference (97.68% Peak / 97.47% Val Acc)"],
        ["LightGBM", "Initial Baseline", "0.05", "100", "Baseline Reference (97.41% Accuracy)"]
    ]

    col_widths = [0.18, 0.22, 0.12, 0.14, 0.34]

    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')

    fig.text(0.5, 0.93, "IEEE-CIS Fraud Detection — Model Learning Rate & Training Specifications",
             ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a', family='sans-serif')
    fig.text(0.5, 0.86, "Comparison of Optimized Learning Rates and Iteration Counts across Architectural Candidates",
             ha='center', va='center', fontsize=11, color='#64748b', family='sans-serif')

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.28, 0.96, 0.50]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#ffffff', weight='bold', size=11)
        cell.set_height(0.12)
        cell.set_edgecolor('#334155')

    for row_idx, row in enumerate(rows, start=1):
        is_champion = (row_idx == 1)
        bg_color = '#ecfdf5' if is_champion else ('#f8fafc' if row_idx % 2 == 1 else '#ffffff')
        edge_color = '#10b981' if is_champion else '#cbd5e1'
        text_color = '#065f46' if is_champion else '#1e293b'

        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(bg_color)
            cell.set_height(0.09)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1.5 if is_champion else 1.0)
            
            if is_champion and col_idx == 2:
                cell.set_text_props(color='#047857', weight='bold', size=12)
            elif is_champion:
                cell.set_text_props(color='#047857', weight='bold', size=10.5)
            else:
                cell.set_text_props(color=text_color, weight='normal', size=10.5)
                
            if col_idx in [0, 1, 4]:
                cell.set_text_props(ha='left')

    fig.text(0.5, 0.16, "Champion Hyperparameter Checkpoint (models/tuned/lightgbm_tuned.pkl):",
             ha='center', va='center', fontsize=11, fontweight='bold', color='#047857')
    fig.text(0.5, 0.09, "learning_rate: 0.04  |  n_estimators: 220  |  num_leaves: 150  |  colsample_bytree: 0.75  |  subsample: 0.80  |  min_child_samples: 30",
             ha='center', va='center', fontsize=10, color='#334155')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_learning_rate_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_comprehensive_evaluation_image()
    generate_comparison_image()
    generate_tuning_image()
    generate_learning_rate_image()
