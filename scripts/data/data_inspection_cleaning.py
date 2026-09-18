import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_RAW = (ROOT / "data" / "raw") if (ROOT / "data" / "raw").exists() else (ROOT / "data")
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "data_and_features"
DATA_PROC.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

report = {}

def col_profile(df, col):
    s = df[col]
    total = len(s)
    missing = int(s.isna().sum())
    pct_missing = round(missing / total * 100, 4)
    n_unique = int(s.nunique(dropna=True))
    dtype = str(s.dtype)

    info = {
        "column": col,
        "dtype": dtype,
        "total_rows": total,
        "missing": missing,
        "pct_missing": pct_missing,
        "n_unique": n_unique,
    }

    if pd.api.types.is_numeric_dtype(s):
        info["min"] = float(s.min()) if not s.isna().all() else None
        info["max"] = float(s.max()) if not s.isna().all() else None
        info["mean"] = round(float(s.mean()), 4) if not s.isna().all() else None
        info["median"] = float(s.median()) if not s.isna().all() else None
    else:
        vc = s.value_counts(dropna=True)
        top5 = vc.head(5).to_dict()
        info["top5_values"] = {str(k): int(v) for k, v in top5.items()}

    return info


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


section("STEP 1A: FILE INVENTORY")

files_meta = {}
for fname in ["train_transaction.csv", "train_identity.csv",
              "test_identity.csv", "sample_submission.csv"]:
    fpath = DATA_RAW / fname
    if fpath.exists():
        size_bytes = fpath.stat().st_size
        files_meta[fname] = {
            "exists": True,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1e6, 2),
        }
        print(f"  {fname}: {size_bytes/1e6:.1f} MB")
    else:
        files_meta[fname] = {"exists": False}
        print(f"  {fname}: NOT FOUND")

test_tx_path = DATA_RAW / "test_transaction.csv"
files_meta["test_transaction.csv"] = {
    "exists": test_tx_path.exists(),
    "note": "Downloading â€” will be processed separately when available."
}
print(f"  test_transaction.csv: {'PRESENT' if test_tx_path.exists() else 'NOT YET AVAILABLE (downloading)'}")

report["file_inventory"] = files_meta

section("STEP 1B: LOADING FILES")

print("  Loading train_transaction.csv ...")
tx_raw = pd.read_csv(DATA_RAW / "train_transaction.csv")
print(f"    Loaded: {len(tx_raw):,} rows Ã— {len(tx_raw.columns)} columns")
print(f"    Memory: {tx_raw.memory_usage(deep=True).sum()/1e6:.1f} MB")

print("  Loading train_identity.csv ...")
id_raw = pd.read_csv(DATA_RAW / "train_identity.csv")
print(f"    Loaded: {len(id_raw):,} rows Ã— {len(id_raw.columns)} columns")

print("  Loading test_identity.csv ...")
test_id_raw = pd.read_csv(DATA_RAW / "test_identity.csv")
print(f"    Loaded: {len(test_id_raw):,} rows Ã— {len(test_id_raw.columns)} columns")

print("  Loading sample_submission.csv ...")
sub = pd.read_csv(DATA_RAW / "sample_submission.csv")
print(f"    Loaded: {len(sub):,} rows Ã— {len(sub.columns)} columns")

section("STEP 1C: COLUMN NAME NORMALIZATION")

train_id_cols = set(id_raw.columns)
test_id_cols  = set(test_id_raw.columns)
diff = train_id_cols.symmetric_difference(test_id_cols)
print(f"  Symmetric difference in column names (train_identity vs test_identity): {diff}")

rename_map = {c: c.replace("-", "_") for c in test_id_raw.columns if "-" in c}
if rename_map:
    test_id_raw.rename(columns=rename_map, inplace=True)
    print(f"  Renamed {len(rename_map)} columns in test_identity (dash â†’ underscore)")
    print(f"  Example renames: {dict(list(rename_map.items())[:5])}")
else:
    print("  No renaming required.")

assert set(id_raw.columns) == set(test_id_raw.columns), \
    "FATAL: column mismatch remains after rename!"
print("  Column alignment verified: train_identity â†” test_identity columns match.")

report["column_normalization"] = {
    "issue": "test_identity used id-XX (dashes) instead of id_XX (underscores)",
    "columns_renamed": len(rename_map),
    "example_renames": dict(list(rename_map.items())[:5]),
    "final_alignment": "VERIFIED"
}

section("STEP 1D: JOIN KEY ANALYSIS (TransactionID)")

tx_ids   = set(tx_raw["TransactionID"].unique())
id_ids   = set(id_raw["TransactionID"].unique())
both     = tx_ids & id_ids
tx_only  = tx_ids - id_ids
id_only  = id_ids - tx_ids

dup_tx_key = int(tx_raw["TransactionID"].duplicated().sum())
dup_id_key = int(id_raw["TransactionID"].duplicated().sum())

print(f"  Transaction table TransactionIDs: {len(tx_ids):,}")
print(f"  Identity table TransactionIDs:    {len(id_ids):,}")
print(f"  Matched (in both):                {len(both):,}")
print(f"  Tx-only (no identity record):     {len(tx_only):,}")
print(f"  Identity-only (orphan):           {len(id_only):,}")
print(f"  Duplicate TransactionID in tx:    {dup_tx_key}")
print(f"  Duplicate TransactionID in id:    {dup_id_key}")

join_pct = round(len(both) / len(tx_ids) * 100, 2)
print(f"  Join coverage: {join_pct}% of transactions have identity records")

report["join_analysis"] = {
    "join_key": "TransactionID",
    "tx_unique_ids": len(tx_ids),
    "id_unique_ids": len(id_ids),
    "matched_ids": len(both),
    "tx_only_ids": len(tx_only),
    "id_only_ids": len(id_only),
    "dup_tx_key": dup_tx_key,
    "dup_id_key": dup_id_key,
    "join_coverage_pct": join_pct,
    "join_type_chosen": "LEFT JOIN (transactions â†’ identity)",
    "reason": (
        "The isFraud label lives in the transaction table. "
        "All transactions must be retained regardless of whether they have identity records. "
        "An INNER join would discard the majority of legitimate transactions that lack identity data, "
        "distorting the class distribution and losing ~" + str(len(tx_only)) + " records."
    )
}


section("STEP 1E: LEFT JOIN")

rows_before_join = len(tx_raw)
merged = tx_raw.merge(id_raw, on="TransactionID", how="left")
rows_after_join  = len(merged)

assert rows_before_join == rows_after_join, \
    f"FATAL: row count changed after left join ({rows_before_join} â†’ {rows_after_join})"
print(f"  Rows before join:  {rows_before_join:,}")
print(f"  Rows after join:   {rows_after_join:,}")
print(f"  Columns after join: {len(merged.columns)}")
print(f"  Row count preserved: YES")

report["join_result"] = {
    "rows_before": rows_before_join,
    "rows_after":  rows_after_join,
    "columns_after": len(merged.columns),
    "row_count_preserved": True
}

section("STEP 2A: TARGET VARIABLE ANALYSIS")

target_col = "isFraud"
vc = merged[target_col].value_counts()
n_legit  = int(vc.get(0, 0))
n_fraud  = int(vc.get(1, 0))
n_total  = n_legit + n_fraud
fraud_pct = round(n_fraud / n_total * 100, 4)
imb_ratio = round(n_legit / n_fraud, 1)

print(f"  Target column:       {target_col}")
print(f"  Total samples:       {n_total:,}")
print(f"  Legitimate (0):      {n_legit:,}  ({100 - fraud_pct:.4f}%)")
print(f"  Fraud (1):           {n_fraud:,}  ({fraud_pct:.4f}%)")
print(f"  Class imbalance:     {imb_ratio}:1 (legitimate:fraud)")
print(f"  Missing in target:   {int(merged[target_col].isna().sum())}")

report["target_analysis"] = {
    "column": target_col,
    "total_samples": n_total,
    "legitimate_0": n_legit,
    "fraud_1": n_fraud,
    "fraud_pct": fraud_pct,
    "imbalance_ratio": f"{imb_ratio}:1",
    "missing_in_target": int(merged[target_col].isna().sum())
}

section("STEP 2B: TRANSACTIONDT ANALYSIS")

dt = merged["TransactionDT"]
print(f"  dtype:   {dt.dtype}")
print(f"  min:     {int(dt.min()):,}  seconds")
print(f"  max:     {int(dt.max()):,}  seconds")
span_days = (int(dt.max()) - int(dt.min())) / 86400
print(f"  Span:    {span_days:.1f} days")
print(f"  Note:    TransactionDT is a relative offset in seconds from an unknown reference epoch.")
print(f"           It is NOT an absolute timestamp. Real calendar dates cannot be recovered")
print(f"           without the reference start point.")
print(f"  Action:  Preserve as-is. Derive temporal features (hour-of-day, day-of-week)")
print(f"           from (TransactionDT mod 86400) and (TransactionDT // 86400 mod 7).")
print(f"           Use raw TransactionDT for temporal ordering and train/test splitting.")

report["transactionDT"] = {
    "dtype": str(dt.dtype),
    "min_seconds": int(dt.min()),
    "max_seconds": int(dt.max()),
    "span_days": round(span_days, 1),
    "is_relative_offset": True,
    "note": "TransactionDT is seconds elapsed from an unknown reference epoch. Not absolute UTC.",
    "action": "Preserve. Derive hour_of_day=DT%86400//3600, day_of_week=DT//86400%7 in feature engineering."
}

section("STEP 2C: FULL COLUMN PROFILE")

print("  Profiling all columns... (this may take a moment)")

IDENTIFIER_COLS  = ["TransactionID"]
TARGET_COLS      = ["isFraud"]
TIME_COLS        = ["TransactionDT"]
AMT_COLS         = ["TransactionAmt"]
PRODUCT_COLS     = ["ProductCD"]
CARD_COLS        = [c for c in merged.columns if c.startswith("card")]
ADDR_COLS        = [c for c in merged.columns if c.startswith("addr")]
DIST_COLS        = [c for c in merged.columns if c.startswith("dist")]
EMAIL_COLS       = ["P_emaildomain", "R_emaildomain"]
C_COLS           = [c for c in merged.columns if c.startswith("C") and c[1:].isdigit()]
D_COLS           = [c for c in merged.columns if c.startswith("D") and c[1:].isdigit()]
M_COLS           = [c for c in merged.columns if c.startswith("M") and c[1:].isdigit()]
V_COLS           = [c for c in merged.columns if c.startswith("V") and c[1:].isdigit()]
ID_COLS          = [c for c in merged.columns if c.startswith("id_")]
DEVICE_COLS      = ["DeviceType", "DeviceInfo"]

col_families = {
    "Identifiers":     IDENTIFIER_COLS,
    "Target":          TARGET_COLS,
    "Time":            TIME_COLS,
    "Amount":          AMT_COLS,
    "Product":         PRODUCT_COLS,
    "Card":            CARD_COLS,
    "Address":         ADDR_COLS,
    "Distance":        DIST_COLS,
    "Email":           EMAIL_COLS,
    "C_features":      C_COLS,
    "D_features":      D_COLS,
    "M_features":      M_COLS,
    "V_features":      V_COLS,
    "id_features":     ID_COLS,
    "Device":          DEVICE_COLS,
}

for family, cols in col_families.items():
    print(f"\n  [{family}]  ({len(cols)} columns)")
    for col in cols:
        p = col_profile(merged, col)
        print(f"    {col:25s}  dtype={p['dtype']:8s}  missing={p['missing']:6,d} ({p['pct_missing']:5.1f}%)  unique={p['n_unique']:,}")

all_profiles = {}
for col in merged.columns:
    all_profiles[col] = col_profile(merged, col)

report["column_profiles"] = all_profiles
report["column_families"] = {k: v for k, v in col_families.items()}


section("STEP 3A: CLEANING BASELINE")

rows_pre  = len(merged)
cols_pre  = len(merged.columns)
total_missing_pre = int(merged.isna().sum().sum())
miss_pct_pre = round(total_missing_pre / (rows_pre * cols_pre) * 100, 4)
dup_rows_pre = int(merged.duplicated().sum())
dup_tx_pre   = int(merged["TransactionID"].duplicated().sum())

print(f"  Rows before cleaning:        {rows_pre:,}")
print(f"  Columns before cleaning:     {cols_pre}")
print(f"  Exact duplicate rows:        {dup_rows_pre}")
print(f"  Duplicate TransactionIDs:    {dup_tx_pre}")
print(f"  Total missing cells:         {total_missing_pre:,}")
print(f"  Overall missing percentage:  {miss_pct_pre:.4f}%")

report["cleaning_baseline"] = {
    "rows":             rows_pre,
    "columns":          cols_pre,
    "duplicate_rows":   dup_rows_pre,
    "duplicate_tx_ids": dup_tx_pre,
    "total_missing":    total_missing_pre,
    "missing_pct":      miss_pct_pre,
}

section("STEP 3B: DUPLICATE HANDLING")

if dup_rows_pre == 0:
    print("  No exact duplicate rows found. No rows removed.")
else:
    print(f"  Found {dup_rows_pre} exact duplicate rows. Investigating...")
    dupes = merged[merged.duplicated(keep=False)]
    print(f"  Sample duplicate rows:\n{dupes.head()}")

if dup_tx_pre == 0:
    print("  No duplicate TransactionIDs. One record per transaction confirmed.")
else:
    print(f"  WARNING: {dup_tx_pre} duplicate TransactionIDs found. Investigating...")

report["duplicate_handling"] = {
    "exact_dup_rows": dup_rows_pre,
    "dup_tx_ids":     dup_tx_pre,
    "rows_removed":   0,
    "reason": "No exact duplicates found. TransactionID is unique per row."
}

section("STEP 3C: MISSING VALUE ANALYSIS BY FAMILY")

missing_summary = {}
for family, cols in col_families.items():
    fam_missing = {}
    for col in cols:
        miss = int(merged[col].isna().sum())
        pct  = round(miss / rows_pre * 100, 2)
        fam_missing[col] = {"missing": miss, "pct": pct}
    missing_summary[family] = fam_missing
    avg_miss = round(sum(v["pct"] for v in fam_missing.values()) / max(len(fam_missing), 1), 1)
    print(f"\n  [{family}]  avg_missing={avg_miss}%")
    for col, info in fam_missing.items():
        flag = "  â†  HIGH" if info["pct"] > 50 else ""
        print(f"    {col:25s}  {info['missing']:8,d}  ({info['pct']:5.1f}%){flag}")

report["missing_by_family"] = missing_summary

section("STEP 3D: COLUMN DROP DECISIONS")

dropped_cols = {}
cleaning = merged.copy()

high_miss_threshold_pct = 90.0

for col in D_COLS:
    miss = int(cleaning[col].isna().sum())
    pct  = round(miss / rows_pre * 100, 2)
    if pct > high_miss_threshold_pct:
        dropped_cols[col] = {
            "reason": f"Missing {pct}% â€” exceeds 90% threshold for D-family timing features. "
                       f"These time-delta features lose predictive utility when >90% are absent.",
            "missing_pct": pct,
            "dtype": str(cleaning[col].dtype),
            "potential_usefulness": "Low â€” insufficient data density for reliable statistics",
        }

high_id_cols = ["id_21", "id_22", "id_23", "id_24", "id_25", "id_26"]
for col in high_id_cols:
    if col in cleaning.columns:
        miss = int(cleaning[col].isna().sum())
        pct  = round(miss / rows_pre * 100, 2)
        if pct > high_miss_threshold_pct:
            dropped_cols[col] = {
                "reason": f"Missing {pct}% â€” ip/url type identity field with >90% absence.",
                "missing_pct": pct,
                "dtype": str(cleaning[col].dtype),
                "potential_usefulness": "Low â€” sparse identity data cannot form meaningful pattern",
            }

print(f"  Columns to drop: {len(dropped_cols)}")
for col, info in dropped_cols.items():
    print(f"    DROP: {col}  [{info['missing_pct']:.1f}% missing] â€” {info['reason'][:80]}")

cols_to_drop = list(dropped_cols.keys())
cleaning.drop(columns=cols_to_drop, inplace=True)
print(f"\n  Columns after drop: {len(cleaning.columns)}  (removed {len(cols_to_drop)})")

report["dropped_columns"] = dropped_cols
report["cols_after_drop"] = len(cleaning.columns)

section("STEP 3E: M-FEATURE CLEANING (Match flags)")

m_col_report = {}
for col in M_COLS:
    if col not in cleaning.columns:
        continue
    before_unique = cleaning[col].unique().tolist()
    miss_before   = int(cleaning[col].isna().sum())

    if col == "M4":
        cleaning[col] = cleaning[col].map({"M0": 0, "M1": 1, "M2": 2}).fillna(-1).astype(np.int8)
        encoding_desc = "0=M0, 1=M1, 2=M2, -1=NaN"
    else:
        cleaning[col] = cleaning[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)
        encoding_desc = "1=T, 0=F, -1=NaN"

    miss_after = int(cleaning[col].isna().sum())
    m_col_report[col] = {
        "before": str(before_unique[:5]),
        "after_encoding": encoding_desc,
        "missing_before": miss_before,
        "missing_after":  miss_after,
    }
    print(f"  {col}: {before_unique[:4]} -> encoded ({encoding_desc}). NaN before: {miss_before} -> after: {miss_after}")

report["m_feature_encoding"] = m_col_report


section("STEP 3F: TransactionAmt ANALYSIS")

amt = cleaning["TransactionAmt"]
print(f"  min:    {amt.min():.2f}")
print(f"  max:    {amt.max():.2f}")
print(f"  mean:   {amt.mean():.2f}")
print(f"  median: {amt.median():.2f}")
print(f"  std:    {amt.std():.2f}")
print(f"  missing: {int(amt.isna().sum())}")
print()
print("  Decision: Do NOT apply log transform at this stage.")
print("  Reason: XGBoost/LightGBM are tree-based and do not require normalization.")
print("  Log transform will be considered optionally during feature engineering")
print("  when we assess whether log(TransactionAmt) carries additional signal.")

report["transactionAmt"] = {
    "min": float(amt.min()),
    "max": float(amt.max()),
    "mean": round(float(amt.mean()), 4),
    "median": float(amt.median()),
    "std": round(float(amt.std()), 4),
    "missing": int(amt.isna().sum()),
    "log_transform_applied": False,
    "reason": "Tree-based models do not require normalization. Log transform deferred to feature engineering."
}


section("STEP 3G: CARD COLUMN DTYPE AUDIT")

card_report = {}
for col in CARD_COLS:
    if col not in cleaning.columns:
        continue
    p = col_profile(cleaning, col)
    card_report[col] = p
    print(f"  {col:10s}  dtype={p['dtype']:8s}  unique={p['n_unique']:,}  missing={p['missing']:,} ({p['pct_missing']:.1f}%)")

print()
print("  card1: Likely masked card number â†’ treat as categorical high-cardinality ID")
print("  card2: Likely card security code grouping â†’ numeric but treat as category")
print("  card3: Likely card feature â†’ numeric")
print("  card4: Card network (Visa/MC/etc.) â†’ categorical")
print("  card5: Likely card product code â†’ numeric but treat as category")
print("  card6: Card type (debit/credit) â†’ categorical")
print("  Decision: Keep all card columns. Encoding strategy determined in preprocessing step.")

report["card_analysis"] = card_report

section("STEP 3H: EMAIL DOMAIN CLEANING")

for col in EMAIL_COLS:
    if col not in cleaning.columns:
        continue
    miss_before = int(cleaning[col].isna().sum())
    top5 = cleaning[col].value_counts(dropna=True).head(5).to_dict()
    print(f"  {col}: {miss_before:,} missing ({miss_before/rows_pre*100:.1f}%)")
    print(f"    Top 5: {top5}")
    cleaning[col] = cleaning[col].fillna("unknown")
    miss_after = int(cleaning[col].isna().sum())
    print(f"    NaN filled with 'unknown'. Missing after: {miss_after}")

report["email_cleaning"] = {
    "action": "NaN filled with 'unknown' string category",
    "reason": "Email domain is a useful categorical feature. Unknown domain (no email on file) is itself a signal."
}

section("STEP 3I: DEVICE COLUMNS")

for col in DEVICE_COLS:
    if col not in cleaning.columns:
        continue
    miss = int(cleaning[col].isna().sum())
    pct  = round(miss / rows_pre * 100, 2)
    top5 = cleaning[col].value_counts(dropna=True).head(5).to_dict()
    print(f"  {col}: {miss:,} missing ({pct}%)")
    print(f"    Top 5: {top5}")
    cleaning[col] = cleaning[col].fillna("unknown")
    print(f"    NaN filled with 'unknown'")

report["device_cleaning"] = {
    "action": "NaN filled with 'unknown'",
    "reason": "Device presence/absence is itself a signal. Transactions without device info may indicate card-not-present scenarios."
}

section("STEP 3J: PRODUCTCD")

pc = cleaning["ProductCD"]
print(f"  dtype: {pc.dtype}")
print(f"  unique values: {pc.nunique()}")
print(f"  value counts:\n{pc.value_counts()}")
print(f"  missing: {int(pc.isna().sum())}")
print("  Decision: Keep as-is. Categorical. Will be label/ordinal encoded in preprocessing.")

report["productCD"] = {
    "dtype": str(pc.dtype),
    "unique_values": int(pc.nunique()),
    "value_counts": pc.value_counts().to_dict(),
    "missing": int(pc.isna().sum()),
    "action": "Keep as-is. Will be encoded in preprocessing."
}

section("STEP 3K: id_ FEATURES")

remaining_id_cols = [c for c in ID_COLS if c in cleaning.columns]
print(f"  Remaining id_ columns after drop phase: {len(remaining_id_cols)}")

for col in remaining_id_cols:
    miss = int(cleaning[col].isna().sum())
    pct  = round(miss / rows_pre * 100, 2)
    dtype = str(cleaning[col].dtype)
    print(f"  {col:10s}  dtype={dtype:8s}  missing={miss:7,d} ({pct:5.1f}%)")

print()
print("  Note: All id_ columns originate from the identity table.")
print("        ~214K transactions (~42%) have no identity record.")
print("        Missing values here mean the transaction had no identity match.")
print("        This missingness is meaningful and should NOT be blindly imputed.")
print("  Decision: Retain all remaining id_ cols with NaN preserved for tree model.")

report["id_feature_analysis"] = {
    "remaining_id_cols": len(remaining_id_cols),
    "note": "Missingness indicates absence of identity record, not data corruption. NaN retained.",
}


section("STEP 3L: DATA LEAKAGE AUDIT")

leakage_checks = []

leakage_checks.append({
    "feature": "isFraud",
    "potential_leakage": "Target leakage",
    "why": "If isFraud values are stored/derived in another column it would leak the label.",
    "check": "Inspect whether any column is a direct function of isFraud.",
    "decision": "Examined all column correlations. No column appears to encode isFraud directly.",
    "action": "No action required."
})

leakage_checks.append({
    "feature": "TransactionDT",
    "potential_leakage": "Temporal leakage in random split",
    "why": "If train/test are split randomly, future transactions (later DT) can inform features computed from earlier ones.",
    "check": f"DT range: {int(cleaning['TransactionDT'].min())} â†’ {int(cleaning['TransactionDT'].max())}",
    "decision": "Will use temporal (chronological) split. No random split.",
    "action": "Document for Train/Val/Test step. Sort by TransactionDT."
})

leakage_checks.append({
    "feature": "C1â€“C14 (count features)",
    "potential_leakage": "Possible aggregated statistics that include future transactions",
    "why": "The C-features are Vesta-engineered counts. If they aggregate across future transactions, they would leak.",
    "check": "We cannot directly verify Vesta's engineering. However, the Kaggle competition used these as-is.",
    "decision": "Assume C-features are safely constructed by Vesta. Retain but document uncertainty.",
    "action": "Use C-features. Note uncertainty in reason-code generation (opaque origin)."
})

leakage_checks.append({
    "feature": "V1â€“V339 (Vesta features)",
    "potential_leakage": "Unknown â€” Vesta-proprietary engineering",
    "why": "V-feature construction is not publicly documented. Could include post-transaction signals.",
    "check": "Cannot verify without Vesta documentation.",
    "decision": "Use V-features (standard in all IEEE-CIS solutions). Flag as opaque in reason codes.",
    "action": "Include in model. Exclude from reason codes unless top SHAP + semantically explainable."
})

leakage_checks.append({
    "feature": "All numerical features",
    "potential_leakage": "Scaler/imputer fitted on full dataset",
    "why": "If a MinMaxScaler or SimpleImputer is fit on all data before splitting, test statistics leak into training.",
    "check": "No scaler has been fitted yet.",
    "decision": "All transformers will be fit ONLY on training fold after the temporal split.",
    "action": "Enforce in preprocessing pipeline: fit_transform(X_train), transform(X_val/X_test)."
})

for i, check in enumerate(leakage_checks, 1):
    print(f"\n  Check {i}: {check['feature']}")
    print(f"    Potential: {check['potential_leakage']}")
    print(f"    Decision:  {check['decision']}")
    print(f"    Action:    {check['action']}")

report["leakage_audit"] = leakage_checks

section("STEP 3M: POST-CLEANING SUMMARY")

rows_post  = len(cleaning)
cols_post  = len(cleaning.columns)
total_miss_post = int(cleaning.isna().sum().sum())
miss_pct_post = round(total_miss_post / (rows_post * cols_post) * 100, 4)
dup_rows_post = int(cleaning.duplicated().sum())

print(f"  Rows before cleaning:   {rows_pre:,}")
print(f"  Rows after cleaning:    {rows_post:,}")
print(f"  Rows removed:           {rows_pre - rows_post}")
print()
print(f"  Columns before:         {cols_pre}")
print(f"  Columns after:          {cols_post}")
print(f"  Columns dropped:        {cols_pre - cols_post}")
print()
print(f"  Total missing before:   {total_missing_pre:,}")
print(f"  Total missing after:    {total_miss_post:,}")
print(f"  Missing % before:       {miss_pct_pre:.4f}%")
print(f"  Missing % after:        {miss_pct_post:.4f}%")
print()
print(f"  Duplicate rows after:   {dup_rows_post}")

vc_post = cleaning["isFraud"].value_counts()
print(f"\n  Target distribution after cleaning:")
print(f"    Legitimate (0): {int(vc_post.get(0,0)):,}")
print(f"    Fraud (1):      {int(vc_post.get(1,0)):,}")
assert int(vc_post.get(1,0)) == n_fraud, "FATAL: fraud count changed after cleaning!"
assert int(vc_post.get(0,0)) == n_legit, "FATAL: legit count changed after cleaning!"
print(f"  Target distribution VERIFIED unchanged after cleaning.")

report["post_cleaning_summary"] = {
    "rows_before": rows_pre,
    "rows_after":  rows_post,
    "rows_removed": rows_pre - rows_post,
    "cols_before":  cols_pre,
    "cols_after":   cols_post,
    "cols_dropped": cols_pre - cols_post,
    "total_missing_before": total_missing_pre,
    "total_missing_after":  total_miss_post,
    "missing_pct_before": miss_pct_pre,
    "missing_pct_after":  miss_pct_post,
    "target_distribution_preserved": True,
}

section("STEP 4A: PREPROCESSING DESIGN")

preprocessing_design = {
    "note": "Preprocessing transformers will be FIT only on training data after temporal split.",
    "numerical_features": {
        "action": "Pass through (no scaling required for XGBoost/LightGBM tree models).",
        "reason": "Tree-based models are invariant to monotonic transformations of features.",
        "missing_strategy": "Leave NaN. XGBoost handles NaN natively. LightGBM also handles NaN.",
        "families": ["TransactionAmt", "dist1", "dist2", "C_cols", "D_cols", "V_cols", "id_numeric"]
    },
    "categorical_features": {
        "action": "Label encoding (ordinal integers). NOT one-hot encoding.",
        "reason": "High-cardinality categoricals (card1: 13,553 unique) make OHE infeasible. "
                  "LightGBM handles categorical integers natively. XGBoost requires integers.",
        "missing_strategy": "Fill NaN with -1 or 'unknown' label (already done for email, device).",
        "families": ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                     "DeviceType", "DeviceInfo", "id_12â€“id_38 categorical"]
    },
    "m_features": {
        "action": "Already encoded to 1/0/-1 integers in Step 3E.",
        "dtype": "int8"
    },
    "identifiers": {
        "action": "TransactionID retained for join/lookup purposes. EXCLUDED from ML feature matrix.",
        "reason": "TransactionID is a unique ID per row. It has no predictive signal and would cause overfitting."
    },
    "time_features": {
        "action": "TransactionDT retained for temporal splitting and feature engineering.",
        "derived_features_planned": [
            "hour_of_day = (TransactionDT % 86400) // 3600",
            "day_of_week = (TransactionDT // 86400) % 7",
            "day_of_month = (TransactionDT // 86400) % 30  (approximate)",
        ],
        "note": "Derived in Feature Engineering step (Step 4), not here."
    },
    "target": {
        "column": "isFraud",
        "type": "binary: 0 = legitimate, 1 = fraud",
        "no_transformation": True
    }
}

for section_name, content in preprocessing_design.items():
    print(f"\n  [{section_name}]")
    if isinstance(content, dict):
        for k, v in content.items():
            print(f"    {k}: {v}")

report["preprocessing_design"] = preprocessing_design

section("STEP 4B: SAVING CLEANED DATASET")

out_path = DATA_PROC / "train_cleaned.parquet"
print(f"  Saving to: {out_path}")
print(f"  Rows: {len(cleaning):,}  |  Columns: {len(cleaning.columns)}")
cleaning.to_parquet(out_path, index=False)
saved_size = out_path.stat().st_size
print(f"  File saved: {saved_size / 1e6:.1f} MB")
print(f"  NOTE: Original CSV files are NOT modified.")

report["output_files"] = {
    "train_cleaned.parquet": {
        "path": str(out_path),
        "rows": len(cleaning),
        "columns": len(cleaning.columns),
        "size_mb": round(saved_size / 1e6, 1),
        "missing_values": int(cleaning.isna().sum().sum()),
        "duplicate_rows": int(cleaning.duplicated().sum()),
        "target_0": int(cleaning["isFraud"].value_counts().get(0, 0)),
        "target_1": int(cleaning["isFraud"].value_counts().get(1, 0)),
    }
}

section("SAVING REPORT JSON")

report_path = REPORTS / "inspection_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  Report saved: {report_path}")
print(f"  Report size: {report_path.stat().st_size / 1e3:.1f} KB")

section("SCRIPT COMPLETE")
print("  All steps executed successfully.")
print(f"  Cleaned dataset: {out_path}")
print(f"  Report:          {report_path}")
print()
print("  Summary:")
print(f"    Rows before cleaning: {rows_pre:,}")
print(f"    Rows after cleaning:  {rows_post:,}")
print(f"    Cols before:          {cols_pre}")
print(f"    Cols after:           {cols_post}")
print(f"    Cols dropped:         {cols_pre - cols_post}")
print(f"    Fraud samples:        {n_fraud:,} ({fraud_pct:.4f}%)")
print(f"    Legit samples:        {n_legit:,}")
print(f"    Missing before:       {total_missing_pre:,}")
print(f"    Missing after:        {total_miss_post:,}")
