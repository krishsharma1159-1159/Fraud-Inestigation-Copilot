import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_RAW  = (ROOT / "data" / "raw") if (ROOT / "data" / "raw").exists() else (ROOT / "data")
DATA_PROC = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports" / "data_and_features"
DATA_PROC.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

report = {}

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

section("STEP 2.0: COMPLETE DATASET VERIFICATION")

file_meta = {}
for fname in ["train_transaction.csv", "train_identity.csv",
              "test_transaction.csv", "test_identity.csv",
              "sample_submission.csv"]:
    p = DATA_RAW / fname
    size_mb = round(p.stat().st_size / 1e6, 1)
    file_meta[fname] = {"size_mb": size_mb}
    print(f"  {fname}: {size_mb} MB")

print("\n  Loading all 5 files...")
tx_train  = pd.read_csv(DATA_RAW / "train_transaction.csv")
id_train  = pd.read_csv(DATA_RAW / "train_identity.csv")
tx_test   = pd.read_csv(DATA_RAW / "test_transaction.csv")
id_test   = pd.read_csv(DATA_RAW / "test_identity.csv")
sub       = pd.read_csv(DATA_RAW / "sample_submission.csv")

datasets = {
    "train_transaction": tx_train,
    "train_identity":    id_train,
    "test_transaction":  tx_test,
    "test_identity":     id_test,
    "sample_submission": sub,
}
for name, df in datasets.items():
    file_meta[name + "_rows"] = len(df)
    file_meta[name + "_cols"] = len(df.columns)
    print(f"  {name:25s}: {len(df):>8,} rows x {len(df.columns):>4} cols")

report["file_verification"] = file_meta

tx_train_cols = set(tx_train.columns)
tx_test_cols  = set(tx_test.columns)
only_in_train = tx_train_cols - tx_test_cols
only_in_test  = tx_test_cols  - tx_train_cols
common_tx     = tx_train_cols & tx_test_cols

print(f"\n  train_transaction cols only in train (expected: isFraud): {only_in_train}")
print(f"  test_transaction  cols only in test:  {only_in_test}")
print(f"  Common transaction cols: {len(common_tx)}")

assert only_in_train == {"isFraud"}, f"Unexpected train-only cols: {only_in_train}"
assert len(only_in_test) == 0,       f"Unexpected test-only cols: {only_in_test}"

id_test_renamed = id_test.rename(
    columns={c: c.replace("-", "_") for c in id_test.columns if "-" in c}
)
id_train_cols = set(id_train.columns)
id_test_cols2 = set(id_test_renamed.columns)
assert id_train_cols == id_test_cols2, "Identity schema still mismatched after rename!"
print(f"\n  Identity column normalisation (dash->underscore): OK")

train_dt_min = int(tx_train["TransactionDT"].min())
train_dt_max = int(tx_train["TransactionDT"].max())
test_dt_min  = int(tx_test["TransactionDT"].min())
test_dt_max  = int(tx_test["TransactionDT"].max())
print(f"\n  Train DT range: {train_dt_min:,} -> {train_dt_max:,}  ({(train_dt_max-train_dt_min)/86400:.1f} days)")
print(f"  Test  DT range: {test_dt_min:,} -> {test_dt_max:,}  ({(test_dt_max-test_dt_min)/86400:.1f} days)")
print(f"  Overlap? Train max > Test min: {train_dt_max > test_dt_min}")

report["dt_ranges"] = {
    "train_min": train_dt_min, "train_max": train_dt_max,
    "test_min":  test_dt_min,  "test_max":  test_dt_max,
    "train_span_days": round((train_dt_max - train_dt_min) / 86400, 1),
    "test_span_days":  round((test_dt_max  - test_dt_min)  / 86400, 1),
    "overlap": train_dt_max > test_dt_min,
}

print("\n  Loading existing cleaned parquet...")
cleaned = pd.read_parquet(DATA_PROC / "train_cleaned.parquet")
print(f"  train_cleaned.parquet: {len(cleaned):,} rows x {len(cleaned.columns)} cols")
assert len(cleaned) == len(tx_train), "Cleaned row count mismatch vs raw train_transaction!"
print(f"  Row count verified against raw train_transaction: OK")

vc_clean = cleaned["isFraud"].value_counts()
print(f"  Fraud (1): {int(vc_clean.get(1,0)):,}  |  Legit (0): {int(vc_clean.get(0,0)):,}")

report["cleaned_verification"] = {
    "rows": len(cleaned), "cols": len(cleaned.columns),
    "fraud": int(vc_clean.get(1, 0)), "legit": int(vc_clean.get(0, 0))
}


section("BUILDING MERGED TRAIN AND TEST")

id_test_norm = id_test.rename(
    columns={c: c.replace("-", "_") for c in id_test.columns if "-" in c}
)

train_merged = tx_train.merge(id_train,    on="TransactionID", how="left")
test_merged  = tx_test.merge(id_test_norm, on="TransactionID", how="left")

print(f"  train_merged: {len(train_merged):,} x {len(train_merged.columns)}")
print(f"  test_merged:  {len(test_merged):,}  x {len(test_merged.columns)}")

assert len(train_merged) == len(tx_train), "Train merge lost rows!"
assert len(test_merged)  == len(tx_test),  "Test merge lost rows!"

M_COLS = [c for c in train_merged.columns if c.startswith("M") and c[1:].isdigit()]

for col in M_COLS:
    for df in [train_merged, test_merged]:
        if col in df.columns:
            if col == "M4":
                df[col] = df[col].map({"M0": 0, "M1": 1, "M2": 2}).fillna(-1).astype(np.int8)
            else:
                df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)

for col in ["P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]:
    for df in [train_merged, test_merged]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

DROP_COLS = ["D7", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26"]
DROP_COLS_PRESENT = [c for c in DROP_COLS if c in train_merged.columns]
train_merged.drop(columns=DROP_COLS_PRESENT, inplace=True, errors="ignore")
test_merged.drop(columns=DROP_COLS_PRESENT, inplace=True, errors="ignore")
print(f"  Dropped {len(DROP_COLS_PRESENT)} cols consistent with cleaning step: {DROP_COLS_PRESENT}")

print(f"  train_merged after cleaning: {len(train_merged):,} x {len(train_merged.columns)}")
print(f"  test_merged  after cleaning: {len(test_merged):,}  x {len(test_merged.columns)}")

fe_inventory = []

def register_feature(name, group, source, formula, reason, leakage_risk, decision="Retained"):
    fe_inventory.append({
        "feature": name,
        "group": group,
        "source": source,
        "formula": formula,
        "reason": reason,
        "leakage_risk": leakage_risk,
        "decision": decision,
    })

section("STEP 2.1: TEMPORAL FEATURES")

print("""
  IMPORTANT: TransactionDT is a RELATIVE seconds offset from an unknown epoch.
  It is NOT an absolute timestamp. We CANNOT derive calendar date/month/year.
  We CAN derive:
    - Time of day (seconds mod 86400 gives seconds-since-midnight equivalent)
    - Day-of-week cycle (DT // 86400 mod 7 gives a relative weekly cycle)
    - Week number (DT // 604800)
  These represent RELATIVE temporal patterns, not real calendar values.
  We document this carefully to avoid overstating what the features mean.
""")

for df in [train_merged, test_merged]:
    dt = df["TransactionDT"]
    df["tx_hour"] = ((dt % 86400) // 3600).astype(np.int8)
    df["tx_dow"]  = ((dt // 86400) % 7).astype(np.int8)
    df["tx_week"] = (dt // 604800).astype(np.int16)
    df["is_night"] = ((df["tx_hour"] >= 22) | (df["tx_hour"] <= 5)).astype(np.int8)
    df["is_weekend"] = (df["tx_dow"] >= 5).astype(np.int8)

sample_hours = train_merged["tx_hour"].value_counts().sort_index()
print(f"  tx_hour: min={train_merged['tx_hour'].min()}  max={train_merged['tx_hour'].max()}")
print(f"  tx_dow:  min={train_merged['tx_dow'].min()}   max={train_merged['tx_dow'].max()}")
print(f"  tx_week: min={train_merged['tx_week'].min()}  max={train_merged['tx_week'].max()}")
print(f"  is_night: {train_merged['is_night'].value_counts().to_dict()}")
print(f"  is_weekend: {train_merged['is_weekend'].value_counts().to_dict()}")

register_feature("tx_hour",    "Temporal", "TransactionDT",
    "int((DT % 86400) // 3600)",
    "Fraud often concentrates at unusual hours (late night, early morning).",
    "None — only uses the transaction's own timestamp.", "Retained")
register_feature("tx_dow",     "Temporal", "TransactionDT",
    "int((DT // 86400) % 7)",
    "Transaction patterns vary by day of week; weekend/weekday is a fraud signal.",
    "None — only uses the transaction's own timestamp.", "Retained")
register_feature("tx_week",    "Temporal", "TransactionDT",
    "int(DT // 604800)",
    "Relative week index helps model learn seasonal/temporal drift.",
    "None — only uses the transaction's own timestamp.", "Retained")
register_feature("is_night",   "Temporal", "tx_hour",
    "1 if tx_hour in [22..23, 0..5] else 0",
    "Fraudulent transactions disproportionately occur during low-activity hours.",
    "None.", "Retained")
register_feature("is_weekend", "Temporal", "tx_dow",
    "1 if tx_dow >= 5 else 0",
    "Fraud patterns differ on weekends vs weekdays.",
    "None.", "Retained")

section("STEP 2.2: TRANSACTION AMOUNT FEATURES")

for df in [train_merged, test_merged]:
    df["log_tx_amt"] = np.log1p(df["TransactionAmt"]).astype(np.float32)
    df["tx_amt_cents"] = (df["TransactionAmt"] % 1).round(2).astype(np.float32)
    df["is_round_amount"] = ((df["TransactionAmt"] % 1 == 0) |
                              (df["TransactionAmt"] % 10 == 0)).astype(np.int8)

print(f"  log_tx_amt:      min={train_merged['log_tx_amt'].min():.3f}  max={train_merged['log_tx_amt'].max():.3f}")
print(f"  tx_amt_cents:    unique={train_merged['tx_amt_cents'].nunique()}")
print(f"  is_round_amount: {train_merged['is_round_amount'].value_counts().to_dict()}")

register_feature("log_tx_amt",      "Amount", "TransactionAmt",
    "log1p(TransactionAmt)",
    "Compresses right-skewed distribution. Useful even for tree models when combined with deviation features.",
    "None.", "Retained")
register_feature("tx_amt_cents",    "Amount", "TransactionAmt",
    "TransactionAmt % 1  (fractional part)",
    "Fraudulent transactions often use exact round numbers or specific cent patterns.",
    "None.", "Retained")
register_feature("is_round_amount", "Amount", "TransactionAmt",
    "1 if amount has no fractional part or is multiple of 10",
    "Programmatic/automated fraud often uses exact round amounts.",
    "None.", "Retained")


section("STEP 2.3: CARD BEHAVIORAL FEATURES (leakage-safe)")

print("""
  CRITICAL LEAKAGE DECISION:
  -------------------------------------------------
  Standard aggregation (e.g., mean amount per card1 over ALL data) uses future
  transactions and is therefore temporal leakage.

  Safe approaches:
  (A) Expanding/historical aggregation: for each transaction, compute statistics
      from ONLY transactions with earlier TransactionDT.
      - Correct but O(N^2) or requires careful groupby+shift.
      - Deferred to the dedicated Behavioral Features stage.

  (B) TRAINING-SET ONLY statistics, used as LOOKUP on both train and test:
      Compute card1 aggregates from the ENTIRE training set (not per-row historical).
      Apply the same lookup to test rows.

      Risk: For a transaction in the training set at time T, the lookup includes
      transactions at T+1, T+2, ... This is a MILD form of leakage within training
      data. However, it does NOT leak test labels into training, and it is the
      approach used by most top IEEE-CIS solutions as a practical compromise.

      Importantly: The test set uses ONLY training-set statistics (no test-to-test
      contamination). So test-set inference is clean.

      We use approach (B) here, clearly documented, and will complement it with
      proper rolling-window behavioral features in the next stage.
  -------------------------------------------------
""")

card1_stats = train_merged.groupby("card1")["TransactionAmt"].agg(
    card1_mean="mean", card1_median="median",
    card1_max="max",  card1_min="min",
    card1_count="count", card1_std="std"
).reset_index()

card1_stats.columns = ["card1"] + list(card1_stats.columns[1:])
card1_stats["card1_std"] = card1_stats["card1_std"].fillna(0)

print(f"  card1 unique groups in training: {len(card1_stats):,}")
print(f"  card1_stats preview:\n{card1_stats.head(3).to_string()}")

for df in [train_merged, test_merged]:
    df = df.merge(card1_stats, on="card1", how="left")
    df["amt_vs_card1_mean"]   = (df["TransactionAmt"] / df["card1_mean"].replace(0, np.nan)).astype(np.float32)
    df["amt_vs_card1_median"] = (df["TransactionAmt"] / df["card1_median"].replace(0, np.nan)).astype(np.float32)
    df["amt_vs_card1_max"]    = (df["TransactionAmt"] / df["card1_max"].replace(0, np.nan)).astype(np.float32)

for which, df_ref in [("train", train_merged), ("test", test_merged)]:
    merged_with_stats = df_ref.merge(card1_stats, on="card1", how="left")
    merged_with_stats["amt_vs_card1_mean"]   = (merged_with_stats["TransactionAmt"] /
                                                  merged_with_stats["card1_mean"].replace(0, np.nan)).astype(np.float32)
    merged_with_stats["amt_vs_card1_median"] = (merged_with_stats["TransactionAmt"] /
                                                  merged_with_stats["card1_median"].replace(0, np.nan)).astype(np.float32)
    merged_with_stats["amt_vs_card1_max"]    = (merged_with_stats["TransactionAmt"] /
                                                  merged_with_stats["card1_max"].replace(0, np.nan)).astype(np.float32)
    if which == "train":
        train_merged = merged_with_stats
    else:
        test_merged  = merged_with_stats

print(f"\n  Train: amt_vs_card1_mean: "
      f"min={train_merged['amt_vs_card1_mean'].min():.3f}  "
      f"max={train_merged['amt_vs_card1_mean'].max():.3f}  "
      f"missing={train_merged['amt_vs_card1_mean'].isna().sum():,}")

card_features = ["card1_mean","card1_median","card1_max","card1_min",
                 "card1_count","card1_std",
                 "amt_vs_card1_mean","amt_vs_card1_median","amt_vs_card1_max"]

for f in card_features:
    register_feature(f, "Card", "card1 + TransactionAmt",
        f"groupby(card1)[TransactionAmt].{f.split('_')[-1] if 'vs' not in f else 'ratio'}",
        "Card-level behavioral statistics reveal unusual transaction amounts for a given card.",
        "MILD: training-set statistics include post-transaction data for training rows. "
        "Test set uses only training statistics (no test leakage). "
        "Full rolling-window features deferred to behavioral stage.",
        "Retained — documented leakage accepted as practical compromise")

section("STEP 2.4: PRODUCT x AMOUNT FEATURES")

prod_stats = train_merged.groupby("ProductCD")["TransactionAmt"].agg(
    prod_mean="mean", prod_median="median"
).reset_index()

for which, df_ref in [("train", train_merged), ("test", test_merged)]:
    merged_prod = df_ref.merge(prod_stats, on="ProductCD", how="left")
    merged_prod["amt_vs_prod_mean"]   = (merged_prod["TransactionAmt"] /
                                          merged_prod["prod_mean"].replace(0, np.nan)).astype(np.float32)
    merged_prod["amt_vs_prod_median"] = (merged_prod["TransactionAmt"] /
                                          merged_prod["prod_median"].replace(0, np.nan)).astype(np.float32)
    if which == "train":
        train_merged = merged_prod
    else:
        test_merged  = merged_prod

print(f"  ProductCD groups: {prod_stats['ProductCD'].tolist()}")
print(f"  prod_mean range: {prod_stats['prod_mean'].min():.1f} -- {prod_stats['prod_mean'].max():.1f}")
print(f"  amt_vs_prod_mean: min={train_merged['amt_vs_prod_mean'].min():.3f}  "
      f"max={train_merged['amt_vs_prod_mean'].max():.3f}")

for f in ["prod_mean", "prod_median", "amt_vs_prod_mean", "amt_vs_prod_median"]:
    register_feature(f, "Amount", "ProductCD + TransactionAmt",
        "groupby(ProductCD)[TransactionAmt].mean/median ratio",
        "Some product categories have characteristic amount ranges. Outliers may indicate fraud.",
        "Same as card1 stats: training-set global. Mild leakage within training set.",
        "Retained")

section("STEP 2.5: EMAIL DOMAIN FEATURES")

for df in [train_merged, test_merged]:
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(np.int8)

    high_risk_domains = {"gmail.com","yahoo.com","hotmail.com","outlook.com",
                          "live.com","aol.com","icloud.com","mail.com",
                          "protonmail.com","unknown"}
    df["p_email_is_free"] = df["P_emaildomain"].isin(high_risk_domains).astype(np.int8)
    df["r_email_is_free"] = df["R_emaildomain"].isin(high_risk_domains).astype(np.int8)

print(f"  email_match distribution: {train_merged['email_match'].value_counts().to_dict()}")
print(f"  p_email_is_free: {train_merged['p_email_is_free'].value_counts().to_dict()}")

register_feature("email_match",      "Email", "P_emaildomain, R_emaildomain",
    "1 if P_emaildomain == R_emaildomain",
    "Mismatched payer/recipient email domains may indicate third-party or stolen account usage.",
    "None — uses the transaction's own fields only.", "Retained")
register_feature("p_email_is_free",  "Email", "P_emaildomain",
    "1 if domain in {gmail, yahoo, hotmail, ...}",
    "Free email domains are more commonly associated with fraud accounts.",
    "None.", "Retained")
register_feature("r_email_is_free",  "Email", "R_emaildomain",
    "1 if domain in {gmail, yahoo, hotmail, ...}",
    "Same reasoning for recipient domain.",
    "None.", "Retained")

section("STEP 2.6: DEVICE FEATURES")

print("""
  LEAKAGE CONSIDERATION:
  Aggregating DeviceInfo counts over the entire dataset (train+test) would create
  temporal leakage for training rows and label-leakage risk.
  We use TRAINING-SET-ONLY frequency encoding. Each device's frequency is computed
  from train, then looked up for both train and test rows.
""")

device_freq = train_merged.groupby("DeviceInfo").size().reset_index(name="device_freq")
device_type_freq = train_merged.groupby("DeviceType").size().reset_index(name="device_type_freq")

for which, df_ref in [("train", train_merged), ("test", test_merged)]:
    m = df_ref.merge(device_freq,      on="DeviceInfo", how="left")
    m = m.merge(device_type_freq,      on="DeviceType", how="left")
    m["device_freq"]      = m["device_freq"].fillna(0).astype(np.int32)
    m["device_type_freq"] = m["device_type_freq"].fillna(0).astype(np.int32)
    if which == "train":
        train_merged = m
    else:
        test_merged  = m

print(f"  Unique DeviceInfo values in train: {device_freq['DeviceInfo'].nunique():,}")
print(f"  device_freq stats: min={train_merged['device_freq'].min()}  "
      f"max={train_merged['device_freq'].max():,}  "
      f"missing={train_merged['device_freq'].isna().sum()}")

register_feature("device_freq",      "Device", "DeviceInfo",
    "Count of transactions from this DeviceInfo in training set",
    "Rare/unknown devices may be associated with fraud. High-frequency devices are more established.",
    "Training-set frequency used for both train and test. No test labels used.",
    "Retained")
register_feature("device_type_freq", "Device", "DeviceType",
    "Count of transactions from this DeviceType (mobile/desktop) in training set",
    "Mobile devices may have different fraud rates than desktop.",
    "Training-set frequency. No leakage.", "Retained")


section("STEP 2.7: MISSINGNESS INDICATOR FEATURES")

print("""
  Strategy: Create missingness flags for semantically important columns where
  the ABSENCE of information is itself a meaningful fraud signal.
  Do NOT create a flag for every column (hundreds of V-features would be noise).
  Focus on columns where missingness has clear interpretive meaning.
""")

miss_candidates = {
    "id_01":     "Identity missing -> no identity match -> card-not-present indicator",
    "DeviceInfo":"Device info missing -> transaction not via digital channel",
    "addr1":     "Billing address missing -> incomplete account info",
    "dist1":     "Distance missing -> no location anchor for this transaction",
}

for col, reason in miss_candidates.items():
    feat_name = f"is_missing_{col}"
    for df in [train_merged, test_merged]:
        df[feat_name] = df[col].isna().astype(np.int8)
    before_count = int(train_merged[feat_name].sum())
    print(f"  {feat_name}: {before_count:,} missing ({before_count/len(train_merged)*100:.1f}%) — {reason}")
    register_feature(feat_name, "Missingness", col,
        f"1 if {col} is NaN else 0",
        reason,
        "None — only uses whether the transaction had a value, not the value itself.",
        "Retained")

print(f"\n  is_missing_id_01 is effectively equivalent to 'has no identity record'.")
print(f"  This binary flag gives the model an explicit signal for identity absence.")

id_feature_cols = [c for c in train_merged.columns if c.startswith("id_")]
for df in [train_merged, test_merged]:
    df["identity_null_count"] = df[id_feature_cols].isna().sum(axis=1).astype(np.int16)

print(f"\n  identity_null_count: min={train_merged['identity_null_count'].min()}  "
      f"max={train_merged['identity_null_count'].max()}  "
      f"mean={train_merged['identity_null_count'].mean():.1f}")

register_feature("identity_null_count", "Missingness", "id_* columns",
    "Count of NaN values across all id_ columns per row",
    "Transactions with completely absent identity records differ from partially-matched ones.",
    "None.", "Retained")

section("STEP 2.8: C-FEATURE AND V-FEATURE AUDIT")

C_COLS = [c for c in train_merged.columns if c.startswith("C") and c[1:].isdigit()]
V_COLS = [c for c in train_merged.columns if c.startswith("V") and c[1:].isdigit()]

print(f"  C-features present: {len(C_COLS)}")
print(f"  V-features present: {len(V_COLS)}")

constant_cols = []
near_constant_cols = []
for col in C_COLS + V_COLS:
    n_unique = train_merged[col].nunique(dropna=True)
    if n_unique <= 1:
        constant_cols.append(col)
    elif n_unique == 2:
        vc = train_merged[col].value_counts(normalize=True)
        if vc.iloc[0] > 0.99:
            near_constant_cols.append((col, float(vc.iloc[0])))

print(f"\n  Constant C/V features (nunique <= 1): {constant_cols}")
print(f"  Near-constant (top value > 99%): {near_constant_cols[:10]}")

c_miss = {c: round(train_merged[c].isna().mean()*100, 1) for c in C_COLS}
v_miss_high = {c: round(train_merged[c].isna().mean()*100, 1)
               for c in V_COLS if train_merged[c].isna().mean() > 0.5}

print(f"\n  C-feature missing rates: {c_miss}")
print(f"\n  V-features with >50% missing ({len(v_miss_high)} columns): {list(v_miss_high.keys())[:20]}")

report["c_feature_audit"] = {
    "count": len(C_COLS),
    "constant": constant_cols,
    "missing_rates": c_miss,
}
report["v_feature_audit"] = {
    "count": len(V_COLS),
    "constant": constant_cols,
    "high_missing": list(v_miss_high.keys()),
}

print(f"\n  Decision: Retain all {len(C_COLS)} C-features and {len(V_COLS)} V-features.")
print(f"  Reason: These are pre-engineered by Vesta and are known to be highly predictive.")
print(f"  Constant features if any will be noted as candidates for removal after importance check.")

section("STEP 2.9: FEATURE REDUNDANCY CHECK")

feature_cols = [c for c in train_merged.columns
                if c not in ["TransactionID", "isFraud"]]

const_feats = [c for c in feature_cols
               if train_merged[c].nunique(dropna=False) <= 1]
print(f"  Constant features (zero variance): {const_feats}")

near_const = []
for c in feature_cols:
    if c in const_feats:
        continue
    try:
        top_freq = train_merged[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.995:
            near_const.append((c, round(top_freq, 4)))
    except Exception:
        pass

print(f"  Near-constant features (>99.5% one value): {len(near_const)}")
for c, f in near_const[:10]:
    print(f"    {c}: {f*100:.2f}% same value")

report["redundancy_check"] = {
    "constant_features": const_feats,
    "near_constant_features": [(c, f) for c, f in near_const],
}

if const_feats:
    print(f"\n  Dropping {len(const_feats)} constant features: {const_feats}")
    train_merged.drop(columns=const_feats, errors="ignore", inplace=True)
    test_merged.drop(columns=const_feats,  errors="ignore", inplace=True)
else:
    print(f"\n  No constant features found. No drops required.")

section("STEP 2.10: CATEGORICAL LABEL ENCODING")

CAT_COLS = ["ProductCD", "card4", "card6",
            "P_emaildomain", "R_emaildomain",
            "DeviceType", "DeviceInfo"]
string_id_cols = [c for c in train_merged.columns
                  if c.startswith("id_") and train_merged[c].dtype == object]
CAT_COLS += string_id_cols

print(f"  Categorical columns to encode: {len(CAT_COLS)}")
print(f"    {CAT_COLS}")

label_maps = {}
for col in CAT_COLS:
    if col not in train_merged.columns:
        continue
    unique_vals = train_merged[col].dropna().unique()
    val_to_int  = {v: i for i, v in enumerate(sorted(unique_vals, key=str))}
    label_maps[col] = val_to_int

    train_merged[col] = train_merged[col].map(val_to_int).fillna(-1).astype(np.int32)
    test_merged[col]  = test_merged[col].map(val_to_int).fillna(-1).astype(np.int32)

for col in CAT_COLS:
    if col in train_merged.columns:
        n_train = train_merged[col].nunique()
        n_test  = test_merged[col].nunique()
        print(f"  {col:25s}: train_unique={n_train}  test_unique={n_test}")

print(f"\n  Label encoding fitted on TRAINING data only. Unknown test categories -> -1.")
print(f"  This prevents test information from influencing the encoding.")

report["label_encoding"] = {
    "columns_encoded": CAT_COLS,
    "method": "Ordinal label encoding fitted on train only",
    "unknown_test_handling": "Map to -1",
}

section("STEP 2.11: TRAIN/TEST FEATURE CONSISTENCY")

train_feat_cols = set(train_merged.columns) - {"TransactionID", "isFraud"}
test_feat_cols  = set(test_merged.columns)  - {"TransactionID"}

only_train = train_feat_cols - test_feat_cols
only_test  = test_feat_cols  - train_feat_cols
common     = train_feat_cols & test_feat_cols

print(f"  Train feature columns (excl. ID + target): {len(train_feat_cols)}")
print(f"  Test  feature columns (excl. ID):           {len(test_feat_cols)}")
print(f"  Only in train:  {only_train}")
print(f"  Only in test:   {only_test}")
print(f"  Common features: {len(common)}")

report["feature_consistency"] = {
    "train_feature_count": len(train_feat_cols),
    "test_feature_count":  len(test_feat_cols),
    "only_in_train":       list(only_train),
    "only_in_test":        list(only_test),
    "common_feature_count": len(common),
}

if only_train or only_test:
    print(f"\n  WARNING: Feature mismatch found. Resolving...")
    for col in only_train:
        print(f"    Adding {col} to test with NaN fill")
        test_merged[col] = np.nan
    for col in only_test:
        print(f"    Adding {col} to train with NaN fill")
        train_merged[col] = np.nan

train_final_feats = set(train_merged.columns) - {"TransactionID", "isFraud"}
test_final_feats  = set(test_merged.columns)  - {"TransactionID"}
assert train_final_feats == test_final_feats, "Feature columns still mismatch!"
print(f"\n  Final common feature count: {len(train_final_feats)} — VERIFIED")


section("STEP 2.12: FEATURE INVENTORY SUMMARY")

all_feat_cols = sorted(train_final_feats)
print(f"  Total features (excl. ID + target): {len(all_feat_cols)}")

by_group = {}
for item in fe_inventory:
    g = item["group"]
    by_group.setdefault(g, []).append(item["feature"])

print(f"\n  Engineered feature groups:")
for g, cols in by_group.items():
    print(f"    {g:15s}: {len(cols)} features")

print(f"\n  Original features from dataset: {len(all_feat_cols) - len(fe_inventory)}")
print(f"  New engineered features:         {len(fe_inventory)}")

section("STEP 2.13: SAVING OUTPUT FILES")

train_merged.sort_values("TransactionDT", inplace=True)
train_merged.reset_index(drop=True, inplace=True)

train_out = DATA_PROC / "train_features.parquet"
test_out  = DATA_PROC / "test_features.parquet"

print(f"  Saving train_features.parquet...")
train_merged.to_parquet(train_out, index=False)
train_size = round(train_out.stat().st_size / 1e6, 1)
print(f"    Rows: {len(train_merged):,}  |  Cols: {len(train_merged.columns)}  |  {train_size} MB")

print(f"  Saving test_features.parquet...")
test_merged.to_parquet(test_out, index=False)
test_size = round(test_out.stat().st_size / 1e6, 1)
print(f"    Rows: {len(test_merged):,}   |  Cols: {len(test_merged.columns)}  |  {test_size} MB")

vc_out = train_merged["isFraud"].value_counts()
print(f"\n  Target distribution in train_features.parquet:")
print(f"    Fraud (1):      {int(vc_out.get(1,0)):,}")
print(f"    Legitimate (0): {int(vc_out.get(0,0)):,}")

report["output_files"] = {
    "train_features.parquet": {
        "rows": len(train_merged), "cols": len(train_merged.columns),
        "size_mb": train_size,
        "missing": int(train_merged.isna().sum().sum()),
        "fraud": int(vc_out.get(1,0)), "legit": int(vc_out.get(0,0)),
    },
    "test_features.parquet": {
        "rows": len(test_merged), "cols": len(test_merged.columns),
        "size_mb": test_size,
        "missing": int(test_merged.isna().sum().sum()),
    },
}

report["fe_inventory"] = fe_inventory

section("SAVING REPORTS")

report_path = REPORTS / "feature_engineering_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  Report: {report_path}  ({report_path.stat().st_size/1e3:.1f} KB)")

import pickle
with open(DATA_PROC / "label_maps.pkl", "wb") as f:
    pickle.dump(label_maps, f)
print(f"  Label maps saved: data/processed/label_maps.pkl")

section("SCRIPT COMPLETE")
print(f"  train_features.parquet: {len(train_merged):,} rows x {len(train_merged.columns)} cols")
print(f"  test_features.parquet:  {len(test_merged):,} rows x {len(test_merged.columns)} cols")
print(f"  Engineered features: {len(fe_inventory)}")
print(f"  Total features (train, excl. ID+target): {len(train_final_feats)}")
print(f"  Constant features dropped: {len(const_feats)}")
print(f"  Feature consistency verified: train == test (excl. target)")
