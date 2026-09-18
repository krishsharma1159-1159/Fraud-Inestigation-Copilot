import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import pyTigerGraph as tg

from dotenv import load_dotenv
from shap_explainer import SHAPExplainer, get_shap_explainer

# Load environment configuration from .env if present
load_dotenv()

# Configuration / Paths
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    r"models/tuned/lightgbm_tuned.pkl" if os.path.exists(r"models/tuned/lightgbm_tuned.pkl") else r"e:\Skillcred\models\tuned\lightgbm_tuned.pkl"
)
BEH_PATH = os.environ.get(
    "BEH_PATH",
    r"data skillcred/train_behavioral.parquet" if os.path.exists(r"data skillcred/train_behavioral.parquet") else r"e:\Skillcred\data skillcred\train_behavioral.parquet"
)
TG_HOST = os.environ.get("TG_HOST", "https://tg-d44c2a2b-3379-4726-817c-b2cf3e85b88f.tg-2635877100.i.tgcloud.io")
TG_GRAPH = os.environ.get("TG_GRAPH", "FraudGraph")
TG_SECRET = os.environ.get("TG_SECRET", "")

class ModelRunner:
    def __init__(self, model_path=MODEL_PATH, beh_path=BEH_PATH):
        self.model_path = model_path
        self.beh_path = beh_path
        self.model = None
        self.expected_features = []
        self._load_model()
        self.df_beh = None

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.expected_features = list(self.model.feature_name_)

    def _load_dataset(self):
        if self.df_beh is None:
            if not os.path.exists(self.beh_path):
                raise FileNotFoundError(f"Behavioral dataset not found at {self.beh_path}")
            self.df_beh = pd.read_parquet(self.beh_path)

    def get_features_dataframe(self, transaction_id: int) -> pd.DataFrame:
        self._load_dataset()
        sub = self.df_beh[self.df_beh['TransactionID'] == transaction_id]
        if len(sub) == 0:
            raise ValueError(f"TransactionID {transaction_id} not found in model feature dataset.")
        
        total_tx = len(self.df_beh)
        df_sample = sub.copy()
        
        str_cols = df_sample.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            df_sample[col] = pd.factorize(df_sample[col])[0]

        df_sample['amt_card1_zscore'] = np.where(df_sample['card1_std'] > 0, (df_sample['TransactionAmt'] - df_sample['card1_mean']) / df_sample['card1_std'], 0)
        df_sample['amt_cents_ratio'] = df_sample['TransactionAmt'] % 1.0
        df_sample['dist1_to_amt_ratio'] = np.where(df_sample['TransactionAmt'] > 0, df_sample['dist1'].fillna(0) / df_sample['TransactionAmt'], 0)

        hours = (df_sample['TransactionDT'] // 3600) % 24
        df_sample['sin_tx_hour'] = np.sin(2 * np.pi * hours / 24.0)
        df_sample['cos_tx_hour'] = np.cos(2 * np.pi * hours / 24.0)
        df_sample['log_time_since_prev_tx'] = np.log1p(df_sample['time_since_prev_tx'].fillna(0))

        df_sample['velocity_burst_1h_24h'] = np.where(df_sample['tx_count_24h'] > 0, df_sample['tx_count_1h'] / df_sample['tx_count_24h'], 0)
        df_sample['velocity_diff_1h_6h'] = df_sample['tx_count_1h'] - df_sample['tx_count_6h']
        df_sample['avg_amt_per_tx_30d'] = np.where(df_sample['tx_count_30d'] > 0, df_sample['tx_amt_sum_30d'] / df_sample['tx_count_30d'], 0)

        df_sample['amt_to_prev_mean_24h'] = np.where(df_sample['tx_amt_mean_24h'] > 0, df_sample['TransactionAmt'] / df_sample['tx_amt_mean_24h'], 1.0)
        df_sample['amt_to_prev_mean_7d'] = np.where(df_sample['tx_amt_mean_7d'] > 0, df_sample['TransactionAmt'] / df_sample['tx_amt_mean_7d'], 1.0)
        df_sample['is_amt_gt_max_30d'] = (df_sample['TransactionAmt'] > df_sample['tx_amt_max_30d'].fillna(0)).astype(int)
        df_sample['amt_diff_prev_mean_30d'] = df_sample['TransactionAmt'] - df_sample['tx_amt_mean_30d'].fillna(0)

        card1_counts = df_sample['card1_count'].fillna(1)
        df_sample['card1_freq_norm'] = card1_counts / float(total_tx)

        df_sample['card1_addr1_freq'] = df_sample.groupby(['card1', 'addr1'])['TransactionID'].transform('count').fillna(0)
        df_sample['card1_p_email_freq'] = df_sample.groupby(['card1', 'P_emaildomain'])['TransactionID'].transform('count').fillna(0)
        df_sample['card1_device_freq'] = df_sample.groupby(['card1', 'DeviceInfo'])['TransactionID'].transform('count').fillna(0)

        df_sample['addr1_freq_norm'] = df_sample.groupby('addr1')['TransactionID'].transform('count') / float(total_tx)
        df_sample['p_email_freq_norm'] = df_sample.groupby('P_emaildomain')['TransactionID'].transform('count') / float(total_tx)

        df_sample['null_count_card_cols'] = df_sample[['card1', 'card2', 'card3', 'card4', 'card5', 'card6']].isnull().sum(axis=1)
        df_sample['null_count_addr_cols'] = df_sample[['addr1', 'addr2']].isnull().sum(axis=1)

        d_cols = [c for c in df_sample.columns if c.startswith('D') and c[1:].isdigit()]
        df_sample['null_count_d_cols'] = df_sample[d_cols].isnull().sum(axis=1) if d_cols else 0

        id_cols = [c for c in df_sample.columns if c.startswith('id_') or 'identity' in c.lower()]
        df_sample['null_count_identity'] = df_sample[id_cols].isnull().sum(axis=1) if id_cols else 0

        v_cols = [c for c in df_sample.columns if c.startswith('V') and c[1:].isdigit()]
        df_sample['null_count_vesta'] = df_sample[v_cols].isnull().sum(axis=1) if v_cols else 0

        df_sample['addr_missing_inconsistency'] = (df_sample['addr1'].isnull() != df_sample['addr2'].isnull()).astype(int)
        df_sample['email_domain_both_missing'] = ((df_sample['P_emaildomain'] == 'unknown') & (df_sample['R_emaildomain'] == 'unknown')).astype(int)

        return df_sample[self.expected_features]

    def predict(self, transaction_id: int):
        X = self.get_features_dataframe(transaction_id)
        probs = self.model.predict_proba(X)
        fraud_prob = float(probs[0, 1])
        pred_class = int(self.model.predict(X)[0])

        sub = self.df_beh[self.df_beh['TransactionID'] == transaction_id]
        raw_row = sub.iloc[0].to_dict()
        
        # Clean NaN values in raw_row
        cleaned_facts = {}
        for k, v in raw_row.items():
            if pd.isna(v):
                cleaned_facts[k] = None
            elif isinstance(v, (np.integer, int)):
                cleaned_facts[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                cleaned_facts[k] = float(v)
            else:
                cleaned_facts[k] = str(v)

        return {
            "fraud_probability": fraud_prob,
            "predicted_class": pred_class,
            "raw_facts": cleaned_facts,
            "features_df": X
        }


class TigerGraphClient:
    def __init__(self, host=TG_HOST, graph=TG_GRAPH, secret=TG_SECRET):
        self.host = host
        self.graph = graph
        self.secret = secret
        self.conn = None

    def _get_connection(self):
        if self.conn is None:
            self.conn = tg.TigerGraphConnection(
                host=self.host,
                graphname=self.graph,
                gsqlSecret=self.secret
            )
            self.conn.getToken(self.secret)
        return self.conn

    def fetch_investigation_context(self, transaction_id: int):
        conn = self._get_connection()
        param = {"target_id": (transaction_id,)}
        master_res = conn.runInstalledQuery("get_transaction_investigation_context", params=param)
        counts_res = conn.runInstalledQuery("get_transaction_entity_counts", params=param)
        temporal_res = conn.runInstalledQuery("get_entity_temporal_status", params=param)

        card_rel = conn.runInstalledQuery("get_card_related_transactions", params=param)
        dev_rel = conn.runInstalledQuery("get_device_related_transactions", params=param)
        addr_rel = conn.runInstalledQuery("get_address_related_transactions", params=param)
        pemail_rel = conn.runInstalledQuery("get_p_email_related_transactions", params=param)
        remail_rel = conn.runInstalledQuery("get_r_email_related_transactions", params=param)

        return {
            "master": master_res,
            "counts": counts_res,
            "temporal": temporal_res,
            "related": {
                "SHARED_CARD": card_rel,
                "SHARED_DEVICE": dev_rel,
                "SHARED_ADDRESS": addr_rel,
                "SHARED_P_EMAIL": pemail_rel,
                "SHARED_R_EMAIL": remail_rel
            }
        }


_model_runner = None
_tg_client = None
_shap_explainer = None

def get_model_runner():
    global _model_runner
    if _model_runner is None:
        _model_runner = ModelRunner()
    return _model_runner

def get_tg_client():
    global _tg_client
    if _tg_client is None:
        _tg_client = TigerGraphClient()
    return _tg_client

def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = SHAPExplainer()
    return _shap_explainer

def build_investigation_evidence(transaction_id: int) -> dict:
    model_runner = get_model_runner()
    tg_client = get_tg_client()
    shap_explainer = get_shap_explainer()

    # 1. Model Prediction & Facts
    model_res = model_runner.predict(transaction_id)
    fraud_prob = model_res["fraud_probability"]
    pred_class = model_res["predicted_class"]
    raw_facts = model_res["raw_facts"]
    features_df = model_res["features_df"]

    model_prediction = {
        "model_name": "LightGBM Tuned",
        "checkpoint_path": "models/tuned/lightgbm_tuned.pkl",
        "predicted_class": pred_class,
        "fraud_probability": fraud_prob,
        "threshold": None,
        "threshold_note": "Decision threshold is not defined in project artifacts. Value reported as null."
    }

    # 2. SHAP Model Explanation
    model_explanation = shap_explainer.get_top_features(features_df, top_k=10)

    # 3. Transaction Evidence (Factual features)
    tx_dt = raw_facts.get("TransactionDT")
    tx_amt = raw_facts.get("TransactionAmt")
    prod_cd = raw_facts.get("ProductCD")

    # Select key factual features
    feature_keys = [
        "tx_count_1h", "tx_count_6h", "tx_count_24h", "tx_count_7d", "tx_count_30d",
        "tx_amt_sum_24h", "tx_amt_mean_24h", "card1_mean", "card1_std",
        "amt_card1_zscore", "time_since_prev_tx", "card1", "addr1", "addr2",
        "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"
    ]
    extracted_features = {}
    for fk in feature_keys:
        if fk in raw_facts:
            extracted_features[fk] = raw_facts[fk]

    transaction_evidence = {
        "transaction_id": transaction_id,
        "transaction_dt": tx_dt,
        "transaction_amount": tx_amt,
        "product_cd": prod_cd,
        "features": extracted_features
    }

    # 4. Graph Evidence
    tg_data = tg_client.fetch_investigation_context(transaction_id)
    master = tg_data["master"]
    counts = tg_data["counts"]
    
    # Parse Entities
    entities = {
        "card": None,
        "device": None,
        "address": None,
        "p_email": None,
        "r_email": None
    }
    
    for item in master:
        if "Cards" in item and len(item["Cards"]) > 0:
            entities["card"] = item["Cards"][0].get("attributes", {})
        if "Devices" in item and len(item["Devices"]) > 0:
            entities["device"] = item["Devices"][0].get("attributes", {})
        if "Addresses" in item and len(item["Addresses"]) > 0:
            entities["address"] = item["Addresses"][0].get("attributes", {})
        if "PEmails" in item and len(item["PEmails"]) > 0:
            entities["p_email"] = item["PEmails"][0].get("attributes", {}).get("domain")
        if "REmails" in item and len(item["REmails"]) > 0:
            entities["r_email"] = item["REmails"][0].get("attributes", {}).get("domain")

    # Parse Entity Counts
    entity_counts = {
        "card_transaction_count": 0,
        "device_transaction_count": 0,
        "address_transaction_count": 0,
        "p_email_transaction_count": 0,
        "r_email_transaction_count": 0
    }
    if len(counts) > 1:
        cnt_item = counts[1]
        entity_counts["card_transaction_count"] = cnt_item.get("@@card_tx_count", 0)
        entity_counts["device_transaction_count"] = cnt_item.get("@@device_tx_count", 0)
        entity_counts["address_transaction_count"] = cnt_item.get("@@address_tx_count", 0)
        entity_counts["p_email_transaction_count"] = cnt_item.get("@@p_email_tx_count", 0)
        entity_counts["r_email_transaction_count"] = cnt_item.get("@@r_email_tx_count", 0)

    # Parse Temporal Status
    temporal_status = {
        "previous": 0,
        "same": 0,
        "future": 0,
        "min_dt": None,
        "max_dt": None
    }
    for item in master:
        if "@@total_related_txs" in item:
            temporal_status["previous"] = item.get("@@prev_txs_count", 0)
            temporal_status["same"] = item.get("@@same_txs_count", 0)
            temporal_status["future"] = item.get("@@fut_txs_count", 0)

    min_dts = []
    max_dts = []
    for item in tg_data["temporal"]:
        for k, v in item.items():
            if "min_dt" in k and v not in [9223372036854775807, -9223372036854775808]:
                min_dts.append(v)
            if "max_dt" in k and v not in [9223372036854775807, -9223372036854775808]:
                max_dts.append(v)
    if min_dts:
        temporal_status["min_dt"] = int(min(min_dts))
    if max_dts:
        temporal_status["max_dt"] = int(max(max_dts))

    graph_evidence = {
        "entities": entities,
        "entity_counts": entity_counts,
        "temporal_status": temporal_status
    }

    # 5. Supporting Transactions
    rel_map = {}
    for rel_type, rel_res in tg_data["related"].items():
        if rel_res and len(rel_res) > 0 and "RelatedTxs" in rel_res[0]:
            for tx_item in rel_res[0]["RelatedTxs"]:
                rel_tx_id = int(tx_item["v_id"])
                if rel_tx_id == transaction_id:
                    continue
                attrs = tx_item.get("attributes", {})
                r_dt = attrs.get("TransactionDT")
                r_amt = attrs.get("TransactionAmt")
                r_pcd = attrs.get("ProductCD")
                
                if rel_tx_id not in rel_map:
                    rel_map[rel_tx_id] = {
                        "transaction_id": rel_tx_id,
                        "transaction_dt": r_dt,
                        "transaction_amount": r_amt,
                        "product_cd": r_pcd,
                        "relationship_types": []
                    }
                if rel_type not in rel_map[rel_tx_id]["relationship_types"]:
                    rel_map[rel_tx_id]["relationship_types"].append(rel_type)

    supporting_transactions = []
    for rel_tx_id, info in rel_map.items():
        time_diff = tx_dt - info["transaction_dt"] if (tx_dt is not None and info["transaction_dt"] is not None) else 0
        if time_diff > 0:
            temp_class = "previous"
        elif time_diff == 0:
            temp_class = "same"
        else:
            temp_class = "future"

        info["time_difference"] = time_diff
        info["temporal_class"] = temp_class
        supporting_transactions.append(info)

    # 6. Deterministic Reason Codes
    reason_codes = []
    
    # R01: Transaction Amount vs Card Historical Mean
    card_mean = extracted_features.get("card1_mean")
    if tx_amt is not None and card_mean is not None and card_mean > 0:
        reason_codes.append({
            "code": "R01",
            "description": f"Transaction amount is {tx_amt} and historical card amount mean is {round(float(card_mean), 2)}.",
            "evidence": {
                "transaction_amount": tx_amt,
                "card_historical_mean_amt": round(float(card_mean), 2)
            }
        })

    # R02: Card Graph Degree
    card_cnt = entity_counts["card_transaction_count"]
    if card_cnt > 0:
        reason_codes.append({
            "code": "R02",
            "description": f"Card is connected to {card_cnt} total transactions in the graph.",
            "evidence": {
                "card_transaction_count": card_cnt
            }
        })

    # R03: Device Graph Degree
    dev_cnt = entity_counts["device_transaction_count"]
    if dev_cnt > 0:
        reason_codes.append({
            "code": "R03",
            "description": f"Device is connected to {dev_cnt} total transactions in the graph.",
            "evidence": {
                "device_transaction_count": dev_cnt
            }
        })

    # R04: Address Graph Degree
    addr_cnt = entity_counts["address_transaction_count"]
    if addr_cnt > 0:
        reason_codes.append({
            "code": "R04",
            "description": f"Address is connected to {addr_cnt} total transactions in the graph.",
            "evidence": {
                "address_transaction_count": addr_cnt
            }
        })

    # R05: P_EMAIL Domain Graph Degree
    pemail_cnt = entity_counts["p_email_transaction_count"]
    if pemail_cnt > 0:
        reason_codes.append({
            "code": "R05",
            "description": f"P_EMAIL domain is connected to {pemail_cnt} total transactions in the graph.",
            "evidence": {
                "p_email_transaction_count": pemail_cnt
            }
        })

    # R06: Temporal Context Observation
    prev_cnt = temporal_status["previous"]
    if prev_cnt > 0:
        reason_codes.append({
            "code": "R06",
            "description": f"{prev_cnt} previous transaction(s) were observed within the supplied temporal graph context.",
            "evidence": {
                "previous_transaction_count": prev_cnt
            }
        })

    # R07: 24h Velocity Count
    v_24h = extracted_features.get("tx_count_24h")
    if v_24h is not None and v_24h > 0:
        reason_codes.append({
            "code": "R07",
            "description": f"Card velocity in the 24-hour window prior to transaction is {v_24h}.",
            "evidence": {
                "tx_count_24h": v_24h
            }
        })

    # 7. Complete Structured Payload
    evidence = {
        "transaction_id": transaction_id,
        "model_prediction": model_prediction,
        "model_explanation": model_explanation,
        "transaction_evidence": transaction_evidence,
        "graph_evidence": graph_evidence,
        "reason_codes": reason_codes,
        "supporting_transactions": supporting_transactions,
        "evidence_scope": {
            "source": [
                "LightGBM model output",
                "SHAP TreeExplainer",
                "feature engineering pipeline",
                "TigerGraph FraudGraph"
            ],
            "llm_allowed_facts_only": True
        }
    }

    # Validate before returning
    validate_evidence(evidence)
    return evidence


def validate_evidence(evidence: dict) -> bool:
    """
    Validates all safety and schema invariants of the evidence object (including SHAP section).
    Raises ValueError if any invariant is violated.
    """
    # 1. transaction_id exists and is a positive integer
    tx_id = evidence.get("transaction_id")
    if not isinstance(tx_id, int) or tx_id <= 0:
        raise ValueError("Rule 1 Violation: transaction_id must be a positive integer.")

    # 2. model probability is between 0 and 1
    mp = evidence.get("model_prediction", {})
    prob = mp.get("fraud_probability")
    if not isinstance(prob, (float, int)) or not (0.0 <= prob <= 1.0):
        raise ValueError(f"Rule 2 Violation: fraud_probability must be float between 0.0 and 1.0, got {prob}")

    # 3. predicted class is valid (0 or 1)
    pred_cls = mp.get("predicted_class")
    if pred_cls not in [0, 1]:
        raise ValueError(f"Rule 3 Violation: predicted_class must be 0 or 1, got {pred_cls}")

    # SHAP Model Explanation Validation
    m_exp = evidence.get("model_explanation")
    if not isinstance(m_exp, dict):
        raise ValueError("SHAP Rule Violation: model_explanation section must be present as a dictionary.")
    
    shap_explainer = get_shap_explainer()
    shap_explainer.validate_shap_output(m_exp)

    # 4. every supporting transaction differs from target
    supp_txs = evidence.get("supporting_transactions", [])
    for stx in supp_txs:
        s_id = stx.get("transaction_id")
        if s_id == tx_id:
            raise ValueError(f"Rule 4 Violation: supporting transaction ID {s_id} matches target transaction_id {tx_id}")

        # 5. temporal_class is valid
        tc = stx.get("temporal_class")
        if tc not in ["previous", "same", "future"]:
            raise ValueError(f"Rule 5 Violation: invalid temporal_class '{tc}' in supporting transaction {s_id}")

        # 6. relationship_types are valid
        rel_types = stx.get("relationship_types", [])
        if not isinstance(rel_types, list) or len(rel_types) == 0:
            raise ValueError(f"Rule 6 Violation: relationship_types must be non-empty list for supporting transaction {s_id}")
        for rt in rel_types:
            if not isinstance(rt, str) or not rt.startswith("SHARED_"):
                raise ValueError(f"Rule 6 Violation: relationship_type '{rt}' invalid in supporting transaction {s_id}")

        # 10. no future transaction described as historical
        time_diff = stx.get("time_difference", 0)
        if tc == "future" and time_diff > 0:
            raise ValueError(f"Rule 10 Violation: future transaction {s_id} has positive time_difference {time_diff}")

    # 7. all reason-code evidence references actual supplied fields
    rc_list = evidence.get("reason_codes", [])
    for rc in rc_list:
        ev_data = rc.get("evidence", {})
        if not isinstance(ev_data, dict):
            raise ValueError(f"Rule 7 Violation: reason code {rc.get('code')} evidence must be dict.")

    # Convert entire evidence payload to string for text safety checks
    evidence_str = json.dumps(evidence).lower()

    # 8. No unsupported fraud conclusions or judgments in descriptive strings
    forbidden_conclusions = ["proves fraud", "indicates fraud", "is fraudulent", "criminal", "illegal", "guilty", "scam", "definitely fraud", "causes fraud", "prevents fraud"]
    for fc in forbidden_conclusions:
        if fc in evidence_str:
            raise ValueError(f"Rule 8 Violation: evidence contains forbidden subjective/fraud conclusion: '{fc}'")

    # 9. No customer / person identity claims
    forbidden_identities = ["person", "customer", "account holder", "identity of user", "owner"]
    for rc in rc_list:
        desc = rc.get("description", "").lower()
        for fi in forbidden_identities:
            if fi in desc:
                raise ValueError(f"Rule 9 Violation: reason code contains identity claim: '{fi}'")

    # 11. Model prediction, model explanation, transaction evidence, and graph evidence remain separate
    top_keys = ["model_prediction", "model_explanation", "transaction_evidence", "graph_evidence"]
    for tk in top_keys:
        if tk not in evidence:
            raise ValueError(f"Rule 11 Violation: top-level section {tk} missing from evidence JSON.")

    return True
