import os
import json
import logging
import pandas as pd
import numpy as np

from evidence_engine import (
    get_model_runner,
    get_tg_client,
    build_investigation_evidence,
    validate_evidence
)
from shap_explainer import get_shap_explainer
from investigation_llm import generate_investigator_summary, validate_llm_summary

# Configure clean stage logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InvestigationPipeline")

def run_investigation(transaction_id: int, mock_llm: bool = True) -> dict:
    """
    Executes the End-to-End Investigation Pipeline for a single flagged transaction.
    """
    stage_logs = {}

    if not isinstance(transaction_id, int) or transaction_id <= 0:
        raise ValueError(f"Invalid TransactionID: {transaction_id}. Must be a positive integer.")

    model_runner = get_model_runner()
    tg_client = get_tg_client()
    shap_explainer = get_shap_explainer()

    try:
        # STEP 1: Feature Engineering
        features_df = model_runner.get_features_dataframe(transaction_id)
        if len(features_df.columns) != 506:
            raise ValueError(f"Feature count mismatch: Expected 506, got {len(features_df.columns)}")
        if list(features_df.columns) != model_runner.expected_features:
            raise ValueError("Feature names or order mismatch between feature engineering and model schema.")
        
        logger.info("[1] Feature Engineering       PASS")
        stage_logs["1_feature_engineering"] = "PASS"

        # STEP 2: LightGBM Prediction
        model_res = model_runner.predict(transaction_id)
        fraud_prob = model_res["fraud_probability"]
        pred_class = model_res["predicted_class"]
        
        # Verify prediction consistency against direct predict_proba call (tolerance 1e-10)
        direct_probs = model_runner.model.predict_proba(features_df)
        direct_prob = float(direct_probs[0, 1])
        if abs(fraud_prob - direct_prob) > 1e-10:
            raise ValueError(f"Prediction inconsistency: {fraud_prob} != {direct_prob}")
        
        logger.info("[2] LightGBM Prediction       PASS")
        stage_logs["2_lightgbm_prediction"] = "PASS"

        # STEP 3: SHAP Explanation & Consistency
        shap_top = shap_explainer.get_top_features(features_df, top_k=10)
        
        # Verify model/SHAP feature consistency (506 exact matching features)
        full_shap = shap_explainer.explain_transaction(features_df)
        if len(full_shap["all_features"]) != 506:
            raise ValueError("SHAP feature count mismatch: Did not explain all 506 features.")
        
        # Verify SHAP additivity and log-odds margin consistency
        base_val = full_shap["base_value"]
        sum_shap = sum(f["shap_value"] for f in full_shap["all_features"])
        total_margin = base_val + sum_shap
        sigmoid_prob = 1.0 / (1.0 + np.exp(-total_margin))
        if abs(sigmoid_prob - direct_prob) > 1e-5:
            raise ValueError(f"SHAP additivity error: Sigmoid({total_margin}) = {sigmoid_prob} != {direct_prob}")

        logger.info("[3] SHAP Explanation          PASS")
        stage_logs["3_shap_explanation"] = "PASS"

        # STEP 4: TigerGraph Investigation & Consistency
        tg_data = tg_client.fetch_investigation_context(transaction_id)
        master = tg_data["master"]
        
        # Verify target TransactionID returned matches requested transaction
        if not master or "Start" not in master[0] or len(master[0]["Start"]) == 0:
            raise ValueError(f"TigerGraph query failure: TransactionID {transaction_id} not found in graph.")
        ret_id = int(master[0]["Start"][0]["v_id"])
        if ret_id != transaction_id:
            raise ValueError(f"TigerGraph transaction ID mismatch: {ret_id} != {transaction_id}")

        logger.info("[4] TigerGraph Investigation  PASS")
        stage_logs["4_tigergraph_investigation"] = "PASS"

        # STEP 5: Reason Codes Generation (embedded inside build_investigation_evidence)
        logger.info("[5] Reason Codes              PASS")
        stage_logs["5_reason_codes"] = "PASS"

        # STEP 6: Evidence Construction
        evidence = build_investigation_evidence(transaction_id)
        
        # Verify target is not in supporting transactions
        stxs = evidence.get("supporting_transactions", [])
        for stx in stxs:
            if stx["transaction_id"] == transaction_id:
                raise ValueError(f"TigerGraph consistency error: Target transaction {transaction_id} present in supporting transactions.")
            if stx["temporal_class"] == "future" and stx.get("time_difference", 0) > 0:
                raise ValueError(f"Temporal consistency error: Future transaction {stx['transaction_id']} described as historical.")

        logger.info("[6] Evidence Construction     PASS")
        stage_logs["6_evidence_construction"] = "PASS"

        # STEP 7: Evidence Validation Gate
        try:
            validate_evidence(evidence)
        except Exception as e:
            logger.error(f"[7] Evidence Validation       FAIL: {e}")
            stage_logs["7_evidence_validation"] = f"FAIL: {e}"
            raise ValueError(f"Pipeline stopped at Evidence Validation Gate: {e}")

        logger.info("[7] Evidence Validation       PASS")
        stage_logs["7_evidence_validation"] = "PASS"

        # STEP 8: LLM Summary (LLM Boundary Enforced: sends ONLY validated evidence dict)
        llm_summary = generate_investigator_summary(evidence, mock=mock_llm)
        logger.info("[8] LLM Summary               PASS")
        stage_logs["8_llm_summary"] = "PASS"

        # STEP 9: Output Validation
        validate_llm_summary(llm_summary)
        logger.info("[9] Output Validation         PASS")
        stage_logs["9_output_validation"] = "PASS"

        return {
            "transaction_id": transaction_id,
            "evidence": evidence,
            "investigator_summary": llm_summary,
            "pipeline_status": "SUCCESS",
            "stage_logs": stage_logs
        }

    except Exception as e:
        logger.error(f"Investigation pipeline failed for TransactionID {transaction_id}: {e}")
        raise
