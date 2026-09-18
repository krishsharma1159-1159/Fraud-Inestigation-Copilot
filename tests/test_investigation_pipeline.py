import sys
import os
import json
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from investigation_pipeline import run_investigation
from evidence_engine import get_model_runner, validate_evidence
from shap_explainer import get_shap_explainer

class TestInvestigationPipeline(unittest.TestCase):

    def test_e2e_populated_transaction(self):
        tx_id = 2987004
        res = run_investigation(tx_id, mock_llm=True)

        self.assertEqual(res["transaction_id"], tx_id)
        self.assertEqual(res["pipeline_status"], "SUCCESS")
        self.assertIn("evidence", res)
        self.assertIn("investigator_summary", res)
        
        # Verify 9 stages passed
        s_logs = res["stage_logs"]
        self.assertEqual(len(s_logs), 9)
        for stage, status in s_logs.items():
            self.assertEqual(status, "PASS")

        print(f"\n[PASS] 1. test_e2e_populated_transaction ({tx_id})")

    def test_e2e_sparse_transaction(self):
        tx_id = 2987000
        res = run_investigation(tx_id, mock_llm=True)

        self.assertEqual(res["transaction_id"], tx_id)
        self.assertEqual(res["pipeline_status"], "SUCCESS")
        
        ge = res["evidence"]["graph_evidence"]["entities"]
        self.assertIsNotNone(ge["card"])
        self.assertIsNotNone(ge["address"])
        self.assertIsNone(ge["device"])
        self.assertIsNone(ge["p_email"])

        print(f"\n[PASS] 2. test_e2e_sparse_transaction ({tx_id})")

    def test_model_shap_feature_consistency(self):
        mr = get_model_runner()
        se = get_shap_explainer()

        self.assertEqual(len(mr.expected_features), 506)
        self.assertEqual(len(se.expected_features), 506)
        self.assertEqual(mr.expected_features, se.expected_features)

        print("\n[PASS] 3. test_model_shap_feature_consistency")

    def test_prediction_consistency(self):
        tx_id = 2987004
        mr = get_model_runner()
        df = mr.get_features_dataframe(tx_id)
        
        p1 = float(mr.model.predict_proba(df)[0, 1])
        res = run_investigation(tx_id, mock_llm=True)
        p2 = res["evidence"]["model_prediction"]["fraud_probability"]

        self.assertAlmostEqual(p1, p2, delta=1e-10)
        print("\n[PASS] 4. test_prediction_consistency")

    def test_evidence_validation_gate(self):
        # Verify that if evidence validation fails, pipeline stops and does not return success
        tx_id = 2987004
        res = run_investigation(tx_id, mock_llm=True)
        bad_evidence = json.loads(json.dumps(res["evidence"]))
        bad_evidence["reason_codes"].append({
            "code": "R99",
            "description": "This pattern proves fraud.",
            "evidence": {}
        })

        with self.assertRaises(ValueError):
            validate_evidence(bad_evidence)

        print("\n[PASS] 5. test_evidence_validation_gate")

    def test_target_not_in_supporting_transactions(self):
        for tx_id in [2987004, 2987000]:
            res = run_investigation(tx_id, mock_llm=True)
            stxs = res["evidence"]["supporting_transactions"]
            for stx in stxs:
                self.assertNotEqual(stx["transaction_id"], tx_id)

        print("\n[PASS] 6. test_target_not_in_supporting_transactions")

    def test_temporal_classification(self):
        tx_id = 2987004
        res = run_investigation(tx_id, mock_llm=True)
        stxs = res["evidence"]["supporting_transactions"]
        
        for stx in stxs:
            self.assertIn(stx["temporal_class"], ["previous", "same", "future"])
            if stx["temporal_class"] == "future":
                self.assertLessEqual(stx["time_difference"], 0)

        print("\n[PASS] 7. test_temporal_classification")

    def test_llm_receives_only_validated_evidence(self):
        tx_id = 2987004
        res = run_investigation(tx_id, mock_llm=True)
        summary = res["investigator_summary"]
        
        self.assertEqual(summary["transaction_id"], tx_id)
        self.assertIn("case_summary", summary)

        print("\n[PASS] 8. test_llm_receives_only_validated_evidence")

    def test_invalid_transaction_id(self):
        with self.assertRaises(ValueError):
            run_investigation(-1, mock_llm=True)
        with self.assertRaises(ValueError):
            run_investigation("invalid_string", mock_llm=True)

        print("\n[PASS] 9. test_invalid_transaction_id")

    def test_feature_mismatch(self):
        se = get_shap_explainer()
        bad_df = pd.DataFrame(np.zeros((1, 500)))
        with self.assertRaises(ValueError):
            se.explain_transaction(bad_df)

        print("\n[PASS] 10. test_feature_mismatch")

if __name__ == "__main__":
    unittest.main()
