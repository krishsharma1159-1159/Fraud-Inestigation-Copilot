import sys
import os
import json
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evidence_engine import build_investigation_evidence, validate_evidence

class TestEvidenceEngine(unittest.TestCase):

    def test_evidence_engine_populated_tx_2987004(self):
        tx_id = 2987004
        evidence = build_investigation_evidence(tx_id)

        # 1. Structure assertions
        self.assertEqual(evidence["transaction_id"], tx_id)
        self.assertIn("model_prediction", evidence)
        self.assertIn("transaction_evidence", evidence)
        self.assertIn("graph_evidence", evidence)
        self.assertIn("reason_codes", evidence)
        self.assertIn("supporting_transactions", evidence)
        self.assertIn("evidence_scope", evidence)

        # 2. Model Prediction assertions
        mp = evidence["model_prediction"]
        self.assertEqual(mp["model_name"], "LightGBM Tuned")
        self.assertTrue(0.0 <= mp["fraud_probability"] <= 1.0)
        self.assertIn(mp["predicted_class"], [0, 1])
        self.assertIsNone(mp["threshold"])

        # 3. Graph Evidence assertions for 2987004 (Populated)
        ge = evidence["graph_evidence"]
        entities = ge["entities"]
        self.assertIsNotNone(entities["card"])
        self.assertIsNotNone(entities["device"])
        self.assertIsNotNone(entities["address"])
        self.assertIsNotNone(entities["p_email"])
        # r_email may be absent/None
        
        # 4. Reason Codes & Supporting Transactions
        rc = evidence["reason_codes"]
        self.assertTrue(len(rc) > 0)
        stxs = evidence["supporting_transactions"]
        for stx in stxs:
            self.assertNotEqual(stx["transaction_id"], tx_id)
            self.assertIn(stx["temporal_class"], ["previous", "same", "future"])

        # 5. Full Validation Assertion
        self.assertTrue(validate_evidence(evidence))
        print(f"\n[PASS] Test Populated Transaction {tx_id}")

    def test_evidence_engine_sparse_tx_2987000(self):
        tx_id = 2987000
        evidence = build_investigation_evidence(tx_id)

        # 1. Structure assertions
        self.assertEqual(evidence["transaction_id"], tx_id)
        self.assertIn("model_prediction", evidence)
        self.assertIn("transaction_evidence", evidence)
        self.assertIn("graph_evidence", evidence)

        # 2. Graph Evidence assertions for 2987000 (Sparse)
        ge = evidence["graph_evidence"]
        entities = ge["entities"]
        self.assertIsNotNone(entities["card"])
        self.assertIsNotNone(entities["address"])
        self.assertIsNone(entities["device"])
        self.assertIsNone(entities["p_email"])
        self.assertIsNone(entities["r_email"])

        # 3. Full Validation Assertion
        self.assertTrue(validate_evidence(evidence))
        print(f"\n[PASS] Test Sparse Transaction {tx_id}")

    def test_evidence_validation_negative_rules(self):
        tx_id = 2987004
        base_evidence = build_investigation_evidence(tx_id)

        # Negative Test 1: Unsupported Fraud Conclusion
        bad_ev1 = json.loads(json.dumps(base_evidence))
        bad_ev1["reason_codes"].append({
            "code": "R99",
            "description": "This pattern proves fraud.",
            "evidence": {}
        })
        with self.assertRaises(ValueError):
            validate_evidence(bad_ev1)

        # Negative Test 2: Customer Identity Claim
        bad_ev2 = json.loads(json.dumps(base_evidence))
        bad_ev2["reason_codes"].append({
            "code": "R98",
            "description": "Card belongs to customer John Doe.",
            "evidence": {}
        })
        with self.assertRaises(ValueError):
            validate_evidence(bad_ev2)

        # Negative Test 3: Target ID in Supporting Transactions
        bad_ev3 = json.loads(json.dumps(base_evidence))
        bad_ev3["supporting_transactions"].append({
            "transaction_id": tx_id,
            "transaction_dt": 86506,
            "transaction_amount": 50.0,
            "product_cd": "H",
            "relationship_types": ["SHARED_CARD"],
            "time_difference": 0,
            "temporal_class": "same"
        })
        with self.assertRaises(ValueError):
            validate_evidence(bad_ev3)

        print("\n[PASS] Test Evidence Validation Negative Rules")

if __name__ == "__main__":
    unittest.main()
