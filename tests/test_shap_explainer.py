import sys
import os
import json
import unittest
import pandas as pd
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from shap_explainer import SHAPExplainer
from evidence_engine import ModelRunner

class TestSHAPExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.explainer = SHAPExplainer()
        cls.model_runner = ModelRunner()

    def test_explainer_initialization(self):
        self.assertEqual(self.explainer.explainer.__class__.__name__, "TreeExplainer")
        self.assertEqual(len(self.explainer.expected_features), 506)
        self.assertIsNotNone(self.explainer.base_value)
        self.assertEqual(self.explainer.output_space, "raw margin (log-odds)")

    def test_explain_populated_tx_2987004(self):
        tx_id = 2987004
        features_df = self.model_runner.get_features_dataframe(tx_id)
        
        shap_res = self.explainer.get_top_features(features_df, top_k=10)
        
        self.assertEqual(shap_res["explainer"], "SHAP TreeExplainer")
        self.assertEqual(shap_res["output_space"], "raw margin (log-odds)")
        self.assertIsNotNone(shap_res["base_value"])
        
        top_feats = shap_res["top_features"]
        self.assertEqual(len(top_feats), 10)
        
        prev_abs = float('inf')
        for item in top_feats:
            self.assertIn(item["feature"], self.explainer.expected_features)
            self.assertIsInstance(item["shap_value"], (float, int))
            self.assertIsInstance(item["absolute_shap_value"], (float, int))
            self.assertAlmostEqual(item["absolute_shap_value"], abs(item["shap_value"]), places=5)
            
            expected_dir = "increases_model_output" if item["shap_value"] > 0 else "decreases_model_output"
            self.assertEqual(item["direction"], expected_dir)
            
            self.assertLessEqual(item["absolute_shap_value"], prev_abs + 1e-5)
            prev_abs = item["absolute_shap_value"]

        print(f"\n[PASS] Test SHAP Populated Transaction {tx_id}")

    def test_explain_sparse_tx_2987000(self):
        tx_id = 2987000
        features_df = self.model_runner.get_features_dataframe(tx_id)
        
        shap_res = self.explainer.get_top_features(features_df, top_k=10)
        top_feats = shap_res["top_features"]
        self.assertEqual(len(top_feats), 10)
        
        self.assertTrue(self.explainer.validate_shap_output(shap_res))
        print(f"\n[PASS] Test SHAP Sparse Transaction {tx_id}")

    def test_invalid_feature_input(self):
        # Case 1: Wrong number of features (e.g. 500 instead of 506)
        bad_df1 = pd.DataFrame(np.zeros((1, 500)))
        with self.assertRaises(ValueError):
            self.explainer.explain_transaction(bad_df1)

        # Case 2: Wrong column names
        bad_df2 = pd.DataFrame(np.zeros((1, 506)), columns=[f"f_{i}" for i in range(506)])
        with self.assertRaises(ValueError):
            self.explainer.explain_transaction(bad_df2)

        # Case 3: Invalid type (list instead of DataFrame)
        with self.assertRaises(TypeError):
            self.explainer.explain_transaction([0] * 506)

        print("\n[PASS] Test Invalid Feature Inputs Rejected")

if __name__ == "__main__":
    unittest.main()
