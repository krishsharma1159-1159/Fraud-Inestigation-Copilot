import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure root and backend directories are in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
SRC_DIR = os.path.join(ROOT_DIR, "src")

for p in [ROOT_DIR, BACKEND_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.main import app

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("components", data)
        self.assertIn("model", data["components"])
        self.assertIn("tigergraph", data["components"])
        self.assertIn("llm", data["components"])
        self.assertIn("dataset", data["components"])

    def test_02_investigate_valid_transaction(self):
        payload = {
            "transaction_id": 2987004,
            "mock_llm": True
        }
        response = self.client.post("/api/investigate", json=payload)
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        data = response.json()
        
        # Verify UI contract schema
        self.assertEqual(data["targetTxnId"], "2987004")
        self.assertIn("ml", data)
        self.assertIn(data["ml"]["prediction"], ["SUSPICIOUS", "NORMAL"])
        self.assertIsInstance(data["ml"]["riskScore"], (int, float))
        self.assertIn("LightGBM", data["ml"]["modelName"])
        
        # Section 3: Calculated Evidence
        self.assertIsInstance(data["calculatedEvidence"], list)
        self.assertGreater(len(data["calculatedEvidence"]), 0)
        first_ev = data["calculatedEvidence"][0]
        self.assertIn("code", first_ev)
        self.assertIn("title", first_ev)
        self.assertIn("metrics", first_ev)

        # Section 4: Graph
        self.assertIn("graph", data)
        self.assertIsInstance(data["graph"]["nodes"], list)
        self.assertIsInstance(data["graph"]["edges"], list)
        self.assertGreater(len(data["graph"]["nodes"]), 0)

        # Section 5: Supporting Transactions
        self.assertIn("supportingTxns", data)
        self.assertIsInstance(data["supportingTxns"], list)
        self.assertGreater(len(data["supportingTxns"]), 0)

        # Section 6: Timeline
        self.assertIn("timeline", data)
        self.assertIsInstance(data["timeline"], list)
        self.assertGreater(len(data["timeline"]), 0)

        # Section 7: AI Brief & Template Brief
        self.assertIn("aiBrief", data)
        self.assertIsInstance(data["aiBrief"], str)
        self.assertGreater(len(data["aiBrief"]), 20)
        self.assertIn("templateBrief", data)

        # Facts Anchors
        self.assertIn("facts", data)
        self.assertIn("txn", data["facts"])
        self.assertIn("risk", data["facts"])

    def test_03_investigate_invalid_transaction(self):
        payload = {
            "transaction_id": -999,
            "mock_llm": True
        }
        response = self.client.post("/api/investigate", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_04_analyst_feedback_valid(self):
        payload = {
            "transaction_id": "2987004",
            "disposition": "CONFIRMED_SUSPICIOUS",
            "analyst_id": "Test Investigator",
            "notes": "Regression test feedback entry"
        }
        response = self.client.post("/api/actions/feedback", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(data["entry_id"].startswith("DISP-"))

    def test_05_analyst_feedback_invalid_disposition(self):
        payload = {
            "transaction_id": "2987004",
            "disposition": "INVALID_DISPOSITION"
        }
        response = self.client.post("/api/actions/feedback", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_06_case_escalation_review(self):
        payload = {
            "transaction_id": "2987004",
            "reason": "Test escalation for AML review",
            "priority": "HIGH"
        }
        response = self.client.post("/api/actions/review", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(data["entry_id"].startswith("ESC-"))

    def test_07_audit_trail_lookup(self):
        response = self.client.get("/api/actions/audit-trail?transaction_id=2987004")
        self.assertEqual(response.status_code, 200)
        entries = response.json()
        self.assertIsInstance(entries, list)
        self.assertGreater(len(entries), 0)

    def test_08_export_dossier(self):
        response = self.client.get("/api/investigate/2987004/export")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["dossier_id"], "ARTHA-DOSSIER-2987004")
        self.assertIn("investigation_payload", data)

    def test_09_similar_cases(self):
        response = self.client.get("/api/similar-cases?transaction_id=2987004&limit=3")
        self.assertEqual(response.status_code, 200)
        cases = response.json()
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)
        self.assertIn("similarity_score", cases[0])

    def test_10_root_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("ARTHADRISHTI", response.text)

if __name__ == "__main__":
    unittest.main()
