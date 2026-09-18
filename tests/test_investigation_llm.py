import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evidence_engine import build_investigation_evidence
from investigation_llm import (
    LLMSummarizer,
    generate_investigator_summary,
    validate_llm_summary,
    FIXED_SYSTEM_INSTRUCTION
)

class TestInvestigationLLM(unittest.TestCase):

    def setUp(self):
        # Sample minimal valid evidence for fast unit testing
        self.sample_evidence = {
            "transaction_id": 2987004,
            "model_prediction": {
                "model_name": "LightGBM Tuned",
                "checkpoint_path": "models/tuned/lightgbm_tuned.pkl",
                "predicted_class": 0,
                "fraud_probability": 0.008169,
                "threshold": None,
                "threshold_note": "Decision threshold is not defined in project artifacts. Value reported as null."
            },
            "model_explanation": {
                "explainer": "SHAP TreeExplainer",
                "output_space": "raw margin (log-odds)",
                "base_value": -4.904269,
                "top_features": [
                    {
                        "feature": "V165",
                        "feature_value": 5155.0,
                        "shap_value": -0.982136,
                        "absolute_shap_value": 0.982136,
                        "direction": "decreases_model_output"
                    }
                ]
            },
            "transaction_evidence": {
                "transaction_id": 2987004,
                "transaction_dt": 86506,
                "transaction_amount": 50.0,
                "product_cd": "W",
                "features": {
                    "card1_mean": 96.97,
                    "tx_count_24h": 0,
                    "DeviceType": "mobile"
                }
            },
            "graph_evidence": {
                "entities": {
                    "card": {"card1": 4497},
                    "device": {"DeviceType": "mobile"},
                    "address": {"addr1": 420, "addr2": 87},
                    "p_email": "gmail.com",
                    "r_email": None
                },
                "entity_counts": {
                    "card_transaction_count": 18,
                    "device_transaction_count": 9,
                    "address_transaction_count": 3581,
                    "p_email_transaction_count": 228355,
                    "r_email_transaction_count": 0
                },
                "temporal_status": {
                    "previous": 1,
                    "same": 0,
                    "future": 0
                }
            },
            "reason_codes": [
                {
                    "code": "R01",
                    "description": "Transaction amount is 50.0 and historical card amount mean is 96.97.",
                    "evidence": {"transaction_amount": 50.0, "card_historical_mean_amt": 96.97}
                }
            ],
            "supporting_transactions": [],
            "evidence_scope": {
                "source": ["LightGBM model output", "SHAP TreeExplainer", "TigerGraph FraudGraph"],
                "llm_allowed_facts_only": True
            }
        }

        # Valid mock LLM response JSON matching required schema
        self.sample_valid_llm_response = {
            "transaction_id": 2987004,
            "case_summary": "Transaction 2987004 evaluated by LightGBM model with low fraud probability.",
            "model_prediction": {
                "model_name": "LightGBM Tuned",
                "predicted_class": 0,
                "fraud_probability": 0.008169,
                "threshold": None
            },
            "model_explanation": {
                "summary": "V165 decreased model output in log-odds space.",
                "top_features": [
                    {
                        "feature": "V165",
                        "feature_value": 5155.0,
                        "shap_value": -0.982136,
                        "direction": "decreases_model_output"
                    }
                ]
            },
            "graph_evidence_summary": {
                "observations": ["Card connected to 18 transactions."]
            },
            "supporting_transactions_summary": [],
            "limitations": ["No decision threshold documented in artifacts."]
        }

    # 1. Mock provider
    def test_mock_provider(self):
        summarizer = LLMSummarizer(mock_mode=True)
        summary = summarizer.summarize(self.sample_evidence)
        self.assertEqual(summary["transaction_id"], 2987004)
        self.assertTrue(validate_llm_summary(summary))

    # 2. OpenRouter configuration validation
    def test_openrouter_configuration_validation(self):
        summarizer = LLMSummarizer(mock_mode=False, provider="unknown_provider")
        with self.assertRaises(ValueError) as ctx:
            summarizer.summarize(self.sample_evidence)
        self.assertIn("Unsupported LLM provider", str(ctx.exception))

    # 3. Missing API key
    def test_missing_api_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "LLM_MODEL": "openai/gpt-4o-mini"}, clear=False):
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o-mini", api_key=None)
            with self.assertRaises(ValueError) as ctx:
                summarizer.summarize(self.sample_evidence)
            self.assertIn("OpenRouter API key is missing", str(ctx.exception))

    # 4. Missing model
    def test_missing_model(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False):
            if "LLM_MODEL" in os.environ:
                del os.environ["LLM_MODEL"]
            if "LLM_MODELS" in os.environ:
                del os.environ["LLM_MODELS"]
            summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model=None, api_key="sk-or-test-key")
            with self.assertRaises(ValueError) as ctx:
                summarizer.summarize(self.sample_evidence)
            self.assertIn("OpenRouter model slug is missing", str(ctx.exception))

    # 5. Evidence validation before API call
    def test_evidence_validation_before_api_call(self):
        bad_evidence = json.loads(json.dumps(self.sample_evidence))
        bad_evidence["reason_codes"].append({
            "code": "R99",
            "description": "This pattern proves fraud.", # forbidden phrase
            "evidence": {}
        })
        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        with self.assertRaises(ValueError) as ctx:
            summarizer.summarize(bad_evidence)
        self.assertIn("Evidence validation failed", str(ctx.exception))

    # 6. No API call if evidence is invalid
    @patch("investigation_llm.OpenAI")
    def test_no_api_call_if_evidence_invalid(self, mock_openai_cls):
        bad_evidence = {"invalid": "not_real_evidence"}
        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        with self.assertRaises(ValueError):
            summarizer.summarize(bad_evidence)
        mock_openai_cls.assert_not_called()

    # 7. Dynamic Evidence JSON transmission
    @patch("investigation_llm.OpenAI")
    def test_dynamic_evidence_json_transmission(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(self.sample_valid_llm_response)))]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        summarizer.summarize(self.sample_evidence)

        # Inspect messages passed to chat.completions.create
        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue(messages[0]["content"].startswith(FIXED_SYSTEM_INSTRUCTION))
        self.assertEqual(messages[1]["role"], "user")
        
        # Verify user message is dynamically serialized valid Evidence JSON matching transaction
        sent_data = json.loads(messages[1]["content"])
        self.assertEqual(sent_data["transaction_id"], 2987004)
        self.assertIn("model_prediction", sent_data)

    # 8. Structured JSON parsing
    @patch("investigation_llm.OpenAI")
    def test_structured_json_parsing(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # Test markdown code fences stripping
        fenced_json = f"```json\n{json.dumps(self.sample_valid_llm_response)}\n```"
        mock_response.choices = [MagicMock(message=MagicMock(content=fenced_json))]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        summary = summarizer.summarize(self.sample_evidence)
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["transaction_id"], 2987004)

    # 9. Malformed response rejection
    @patch("investigation_llm.OpenAI")
    def test_malformed_response_rejection(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="This is not JSON"))]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        with self.assertRaises(ValueError) as ctx:
            summarizer.summarize(self.sample_evidence)
        self.assertIn("could not be parsed as valid JSON", str(ctx.exception))

    # 10. Transaction ID preservation
    @patch("investigation_llm.OpenAI")
    def test_transaction_id_preservation(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mismatched_response = json.loads(json.dumps(self.sample_valid_llm_response))
        mismatched_response["transaction_id"] = 9999999 # Wrong ID
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(mismatched_response)))]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        with self.assertRaises(ValueError) as ctx:
            summarizer.summarize(self.sample_evidence)
        self.assertIn("Transaction ID mismatch", str(ctx.exception))

    # 11. API failure handling
    @patch("investigation_llm.OpenAI")
    def test_api_failure_handling(self, mock_openai_cls):
        from openai import RateLimitError
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded", response=MagicMock(status_code=429), body=None
        )

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        with self.assertRaises(ValueError) as ctx:
            summarizer.summarize(self.sample_evidence)
        self.assertIn("Rate Limit", str(ctx.exception))

    # 12. API key not exposed in logs or errors
    @patch("investigation_llm.OpenAI")
    def test_api_key_not_exposed(self, mock_openai_cls):
        from openai import AuthenticationError
        secret_key = "sk-or-v1-my-secret-top-confidential-token-12345"
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid auth token provided", response=MagicMock(status_code=401), body=None
        )

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key=secret_key)
        try:
            summarizer.summarize(self.sample_evidence)
            self.fail("Expected ValueError")
        except ValueError as e:
            err_msg = str(e)
            self.assertNotIn(secret_key, err_msg)
            self.assertIn("Authentication Failure", err_msg)

    # 13. Security Test (Section 17)
    @patch("investigation_llm.OpenAI")
    def test_security_payload_boundary(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(self.sample_valid_llm_response)))]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = LLMSummarizer(mock_mode=False, provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        summarizer.summarize(self.sample_evidence)

        _, kwargs = mock_client.chat.completions.create.call_args
        full_call_str = json.dumps(kwargs)
        
        # Invariant checks: MUST NOT contain internal components or secrets
        self.assertNotIn("sk-test", full_call_str)
        self.assertNotIn("TigerGraphClient", full_call_str)
        self.assertNotIn("ModelRunner", full_call_str)
        self.assertNotIn("SHAPExplainer", full_call_str)
        self.assertNotIn(".parquet", full_call_str)
        self.assertNotIn("rgv2796262i110272u8h35ut11nasaup", full_call_str) # tigergraph secret

if __name__ == "__main__":
    unittest.main()

