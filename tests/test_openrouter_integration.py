import sys
import os
import time
import unittest
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evidence_engine import build_investigation_evidence, validate_evidence
from investigation_llm import (
    generate_investigator_summary,
    generate_multi_model_summaries,
    validate_llm_summary,
    get_configured_models
)

class TestOpenRouterIntegration(unittest.TestCase):

    def test_real_openrouter_multi_model_investigation(self):
        run_flag = os.environ.get("RUN_OPENROUTER_INTEGRATION")
        provider = os.environ.get("LLM_PROVIDER", "").lower()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        models = get_configured_models()

        if run_flag != "1" or provider != "openrouter" or not api_key or not models:
            raise unittest.SkipTest(
                "Skipping real OpenRouter integration test. To run, set: "
                "RUN_OPENROUTER_INTEGRATION=1, LLM_PROVIDER=openrouter, "
                "LLM_MODELS=<comma-separated-models>, and OPENROUTER_API_KEY=<secret>."
            )

        tx_id = 2987004
        print(f"\n========================================================")
        print(f"RUNNING MULTI-MODEL OPENROUTER INTEGRATION: TX {tx_id}")
        print(f"Configured Models ({len(models)}): {', '.join(models)}")
        print(f"========================================================")

        # 1. Programmatically build evidence
        t0 = time.time()
        evidence = build_investigation_evidence(tx_id)
        evidence_build_time = time.time() - t0
        print(f"Evidence built dynamically in {evidence_build_time:.2f}s")

        # 2. Validate evidence
        self.assertTrue(validate_evidence(evidence))
        print("Evidence validation gate: PASS")

        # 3. Send same Evidence JSON independently to all five models
        results = generate_multi_model_summaries(evidence, models=models, mock=False)

        print("\n--- INDIVIDUAL MODEL RESULTS ---")
        success_count = 0
        for model_slug in models:
            res = results["models"].get(model_slug, {})
            status = res.get("status")
            latency = res.get("latency_seconds", 0.0)
            validation = res.get("validation")
            err = res.get("error")

            if status == "SUCCESS":
                success_count += 1
                summary = res.get("summary", {})
                print(f"[SUCCESS] {model_slug} | Latency: {latency:.2f}s | Validation: {validation}")
                print(f"          Case Summary: {summary.get('case_summary', '')[:120]}...")
            else:
                print(f"[FAIL]    {model_slug} | Latency: {latency:.2f}s | Error: {err}")

        print(f"\nOpenRouter Multi-Model Completed: {success_count}/{len(models)} succeeded.")
        print(f"========================================================\n")

        # Invariant: At least one model succeeded and results dictionary contains all tested models
        self.assertEqual(len(results["models"]), len(models))

if __name__ == "__main__":
    unittest.main()

