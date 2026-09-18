import os
import sys
from datetime import datetime, timezone
from fastapi import APIRouter

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schemas import HealthResponse
from evidence_engine import get_model_runner, get_tg_client
from investigation_llm import get_configured_models

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_system_health():
    """
    Returns full health status across LightGBM, TigerGraph, OpenRouter, and Dataset.
    """
    components = {}

    # 1. Model Runner Health
    try:
        runner = get_model_runner()
        model_status = "OK" if runner.model is not None else "DEGRADED"
        components["model"] = {
            "status": model_status,
            "model_name": "LightGBM Tuned",
            "features_count": len(runner.expected_features) if runner.expected_features else 506
        }
    except Exception as e:
        components["model"] = {"status": "DOWN", "error": str(e)}

    # 2. TigerGraph Health
    try:
        tg_client = get_tg_client()
        conn = tg_client._get_connection()
        components["tigergraph"] = {
            "status": "OK" if conn is not None else "DEGRADED",
            "host": tg_client.host,
            "graph": tg_client.graph
        }
    except Exception as e:
        components["tigergraph"] = {"status": "DOWN", "error": str(e)}

    # 3. LLM / OpenRouter Health
    llm_provider = os.environ.get("LLM_PROVIDER", "mock")
    models = get_configured_models()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    components["llm"] = {
        "status": "OK" if (has_key or llm_provider == "mock") else "DEGRADED",
        "provider": llm_provider,
        "configured_models": models,
        "api_key_configured": has_key
    }

    # 4. Behavioral Dataset Health
    beh_path = os.environ.get("BEH_PATH", r"e:\Skillcred\data skillcred\train_behavioral.parquet")
    components["dataset"] = {
        "status": "OK" if os.path.exists(beh_path) else "DEGRADED",
        "path": beh_path
    }

    overall_status = "HEALTHY" if all(c.get("status") in ["OK", "HEALTHY"] for c in components.values()) else "DEGRADED"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components
    )
