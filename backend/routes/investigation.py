import os
import sys
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schemas import (
    InvestigateRequest,
    InvestigationUIResponse,
    MultiModelInvestigateRequest,
    SimilarCase
)
from services.investigation_service import (
    perform_investigation,
    run_multi_model,
    find_similar_cases
)

router = APIRouter(prefix="/api", tags=["Investigation"])

@router.post("/investigate", response_model=InvestigationUIResponse)
def investigate_transaction(req: InvestigateRequest):
    """
    Executes the End-to-End Investigation Pipeline for a single transaction.
    Returns complete UI payload matching code.html requirements.
    """
    try:
        response_data = perform_investigation(
            transaction_id=req.transaction_id,
            mock_llm=req.mock_llm,
            model_name=req.model_name
        )
        return response_data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")

@router.post("/investigate/multi-model")
def investigate_multi_model(req: MultiModelInvestigateRequest):
    """
    Dispatches validated Evidence JSON to all configured OpenRouter models.
    """
    try:
        results = run_multi_model(
            transaction_id=req.transaction_id,
            models=req.models,
            mock_llm=req.mock_llm
        )
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-model investigation failed: {str(e)}")

@router.get("/investigate/{transaction_id}/export")
def export_investigation_dossier(transaction_id: str, mock_llm: bool = True):
    """
    Returns full forensic case dossier in downloadable JSON format.
    """
    try:
        response_data = perform_investigation(transaction_id=transaction_id, mock_llm=mock_llm)
        dossier = {
            "dossier_id": f"ARTHA-DOSSIER-{transaction_id}",
            "transaction_id": transaction_id,
            "export_metadata": {
                "generated_by": "Arthadrishti Fraud Investigation Workspace",
                "system_version": "v1.12-rt",
                "audited_by": "Sr. Investigator (Tier-3)"
            },
            "investigation_payload": response_data
        }
        return JSONResponse(
            content=dossier,
            headers={
                "Content-Disposition": f'attachment; filename="Arthadrishti_{transaction_id}_Dossier.json"'
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/similar-cases", response_model=List[SimilarCase])
def get_similar_cases(
    transaction_id: str = Query(..., description="Active Transaction ID to match"),
    limit: int = Query(4, ge=1, le=10)
):
    """
    Returns historically correlated similar cases from dataset.
    """
    try:
        return find_similar_cases(transaction_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity lookup failed: {str(e)}")
