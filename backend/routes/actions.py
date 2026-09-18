from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from schemas import FeedbackRequest, ReviewRequest, ActionResponse
from services.audit_service import record_disposition, record_escalation, get_audit_trail

router = APIRouter(prefix="/api/actions", tags=["Actions"])

@router.post("/feedback", response_model=ActionResponse)
def submit_feedback(req: FeedbackRequest):
    """
    Records human investigator disposition sign-off into the audit log.
    """
    valid_dispositions = ["CONFIRMED_SUSPICIOUS", "NOT_SUSPICIOUS", "NEEDS_REVIEW"]
    if req.disposition not in valid_dispositions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid disposition '{req.disposition}'. Must be one of {valid_dispositions}."
        )

    entry = record_disposition(
        transaction_id=str(req.transaction_id),
        disposition=req.disposition,
        analyst_id=req.analyst_id or "Sr. Investigator (Tier-3)",
        notes=req.notes
    )

    return ActionResponse(
        status="SUCCESS",
        message=f"Disposition '{req.disposition}' recorded into audit stream.",
        entry_id=entry["entry_id"],
        timestamp=entry["timestamp"]
    )

@router.post("/review", response_model=ActionResponse)
def mark_for_review(req: ReviewRequest):
    """
    Escalates transaction to the Senior Compliance Review queue.
    """
    entry = record_escalation(
        transaction_id=str(req.transaction_id),
        analyst_id=req.analyst_id or "Sr. Investigator (Tier-3)",
        reason=req.reason or "Escalated for Senior Compliance Review",
        priority=req.priority or "HIGH"
    )

    return ActionResponse(
        status="SUCCESS",
        message=f"Case {req.transaction_id} escalated to Senior Compliance Review.",
        entry_id=entry["entry_id"],
        timestamp=entry["timestamp"]
    )

@router.get("/audit-trail")
def list_audit_trail(transaction_id: Optional[str] = Query(None)):
    """
    Retrieves compliance audit trail records.
    """
    return get_audit_trail(transaction_id)
