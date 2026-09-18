import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger("AuditService")

AUDIT_LOG_FILE = os.environ.get("AUDIT_LOG_PATH", r"e:\Skillcred\data\audit_log.json")

def _load_log() -> List[Dict[str, Any]]:
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        logger.warning(f"Could not read audit log file: {e}")
        return []

def _save_log(entries: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    temp_file = AUDIT_LOG_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, AUDIT_LOG_FILE)

def record_disposition(
    transaction_id: str,
    disposition: str,
    analyst_id: str = "Sr. Investigator (Tier-3)",
    notes: Optional[str] = None
) -> Dict[str, Any]:
    entry_id = f"DISP-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "entry_id": entry_id,
        "action_type": "ANALYST_DISPOSITION",
        "transaction_id": str(transaction_id),
        "disposition": disposition,
        "analyst_id": analyst_id,
        "notes": notes or f"Investigator marked disposition as {disposition}",
        "timestamp": timestamp
    }
    entries = _load_log()
    entries.append(entry)
    _save_log(entries)
    logger.info(f"Recorded analyst disposition {disposition} for {transaction_id} [ID: {entry_id}]")
    return entry

def record_escalation(
    transaction_id: str,
    analyst_id: str = "Sr. Investigator (Tier-3)",
    reason: str = "Escalated for Senior Compliance Review",
    priority: str = "HIGH"
) -> Dict[str, Any]:
    entry_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "entry_id": entry_id,
        "action_type": "CASE_ESCALATION",
        "transaction_id": str(transaction_id),
        "priority": priority,
        "analyst_id": analyst_id,
        "reason": reason,
        "status": "QUEUED_FOR_REVIEW",
        "timestamp": timestamp
    }
    entries = _load_log()
    entries.append(entry)
    _save_log(entries)
    logger.info(f"Recorded case escalation for {transaction_id} [ID: {entry_id}]")
    return entry

def get_audit_trail(transaction_id: Optional[str] = None) -> List[Dict[str, Any]]:
    entries = _load_log()
    if transaction_id:
        return [e for e in entries if e.get("transaction_id") == str(transaction_id)]
    return entries
