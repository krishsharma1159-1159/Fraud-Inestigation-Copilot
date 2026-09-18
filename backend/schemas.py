from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

# ==========================================
# REQUEST SCHEMAS
# ==========================================

class InvestigateRequest(BaseModel):
    transaction_id: Union[int, str] = Field(..., description="Target transaction ID from IEEE-CIS dataset or scenario")
    mock_llm: Optional[bool] = Field(default=None, description="Force mock LLM or use live OpenRouter configured model")
    model_name: Optional[str] = Field(default=None, description="Optional specific model slug")

class MultiModelInvestigateRequest(BaseModel):
    transaction_id: Union[int, str] = Field(..., description="Target transaction ID")
    models: Optional[List[str]] = Field(default=None, description="List of model slugs to evaluate")
    mock_llm: Optional[bool] = Field(default=None, description="Force mock mode if true")

class FeedbackRequest(BaseModel):
    transaction_id: Union[int, str]
    disposition: str = Field(..., description="CONFIRMED_SUSPICIOUS, NOT_SUSPICIOUS, or NEEDS_REVIEW")
    analyst_id: Optional[str] = Field(default="Sr. Investigator (Tier-3)")
    notes: Optional[str] = Field(default=None)

class ReviewRequest(BaseModel):
    transaction_id: Union[int, str]
    analyst_id: Optional[str] = Field(default="Sr. Investigator (Tier-3)")
    reason: Optional[str] = Field(default="Escalated for Senior Compliance Review")
    priority: Optional[str] = Field(default="HIGH")

# ==========================================
# RESPONSE SCHEMAS (Matches code.html Contract)
# ==========================================

class MLModelPrediction(BaseModel):
    prediction: str = Field(..., description="SUSPICIOUS or NORMAL")
    riskScore: float = Field(..., description="Risk score percentage 0.0 - 100.0")
    modelName: str = Field(..., description="E.g. LightGBM Tuned (506 features)")
    confidence: str = Field(..., description="Confidence label, e.g. 98.1% (High)")
    inferenceLatencyMs: Optional[float] = Field(default=14.8, description="Model latency in ms")

class EvidenceMetric(BaseModel):
    label: str
    value: str

class CalculatedEvidenceItem(BaseModel):
    code: str
    title: str
    desc: str
    metrics: List[EvidenceMetric]
    severity: str = Field(default="warning", description="critical or warning")

class GraphNode(BaseModel):
    id: str
    label: str
    sublabel: Optional[str] = None
    type: str = Field(..., description="ACCOUNT | FLAGGED_TXN | SUPPORTING_TXN | RECIPIENT")
    x: float
    y: float
    totalTxn: Optional[Union[int, str]] = None
    connected: Optional[Union[int, str]] = None
    flagged: Optional[Union[int, str]] = None
    avg: Optional[str] = None
    amount: Optional[str] = None
    status: Optional[str] = None
    note: Optional[str] = None

class GraphEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: str = Field(..., alias="from")
    to: str
    label: str
    flagged: bool = False
    supporting: bool = False

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class SupportingTxnRow(BaseModel):
    id: str
    time: str
    amount: str
    type: str
    recipient: str
    status: str
    isFlagged: bool = False
    isSupporting: bool = False
    isNormal: bool = False

class TimelineEvent(BaseModel):
    time: str
    amount: str
    state: str = Field(..., description="Flagged, Suspicious, Unusual, or Normal")
    desc: str

class InvestigationUIResponse(BaseModel):
    targetTxnId: str
    ml: MLModelPrediction
    calculatedEvidence: List[CalculatedEvidenceItem]
    graph: GraphData
    supportingTxns: List[SupportingTxnRow]
    timeline: List[TimelineEvent]
    aiBrief: str
    templateBrief: str
    facts: Dict[str, Any]
    rawEvidence: Optional[Dict[str, Any]] = None
    pipelineStatus: str = "SUCCESS"

class SimilarCase(BaseModel):
    transaction_id: str
    similarity_score: float
    risk_score: float
    amount: str
    shared_attributes: List[str]
    reason: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]

class ActionResponse(BaseModel):
    status: str = "SUCCESS"
    message: str
    entry_id: Optional[str] = None
    timestamp: str
