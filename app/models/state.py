''' Defines shared Pydantic models used in both ingestion and runtime. '''

# app/models/state.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class QuestionItem(BaseModel):
    question_id: str
    question: str
    suggested_domain: Optional[str] = None

class RetrievedMatch(BaseModel):
    source_id: str
    domain: str
    historical_question: str
    historical_answer: str
    similarity: float

class EnrichedQuestion(BaseModel):
    question_id: str
    question: str

    # classification
    domain: str
    domain_confidence: float
    domain_used: Literal["rules", "llm"]
    domain_rationale: str
    risk_hints: List[str] = Field(default_factory=list)

    # retrieval
    matches: List[RetrievedMatch] = Field(default_factory=list)
    top_similarity: float = 0.0
    match_strength: Literal["STRONG", "WEAK", "NONE"] = "NONE"
    route: Literal["HISTORICAL_GUIDED", "TEMPLATE_SAFE"] = "TEMPLATE_SAFE"

    # generation outputs
    answer: str = ""
    confidence: float = 0.0
    alignment: Literal["HIGH", "MED", "LOW"] = "LOW"
    flags: List[str] = Field(default_factory=list)

    error: Optional[str] = None

class RunSummary(BaseModel):
    total: int = 0
    errors: int = 0
    routed_historical: int = 0
    routed_template: int = 0
    flagged_risk: int = 0
    status: str = "PENDING"

    requires_sme_review: int = 0
    unsupported: int = 0
    inconsistent: int = 0
    overclaim: int = 0
    ready_to_submit: int = 0
    needs_review: int = 0

    confidence_min: float = 0.0
    confidence_avg: float = 0.0
    confidence_max: float = 0.0

class TenderState(BaseModel):
    request_id: str
    questions: List[QuestionItem] = Field(default_factory=list)
    enriched: List[EnrichedQuestion] = Field(default_factory=list)
    summary: RunSummary = Field(default_factory=RunSummary)

    retrieval_top_k: int = 3
    strong_threshold: float = 0.70
    weak_threshold: float = 0.60
