'''API request/response models for Swagger and validation.'''

from pydantic import BaseModel
from typing import List, Optional
from app.models.state import QuestionItem, TenderState, RetrievedMatch

class AskRequest(BaseModel):
    request_id: Optional[str] = None
    questions: List[QuestionItem]

    retrieval_top_k: Optional[int] = None
    strong_threshold: Optional[float] = None
    weak_threshold: Optional[float] = None

class AskResponse(BaseModel):
    state: TenderState

class RetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = None

class RetrieveResponse(BaseModel):
    query: str
    matches: List[RetrievedMatch]