'''Defines endpoints (/ask) and calls the graph. Initial point for runtime path'''

from fastapi import APIRouter
import uuid

from app.api.schemas import AskRequest, AskResponse, RetrieveRequest, RetrieveResponse
from app.models.state import TenderState
from app.core.config import settings

from app.rag.retriever import retrieve_matches
from app.rag.chroma_store import init_stores, collection_count

from app.graph.tender_graph import build_graph

router = APIRouter()
tender_app = build_graph()

@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/stores/info")
def stores_info():
    emb = get_embeddings()
    hist_vs, gen_vs = init_stores(emb)
    return {
        "persist_dir": settings.persist_dir,
        "hist_collection": settings.hist_collection,
        "hist_count": collection_count(hist_vs),
        "gen_collection": settings.gen_collection,
        "gen_count": collection_count(gen_vs),
    }

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    request_id = req.request_id or str(uuid.uuid4())

    state = TenderState(
        request_id=request_id,
        questions=req.questions,
        retrieval_top_k=req.retrieval_top_k,
        strong_threshold=req.strong_threshold,
        weak_threshold=req.weak_threshold,
    )

    # Run graph
    final_state = tender_app.invoke(state)
    return AskResponse(state=final_state)

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    matches = retrieve_matches(req.query, req.k)
    return RetrieveResponse(query=req.query, matches=matches)

