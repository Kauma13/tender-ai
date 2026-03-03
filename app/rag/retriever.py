'''Purpose: Centralize mapping from vector results to RetrievedMatch.'''

from typing import List
from app.core.llm import get_embeddings
from app.rag.chroma_store import init_stores, retrieve_top_k, hits_to_matches
from app.models.state import RetrievedMatch
from app.core.config import settings

def retrieve_matches(query: str, k: int | None = None) -> List[RetrievedMatch]:
    emb = get_embeddings()
    hist_vs, _gen_vs = init_stores(emb)

    hits = retrieve_top_k(hist_vs, query=query, k=(k or settings.retrieval_top_k))
    return hits_to_matches(hits)

def retrieve_matches_from_vs(hist_vs, query: str, k: int) -> List[RetrievedMatch]:
    hits = retrieve_top_k(hist_vs, query=query, k=(k or settings.retrieval_top_k))
    return hits_to_matches(hits)