'''Purpose: For each question:
    domain classify(hybrid_classify) + risk hints    
    retrieve top-k historical matches    
    compute match strength + decide route
    output state.enriched: List[EnrichedQuestion] +  Error Handling'''

from typing import List

from app.models.state import TenderState, EnrichedQuestion, RetrievedMatch
from app.pipeline.classify import hybrid_classify
from app.rag.retriever import retrieve_matches_from_vs
from app.pipeline.routing import match_strength, route_from_strength
from app.core.llm import get_embeddings
from app.rag.chroma_store import init_stores


def classify_retrieve_route_node(state: TenderState) -> TenderState:
    out: List[EnrichedQuestion] = []

    emb = get_embeddings()
    hist_vs, _gen_vs = init_stores(emb)

    for item in state.questions:
        try:
            # 1) classify
            dec = hybrid_classify(item.question)

            # 2) retrieve
            matches = retrieve_matches_from_vs(hist_vs, item.question, state.retrieval_top_k)

            top_sim = matches[0].similarity if matches else 0.0
            strength = match_strength(
                top_sim,
                strong=state.strong_threshold,
                weak=state.weak_threshold,
            )

            route = route_from_strength(strength)

            out.append(
                EnrichedQuestion(
                    question_id=item.question_id,
                    question=item.question,
                    domain=dec.domain,
                    domain_confidence=dec.confidence,
                    domain_used=dec.used,
                    domain_rationale=dec.rationale,
                    risk_hints=dec.risk_hints,
                    matches=matches,
                    top_similarity=top_sim,
                    match_strength=strength,
                    route=route,
                )
            )

        except Exception as e:
            out.append(
                EnrichedQuestion(
                    question_id=item.question_id,
                    question=item.question,
                    domain="General",
                    domain_confidence=0.0,
                    domain_used="rules",
                    domain_rationale="classification/retrieval failed",
                    risk_hints=[],
                    matches=[],
                    top_similarity=0.0,
                    match_strength="NONE",
                    route="TEMPLATE_SAFE",
                    error=str(e),
                )
            )

    state.enriched = out
    return state