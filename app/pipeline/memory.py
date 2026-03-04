'''Purpose: Two persistence operations:
    always write audit JSON file (write_run_audit)
    optionally ingest generated answers into generated vector store (ingest_generated_answers) using gating policy (should_store_generated)'''

import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.models.state import TenderState, EnrichedQuestion
from app.core.config import settings
from app.core.llm import get_embeddings
from app.rag.chroma_store import init_stores
from app.pipeline.prompts import build_evidence_pack
from app.pipeline.utils import stable_id

# ---------------------------
# 1) Audit logging
# ---------------------------
def write_run_audit(state: TenderState) -> str:
    """
    Writes a run audit JSON to settings.runs_dir.
    Returns absolute path of the written file.

    Keep it robust: never crash the pipeline if audit fails.
    """
    os.makedirs(settings.runs_dir, exist_ok=True)

    payload = {
        "request_id": state.request_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "summary": state.summary.model_dump() if hasattr(state.summary, "model_dump") else state.summary.dict(),
        "questions": [q.model_dump() for q in state.questions] if hasattr(state.questions[0], "model_dump") else [q.dict() for q in state.questions],
        "enriched": [],
    }

    for q in state.enriched:
        evidence = build_evidence_pack(q, max_items=3)
        payload["enriched"].append({
            "question_id": q.question_id,
            "question": q.question,
            "domain": q.domain,
            "domain_confidence": q.domain_confidence,
            "domain_used": q.domain_used,
            "risk_hints": q.risk_hints,
            "route": q.route,
            "match_strength": q.match_strength,
            "top_similarity": q.top_similarity,
            "flags": q.flags,
            "alignment": q.alignment,
            "confidence": q.confidence,
            "answer": q.answer,
            "error": q.error,
            "evidence_pack": evidence,
        })

    out_path = os.path.join(settings.runs_dir, f"{state.request_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return os.path.abspath(out_path)

# ---------------------------
# 2) Store gating policy
# ---------------------------
def should_store_generated(q: EnrichedQuestion) -> Tuple[bool, str]:
    """
    Decide if we should store the generated answer into the 'generated' long-term memory store.

    flags = set(q.flags or [])

    if not (q.answer or "").strip():
        return False, "empty_answer"

    # Hard blockers
    hard_block = {
        "GENERATION_ERROR",
        "VERIFICATION_ERROR",
        "INCONSISTENT_WITH_EVIDENCE",
        "UNSUPPORTED_BY_EVIDENCE",
        "REQUIRES_SME_REVIEW",
    }
    if flags.intersection(hard_block):
        return False, f"blocked_by_flags:{sorted(flags.intersection(hard_block))}"

    # If verifier says partial support, still stores, but require higher confidence
    partially = "PARTIALLY_SUPPORTED_BY_EVIDENCE" in flags

    # Confidence gates
    if partially:
        if float(q.confidence or 0.0) < 0.75:
            return False, "partial_support_low_confidence"
    else:
        if float(q.confidence or 0.0) < 0.65:
            return False, "low_confidence"

    return True, "ok"

# ---------------------------
# 3) Ingest generated answers
# ---------------------------
def ingest_generated_answers(state: TenderState) -> int:
    """
    Writes selected generated answers into the generated Chroma collection.
    Returns number of stored docs.
    """
    embeddings = get_embeddings()
    _hist_vs, gen_vs = init_stores(embeddings)

    docs: List[Document] = []
    ids: List[str] = []

    for q in state.enriched:
        ok, reason = should_store_generated(q)
        if not ok:
            continue

        evidence = build_evidence_pack(q, max_items=3)

        content = (
            f"QUESTION:\n{q.question}\n\n"
            f"ANSWER:\n{q.answer}\n\n"
            f"EVIDENCE:\n{evidence}\n"
        )

        # doc id prevents duplicates (stable per question+answer)
        doc_id = stable_id("generated", q.domain, q.question, q.answer)

        metadata={
                    "kind": "generated",
                    "request_id": state.request_id,
                    "question_id": q.question_id,
                    "domain": q.domain,
                    "route": q.route,
                    "match_strength": q.match_strength,
                    "top_similarity": float(q.top_similarity or 0.0),
                    "confidence": float(q.confidence or 0.0),
                    "alignment": q.alignment,
                    # "flags": list(q.flags or []),
                    "store_reason": reason,
                }

        flags = list(q.flags or [])
        if flags:  
            metadata["flags"] = flags

        docs.append(
            Document(
                page_content=content,
                metadata = metadata,
            )
        )
        ids.append(doc_id)

    if not docs:
        return 0

    gen_vs.add_documents(docs, ids=ids)

    try:
        gen_vs.persist()
    except Exception:
        pass

    return len(docs)

# ---------------------------
# 4) Node wrapper for LangGraph
# ---------------------------
def persist_memory_node(state: TenderState) -> TenderState:
    """
    Persist:
    - run audit JSON (always best-effort)
    - generated long-term memory (conditional)
    """
    # try:
    audit_path = write_run_audit(state)

    # except Exception as e:
    #     # never fail the pipeline due to audit
    #     for q in state.enriched:
    #         q.flags = sorted(set(q.flags or []) | {"AUDIT_WRITE_ERROR"})
    #     # keep state running

    # try:
    stored = ingest_generated_answers(state)

    # except Exception:
    #     for q in state.enriched:
    #         q.flags = sorted(set(q.flags or []) | {"GENERATED_MEMORY_WRITE_ERROR"})

    return state
