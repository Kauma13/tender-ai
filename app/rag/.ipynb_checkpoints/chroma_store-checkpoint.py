'''Chroma store initialization and persistence.This is the “database connector” layer.
Even if you already ingested data earlier, runtime must still “open” the persisted store to query it.'''


import os
from typing import Any, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.models.state import RetrievedMatch
from app.pipeline.utils import stable_id


def get_field(obj, name: str, default=None):
    """
    Works for dicts, Pydantic models (v1/v2), dataclasses, and plain objects.
    """
    if obj is None:
        return default

    # dict
    if isinstance(obj, dict):
        return obj.get(name, default)

    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump().get(name, default)
        except Exception:
            pass

    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict().get(name, default)
        except Exception:
            pass

    # plain object / dataclass
    return getattr(obj, name, default)

def init_stores(embeddings):
    os.makedirs(settings.persist_dir, exist_ok=True)
    hist = Chroma(
        collection_name=settings.hist_collection,
        persist_directory=settings.persist_dir,
        embedding_function=embeddings,
    )
    gen = Chroma(
        collection_name=settings.gen_collection,
        persist_directory=settings.persist_dir,
        embedding_function=embeddings,
    )
    return hist, gen

def ingest_historical_rows(rows: List[Any], vs) -> int:

    print("Enter")
    docs: List[Document] = []
    ids: List[str] = []

    for r in rows:
        source_id = str(get_field(r, "source_id", "HIST")).strip()
        domain = str(get_field(r, "domain", "General")).strip()
        q = str(get_field(r, "question", "")).strip()
        a = str(get_field(r, "answer", "")).strip()

        if not q or not a:
            continue

        # stable per-question id (not just source_id)
        question_id = f"{source_id}_{stable_id(q)[:8]}"

        content = f"QUESTION:\n{q}\n\nANSWER:\n{a}\n"

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "question_id": question_id,
                    "source_id": source_id,
                    "domain": domain,
                    "historical_question": q,
                    "historical_answer": a,
                    "kind": "historical",
                },
            )
        )

        # stable doc id for Chroma (prevents duplicates)
        doc_id = stable_id("historical", source_id, q, a)
        ids.append(doc_id)

    if not docs:
        return 0

    vs.add_documents(docs, ids=ids)
    try:
        vs.persist()
    except Exception:
        pass

    return len(docs)

def retrieve_top_k(vs: Chroma, query: str, k: int) -> List[Tuple[Document, float]]:
    return vs.similarity_search_with_relevance_scores(query, k=k)

def hits_to_matches(hits: List[Tuple[Document, float]]) -> List[RetrievedMatch]:

    # hits = vs.similarity_search_with_score("information security management", k=3)
    # for doc, score in hits:
    #     print("score:", score)
    #     print("meta keys:", sorted(doc.metadata.keys()))
    #     print("historical_question:", doc.metadata.get("historical_question"))
    #     print("question:", doc.metadata.get("question"))
    #     print("---")
    out: List[RetrievedMatch] = []
    for doc, score in hits:
        out.append(
            RetrievedMatch(
                source_id=doc.metadata.get("source_id", ""),
                domain=doc.metadata.get("domain", ""),
                historical_question=doc.metadata.get("historical_question", ""),
                historical_answer=doc.metadata.get("historical_answer", ""),
                similarity=float(score),
            )
        )
    return out



def collection_count(vs: Chroma) -> int:
    try:
        # Chroma exposes collection internally
        return vs._collection.count()
    except Exception:
        return -1





