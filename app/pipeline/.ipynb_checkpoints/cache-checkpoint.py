'''Purpose: In-memory caching for dev speed.'''

import hashlib, json
from typing import Any, Dict
from app.pipeline.prompts import build_evidence_pack

# Simple in-memory caches (works great in notebooks)
_CLASSIFY_CACHE: Dict[str, Any] = {}
_VERIFY_CACHE: Dict[str, Any] = {}
_GENERATE_CACHE: Dict[str, str] = {}
_SOFTEN_CACHE: Dict[str, Any] = {}

def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def _norm(text: str) -> str:
    return " ".join((text or "").split()).strip()

def _cache_key_classify(question: str) -> str:
    # normalize to avoid cache misses from whitespace differences
    return _sha(_norm(question))

def _cache_key_verify(question: str, answer: str, evidence: str) -> str:
    # evidence changes should invalidate cache
    payload = {
        "q": _norm(question),
        "a": _norm(answer),
        "e": _norm(evidence),
    }
    return _sha(json.dumps(payload, sort_keys=True))

def _cache_key_generate(q) -> str:
    # Include everything that affects the prompt/output
    payload = {
        "question": _norm(q.question),
        "domain": _norm(q.domain or ""),
        "route": q.route,
        "match_strength": q.match_strength,
        "top_similarity": float(q.top_similarity or 0.0),
        "risk_hints": sorted(list(q.risk_hints or [])),
        # Evidence matters for HISTORICAL_GUIDED
        "evidence": _norm(build_evidence_pack(q)) if q.route == "HISTORICAL_GUIDED" else "",
    }
    return _sha(json.dumps(payload, sort_keys=True))

def _cache_key_soften(question: str, domain: str, draft: str) -> str:
    payload = {
        "q": _norm(question),
        "d": _norm(domain or ""),
        "draft": _norm(draft),
    }
    return _sha(json.dumps(payload, sort_keys=True))

def cache_stats():
    return {
        "classify": len(_CLASSIFY_CACHE),
        "verify": len(_VERIFY_CACHE),
        "generate": len(_GENERATE_CACHE),
        "soften": len(_SOFTEN_CACHE),
    }

def cache_clear():
    _CLASSIFY_CACHE.clear()
    _VERIFY_CACHE.clear()
    _GENERATE_CACHE.clear()
    _SOFTEN_CACHE.clear()