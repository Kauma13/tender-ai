'''Purpose: Generates answers using either:
    evidence-guided prompt (HISTORICAL_GUIDED) 
    safe template prompt (TEMPLATE_SAFE)
Then applies:
    soften rewrite   
    deterministic guardrails    
    confidence scoring    
    alignment scoring    
    Main steps per question
Error Handler'''

from typing import List

from app.models.state import TenderState, EnrichedQuestion
from app.core.llm import get_generator_llm
from app.pipeline.prompts import build_historical_guided_prompt, build_template_safe_prompt
from app.pipeline.soften import llm_soften_answer
from app.pipeline.guardrails import apply_guardrails
from app.pipeline.confidence import confidence_from_similarity
from app.pipeline.cache import _GENERATE_CACHE, _cache_key_generate, _cache_key_soften 


def compute_alignment(q: EnrichedQuestion) -> str:
    if q.match_strength == "STRONG":
        return "HIGH"
    if q.match_strength == "WEAK":
        return "MED"
    return "LOW"

def generate_draft_cached(q, prompt: str) -> str:
    gen_llm = get_generator_llm()
    key = _cache_key_generate(q)
    
    if key in _GENERATE_CACHE:
        return _GENERATE_CACHE[key]

    draft = gen_llm.invoke(prompt).content.strip()
    _GENERATE_CACHE[key] = draft
    return draft


def llm_soften_answer_cached(question: str, domain: str, draft: str):
    """
    Caches the output of llm_soften_answer(question, domain, draft).
    Returns the same object your llm_soften_answer returns (e.g., with rewritten_answer field).
    """
    key = _cache_key_soften(question, domain, draft)
    
    if key in _SOFTEN_CACHE:
        return _SOFTEN_CACHE[key]

    out = llm_soften_answer(question, domain, draft)  
    _SOFTEN_CACHE[key] = out
    return out


def generate_answers_node(state: TenderState) -> TenderState:
    enriched_out: List[EnrichedQuestion] = []

    for q in state.enriched:
        try:
            # Start flags with routing information
            # flags = list(q.flags)
            flags = list(q.flags or [])

            # Set match-based flags (useful later)
            if q.match_strength == "NONE":
                if "NO_STRONG_HISTORICAL_MATCH" not in flags:
                    flags.append("NO_STRONG_HISTORICAL_MATCH")

            # Branch prompt
            if q.route == "HISTORICAL_GUIDED":
                prompt = build_historical_guided_prompt(q)
            else:
                prompt = build_template_safe_prompt(q)

            # LLM draft
            # draft = gen_llm.invoke(prompt).content.strip()
            draft = generate_draft_cached(q, prompt)

            # Apply LLM cautious rewrite for weak or no historical matches
            if q.route == "TEMPLATE_SAFE" or q.match_strength == "NONE":
                try:
                    rewrite = llm_soften_answer_cached(q.question, q.domain, draft)
                    draft = rewrite.rewritten_answer.strip()
                    # rewrite = llm_soften_answer(q.question, q.domain, draft)
                    # draft = rewrite.rewritten_answer.strip()
                except Exception:
                    pass

            # Apply deterministic guardrails
            safe_answer, extra_flags = apply_guardrails(q.question, draft, q.risk_hints)
            flags = sorted(set(flags) | set(extra_flags))

            # If guardrails were triggered, we also nudge alignment down slightly
            alignment = compute_alignment(q)
            if "HIGH_RISK" in flags and alignment == "HIGH":
                alignment = "MED"
            
            # Store answer + confidence
            q.answer = safe_answer
            q.flags = flags
            q.alignment = alignment

            # ----- confidence calculation ----- Thresholded evidence confidence
            q.confidence = confidence_from_similarity(
                s1=q.top_similarity,
                match_strength=q.match_strength,
                flags=q.flags
            )
            enriched_out.append(q)

        except Exception as e:
            q.error = str(e)
            q.flags = sorted(set(q.flags) | {"GENERATION_ERROR"})
            q.answer = ""
            q.confidence = 0.0
            q.alignment = "LOW"
            enriched_out.append(q)

    state.enriched = enriched_out
    return state