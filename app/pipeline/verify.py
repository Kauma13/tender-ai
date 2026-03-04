'''Verifies generated answers against evidence (only when evidence exists).'''


from typing import List, Optional, Literal
from pydantic import BaseModel, Field

import re

from app.models.state import TenderState, EnrichedQuestion
from app.pipeline.prompts import build_evidence_pack
from app.core.llm import get_llm
from app.pipeline.cache import _VERIFY_CACHE, _cache_key_verify
from app.core.config import settings
from app.core.logger import logger


VerifierLabel = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTS_EVIDENCE",
]

ASSERTION_PAT = re.compile(r"\b(we are|we have|we hold|we guarantee|we comply|we use)\b", re.I)
CONDITIONAL_PAT = re.compile(r"\b(can|may|might|could|upon request|due diligence|subject to)\b", re.I)

class VerificationResult(BaseModel):
    label: VerifierLabel
    support_score: float = Field(ge=0.0, le=1.0, description="How well the evidence supports the answer")
    missing_points: List[str] = Field(default_factory=list, description="What info is missing from evidence")
    risky_claims: List[str] = Field(default_factory=list, description="Claims that require evidence or sound like commitments")
    evidence_quotes: List[str] = Field(default_factory=list, description="Short snippets from evidence supporting/contradicting (<= 2 snippets)")
    rationale: str = Field(description="1-3 sentences explaining the judgment")


def is_true_overclaim(claim: str) -> bool:
    claim = (claim or "").strip()
    if not claim:
        return False

    # must look like an assertion
    if not ASSERTION_PAT.search(claim):
        return False

    # conditional language cancels overclaim
    if CONDITIONAL_PAT.search(claim):
        return False

    return True

  
def verify_answer_against_evidence(q: EnrichedQuestion) -> VerificationResult:
    """
    Runs only when we have historical evidence (HISTORICAL_GUIDED).
    """
    evidence = build_evidence_pack(q)
    answer = (q.answer or "").strip()
    question = q.question

    system = (
        "You are a strict auditor for a tender-response assistant. "
        "Judge whether the provided answer is supported by the provided evidence. "
        "Do NOT assume facts not in evidence."
    )
    user = f"""
        Task:
        Given a tender QUESTION, a DRAFT ANSWER, and EVIDENCE (historical Q/A snippets),
        assess whether the answer is supported by the evidence.
        
        Output requirements:
        - Be strict and evidence-grounded.
        - When listing risky_claims: ONLY list claims you can quote directly from the DRAFT ANSWER (copy the exact phrase).
        - Do NOT list generic concerns. If there are no quotable risky claims, return risky_claims=[].
        
        Labels:
        - SUPPORTED: Evidence supports the main claims in the answer.
        - PARTIALLY_SUPPORTED: Evidence supports some claims, but the answer contains additional unverified specifics.
        - UNSUPPORTED: Evidence does not support the key claims (mostly generic / not grounded).
        - CONTRADICTS_EVIDENCE: Answer conflicts with evidence.
        
        Decision rules:
        1) CONTRADICTS_EVIDENCE if the answer states the opposite of evidence.
        2) If the answer introduces extra details not in evidence, choose PARTIALLY_SUPPORTED (not UNSUPPORTED) unless the extra details are the key part of the answer.
        3) Use UNSUPPORTED only when evidence does not support the main point at all.
        
        OVERCLAIM (single strict definition):
        - Flag OVERCLAIM ONLY when the DRAFT ANSWER contains an UNQUALIFIED assertion of possession/guarantee/compliance/certification/standard
          that is NOT present in evidence.
        - UNQUALIFIED assertion means confident factual language like:
          "we are", "we have", "we hold", "we guarantee", "we comply", "we use <specific standard/version/value>".
        
        NOT OVERCLAIM:
        - Conditional / cautious language such as:
          "can be provided during due diligence", "upon request", "subject to confidentiality",
          "may include", "we monitor/evaluate/explore", "industry-standard" (without numbers/versions).
        - Specifically: "We can provide certification documentation during due diligence" is NOT an overclaim.
        
        If the answer is cautious and does not claim possession/guarantee, do NOT flag OVERCLAIM.
        
        QUESTION:
        {question}
        
        DRAFT ANSWER:
        {answer}
        
        EVIDENCE:
        {evidence}
        """.strip()

    print("OpenAI Verifier initialized")

    gen_llm = get_llm()

    structured = gen_llm.with_structured_output(VerificationResult)
    print("OpenAI Verifier done")
    return structured.invoke([("system", system), ("user", user)])

def verify_answer_cached(q: EnrichedQuestion) -> VerificationResult:
    """
    Wraps verify_answer_against_evidence(q) with caching.
    Cache key includes question + answer + evidence pack.
    """
    evidence = build_evidence_pack(q)
    k = _cache_key_verify(q.question, q.answer or "", evidence)
    
    if settings.enable_cache and k in _VERIFY_CACHE:
        return _VERIFY_CACHE[k]
        
    out = verify_answer_against_evidence(q)
    
    if settings.enable_cache:
        _VERIFY_CACHE[k] = out
    return out


def apply_verification(q: EnrichedQuestion, v: VerificationResult) -> None:
    flags = set(q.flags or [])

    # 1) Map labels to flags
    if v.label == "CONTRADICTS_EVIDENCE":
        flags.add("INCONSISTENT_WITH_EVIDENCE")
    elif v.label == "UNSUPPORTED":
        flags.add("UNSUPPORTED_BY_EVIDENCE")
    elif v.label == "PARTIALLY_SUPPORTED":
        flags.add("PARTIALLY_SUPPORTED_BY_EVIDENCE")

    # Overclaim detection from verifier (deterministic filter)
    true_overclaims = [c for c in (v.risky_claims or []) if is_true_overclaim(c)]
    if true_overclaims:
        flags.add("OVERCLAIM")

    # 2) Alignment policy (keep STRONG matches high unless support_score is genuinely low)
    if v.label == "SUPPORTED":
        pass
    elif v.label == "PARTIALLY_SUPPORTED":
        if q.alignment == "HIGH" and float(v.support_score) < 0.75:
            q.alignment = "MED"
    else:  # UNSUPPORTED or CONTRADICTS_EVIDENCE
        q.alignment = "LOW"

    # 3) Confidence calibration (single clear step)
    base = float(q.confidence or 0.0)
    conf = 0.6 * base + 0.4 * float(v.support_score)

    # 4) Small deterministic deltas
    if v.label == "PARTIALLY_SUPPORTED":
        conf -= 0.05
    elif v.label == "UNSUPPORTED":
        conf -= 0.15
    elif v.label == "CONTRADICTS_EVIDENCE":
        conf -= 0.25

    if "OVERCLAIM" in flags:
        conf -= 0.08

    # Clamp
    conf = max(0.05, min(0.95, conf))

    # Evidence-required cap (prevents high confidence on claims needing proof)
    if "NEEDS_EVIDENCE" in flags:
        conf = min(conf, 0.40)

    q.confidence = round(conf, 2)
    q.flags = sorted(flags)


def verify_answers_node(state: TenderState) -> TenderState:
    out = []

    for q in state.enriched:
        try:
            should_verify = (
                q.route == "HISTORICAL_GUIDED"
                and bool(q.matches)
                and q.top_similarity >= state.weak_threshold
                and bool(q.answer and q.answer.strip())
            )

            if should_verify:
                
                # v = verify_answer_against_evidence(q)
                v = verify_answer_cached(q)  # cached
                apply_verification(q, v)

                # Force SME review for any serious verifier outcome
                flags = set(q.flags or [])
                
                if v.label in {"UNSUPPORTED", "CONTRADICTS_EVIDENCE"}:
                    flags.add("REQUIRES_SME_REVIEW")
                    
                if "INCONSISTENT_WITH_EVIDENCE" in flags:
                    flags.add("REQUIRES_SME_REVIEW")

                q.flags = sorted(flags)

            out.append(q)

        except Exception as e:
            q.flags = sorted(set(q.flags or []) | {"VERIFICATION_ERROR"})
            q.error = f"verification_error: {e}"
            out.append(q)

    state.enriched = out
    return state
