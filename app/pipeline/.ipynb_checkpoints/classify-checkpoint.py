'''Purpose: Domain assignment + risk hints, with fallback strategy.'''

import re
from typing import Dict, List, Set, Literal
from pydantic import BaseModel, Field

from app.models.state import EnrichedQuestion
from app.pipeline.cache import _CLASSIFY_CACHE, _cache_key_classify  

from app.core.llm import get_llm

DOMAINS = [
    "Security & Compliance",
    "Architecture",
    "Data Governance",
    "Delivery",
    "Operations & Support",
    "AI & Data",
    "Commercial & Risk",
    "Change & Training",
    "General",
]

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Security & Compliance": ["security", "iso", "soc", "encrypt", "encryption", "tls", "aes", "incident", "breach", "vulnerability", "pen", "penetration", "audit", "compliance"],
    "Architecture": ["architecture", "deployment", "saas", "on-prem", "hybrid", "components", "integration", "api", "identity", "sso"],
    "Data Governance": ["pii", "privacy", "data residency", "residency", "retention", "deletion", "offboarding", "data hosting", "subcontractor", "subprocessor"],
    "Delivery": ["implementation", "timeline", "plan", "methodology", "governance model", "roles", "responsibilities", "reporting cadence", "project"],
    "Operations & Support": ["support", "escalation", "response time", "sla", "uptime", "availability", "monitoring", "service desk"],
    "AI & Data": ["ai", "ml", "machine learning", "llm", "rag", "hallucination", "evaluation", "data leakage", "model"],
    "Commercial & Risk": ["assumption", "constraint", "dependency", "risk", "mitigation", "commercial", "pricing"],
    "Change & Training": ["change", "training", "adoption", "comms", "communications", "stakeholder"],
}

## Total Risks
AllowedRisk = Literal["CERTIFICATION_CLAIM", "SLA_PROMISE", "LEGAL_COMPLIANCE"]


# Risk patterns (regex)
RISK_PATTERNS = [
    ("CERTIFICATION_CLAIM", r"\b(iso\s*27001|soc\s*2|certificat(e|ion)|certified)\b"),
    ("SLA_PROMISE", r"\b(uptime|availability|sla|response time|guarantee)\b"),
    ("LEGAL_COMPLIANCE",
     r"\b(gdpr|privacy act|australian privacy principles|notifiable data breaches|ndb scheme|"
     r"oaic|apra|cps\s*234|cps\s*230|pci\s*dss|hipaa|sox|glba)\b"),
    ("COMPLIANCE_GENERAL", r"\b(regulatory|regulation|legislation|legal requirements)\b"),
]

def detect_risks(question: str) -> List[str]:
    q = (question or "").lower()
    hits = []
    for label, pat in RISK_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            hits.append(label)
    return sorted(set(hits))

def validate_llm_risks(question: str, llm_risks: list[str]) -> list[str]:
    rule_risks = set(detect_risks(question))  # deterministic truth
    return sorted(set(llm_risks) & rule_risks)
    

def keyword_hit(text: str, kw: str) -> bool:
    kw = kw.strip().lower()
    if len(kw) <= 4:  # short keywords -> word boundary to avoid false positives
        return re.search(rf"\b{re.escape(kw)}\b", text) is not None
    return kw in text


class DomainDecision(BaseModel):
    domain: str
    confidence: float
    rationale: str
    used: str  # rules/llm
    risk_hints: List[str] = Field(default_factory=list)

# @dataclass
# class DomainDecision:
#     domain: str
#     confidence: float
#     rationale: str
#     used: str  # "rules" or "llm"
#     risk_hints: List[str]


# ---------------- Rules classifier ----------------
def rule_classify(question: str) -> DomainDecision:
    q = (question or "").lower()

    scores: Dict[str, int] = {d: 0 for d in DOMAINS}
    
    # for domain, kws in DOMAIN_KEYWORDS.items():
    #     for kw in kws:
    #         if kw in q:
    #             scores[domain] += 1

    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if keyword_hit(q, kw):
                scores[domain] += 1

    # pick the best non-General
    best_domain = max(scores, key=lambda d: scores[d])
    best_score = scores[best_domain]

    # determine ambiguity: low score or tie
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2 = sorted_scores[:2]
    tie = len(top2) == 2 and top2[0][1] == top2[1][1] and top2[0][1] > 0

    risk_hints = detect_risks(question)

    if best_score == 0:
        return DomainDecision(
            domain="General",
            confidence=0.40,
            rationale="No strong keyword signal; defaulting to General.",
            used="rules",
            risk_hints=risk_hints,
        )

    if tie or best_score == 1:
        # ambiguous; return a low-confidence rules decision (LLM can refine)
        return DomainDecision(
            domain=best_domain,
            confidence=0.55,
            rationale=f"Keyword signal is weak/ambiguous (top score={best_score}).",
            used="rules",
            risk_hints=risk_hints,
        )

    # strong enough
    conf = min(0.90, 0.60 + 0.10 * best_score)  # simple scaling
    return DomainDecision(
        domain=best_domain,
        confidence=conf,
        rationale=f"Matched {best_score} domain keywords for '{best_domain}'.",
        used="rules",
        risk_hints=risk_hints,
    )


# ---------------- LLM classifier (structured output) + Risk ---------------
class RiskEvidenceItem(BaseModel):
    hint: AllowedRisk = Field(description="The risk hint label")
    evidence: str = Field(description="Exact phrase from the question that triggered this hint")

class LLMClassification(BaseModel):
    domain: str = Field(description=f"One of: {', '.join(DOMAINS)}")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0..1")
    rationale: str = Field(description="Short explanation (1-2 lines)")
    risk_hints: List[AllowedRisk] = Field(default_factory=list)
    risk_evidence: List[RiskEvidenceItem] = Field(default_factory=list)


def llm_classify(question: str) -> DomainDecision:
    ALLOWED_RISKS = {"CERTIFICATION_CLAIM", "SLA_PROMISE", "LEGAL_COMPLIANCE"}
    
    system = (
        "You are classifying tender questionnaire items into a fixed domain taxonomy. "
        "Return only the requested structured fields."
    )

    user = f"""
    Classify the following question into one domain from this list:
    {DOMAINS}
    
    Now detect risk hints ONLY if the question explicitly contains evidence for them.

    Definitions (evidence-gated):
    - CERTIFICATION_CLAIM: ONLY if the question explicitly mentions a certification or audit standard
      (e.g., ISO 27001, SOC 2, "certified", "certification", "certificate number", "audit report").
    - SLA_PROMISE: ONLY if the question explicitly asks for uptime/availability/SLA/guarantees/response times.
    - LEGAL_COMPLIANCE: ONLY if the question explicitly names a law/regulation/standard
      (e.g., GDPR, Privacy Act, Australian Privacy Principles, APRA CPS 234, PCI DSS, HIPAA).
    
    Important:
    - If the evidence is not present in the question text, return an empty risk_hints list [].
    - Do not mention specific external standards bodies (e.g., NIST) and similar in context, unless the question explicitly mentions them.
    - Do NOT infer risk hints from general security wording.
    - Return only the structured fields.
    - Do not assume if it is a risk. Check carefully, understand it and then take the decision.
    - For each risk hint you output, include the exact triggering phrase in risk_evidence.
    
    Question:
    {question}
    """.strip()

    gen_llm = get_llm()
    
    structured = gen_llm.with_structured_output(LLMClassification)
    out: LLMClassification = structured.invoke([("system", system), ("user", user)])
    
    # Hard safety: if model outputs unknown domain, clamp to General
    domain = out.domain if out.domain in DOMAINS else "General"

    # Keep only allowed risks
    llm_risks = [r for r in out.risk_hints if r in ALLOWED_RISKS]

    # keep only risks with evidence: keep only risks that have non-empty evidence span
    evidence_map = {item.hint: item.evidence.strip() for item in out.risk_evidence if item.evidence and item.evidence.strip()}
    llm_risks = [r for r in llm_risks if evidence_map.get(r)]

    return DomainDecision(
        domain=domain,
        confidence=float(out.confidence),
        rationale=out.rationale.strip(),
        used="llm",
        risk_hints=sorted(set(llm_risks)),
    )


def llm_classify_cached(question: str) -> DomainDecision:
    key = _cache_key_classify(question)
    if key in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[key]

    out = llm_classify(question)          
    _CLASSIFY_CACHE[key] = out
    return out


# ---------------- Hybrid classifier including Cache and Validation ----------------
def hybrid_classify(question: str) -> DomainDecision:
    rule_decision = rule_classify(question)

    # Deterministic risks from question only (single source of truth)
    rule_risks = detect_risks(question)
    rule_decision.risk_hints = rule_risks

    # if rules are strong enough, skip LLM
    if rule_decision.confidence >= 0.75 and rule_decision.domain != "General":
        return rule_decision

    # Otherwise ask LLM (domain), but accept risks only if validated by rules
    try:
        # llm_decision = llm_classify(question)
        llm_decision = llm_classify_cached(question) #cache first

        validated_llm_risks = validate_llm_risks(question, llm_decision.risk_hints)

        # merge (safe): rules risks + validated LLM risks
        llm_decision.risk_hints = sorted(set(rule_risks) | set(validated_llm_risks))

        return llm_decision

    except Exception as e:
        fallback = rule_decision
        fallback.rationale += f" (LLM fallback due to error: {e})"
        return fallback