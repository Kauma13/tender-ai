'''Purpose: Centralize prompt construction and evidence formatting.'''

from app.models.state import EnrichedQuestion

DOMAIN_TEMPLATES = {
    "Security & Compliance": (
        "We follow established information security practices including governance, access control, "
        "logging/monitoring, vulnerability management, and incident response. Specific controls and "
        "evidence (e.g., policies, audit reports) can be provided during due diligence, subject to confidentiality."
    ),
    "Architecture": (
        "We propose a modular architecture aligned to the required deployment model (SaaS/on-prem/hybrid). "
        "Key components include identity/access management, integration interfaces (APIs), data storage, and reporting. "
        "A detailed architecture can be provided following discovery and solution design."
    ),
    "Data Governance": (
        "We manage data securely using least-privilege access, auditing, and secure handling processes. "
        "Data residency, retention, deletion, and offboarding are aligned to customer requirements and applicable regulations. "
        "Subprocessors and hosting details can be disclosed during due diligence."
    ),
    "Delivery": (
        "We deliver using a phased approach: kickoff and planning, requirements confirmation, configuration/integration, "
        "testing (SIT/UAT), training, and go-live readiness. Timelines depend on scope stability, stakeholder availability, "
        "and environment readiness."
    ),
    "Operations & Support": (
        "We provide a tiered support model with defined escalation paths. Service levels (response targets, availability) "
        "are agreed based on the selected support plan and are formalised contractually. Monitoring and reporting are provided "
        "through agreed service reporting."
    ),
    "AI & Data": (
        "Where AI is used, it is governed with access controls, logging, and evaluation. Outputs are reviewed by SMEs where required. "
        "Safeguards include curated test sets, prompt/data controls, and measures to reduce hallucinations and prevent sensitive data leakage."
    ),
    "Commercial & Risk": (
        "We document assumptions, constraints, dependencies, and risks early and review them regularly through governance forums. "
        "Mitigations include staged delivery, clear change control, and early integration and data readiness activities."
    ),
    "Change & Training": (
        "We support change through stakeholder engagement, communications planning, role-based training, and readiness checks. "
        "Training materials and sessions are aligned to user roles, with reinforcement and feedback loops around go-live."
    ),
    "General": (
        "We can address this requirement based on discovery and confirmation of scope, constraints, and stakeholder expectations. "
        "Further detail can be provided during due diligence and solution design."
    ),
}

def build_evidence_pack(q: EnrichedQuestion, max_items: int = 3) -> str:
    """
    q is EnrichedQuestion
    Returns a compact evidence string of top-k matches.
    """
    
    lines = []
    for i, m in enumerate((q.matches or [])[:max_items], 1):
        lines.append(
            f"[{i}] source={m.source_id} domain={m.domain} sim={m.similarity:.3f}\n"
            f"Q: {m.historical_question}\n"
            f"A: {m.historical_answer}\n"
        )
    return "\n".join(lines).strip()


def build_historical_guided_prompt(q: EnrichedQuestion) -> str:
    # top matches as evidence
    evidence = build_evidence_pack(q)

    return f"""
    You are generating a tender response. Use the historical answers as guidance and remain consistent with them.
    Do NOT invent certifications, certificate numbers, audit results, uptime SLAs, or legal compliance claims unless explicitly supported by the evidence.
    Keep the response professional, concise, and aligned to the question.

    Question:
    {q.question}
    
    Detected Domain: {q.domain}
    Risk Hints: {q.risk_hints}
    
    Historical Evidence:
    {evidence}

    Evidence anchoring rules (must follow):
        - Use ONLY the information present in the EVIDENCE snippets below.
        - Do NOT introduce specific versions/standards/numbers (e.g., TLS 1.2, AES-256, ISO numbers, uptime %) unless they explicitly appear in EVIDENCE.
        - If EVIDENCE is generic, keep the answer generic and state that details can be provided during due diligence / upon request.
        - If the question asks for specifics but EVIDENCE does not contain them, do not guess; instead respond cautiously and request/offer to provide evidence.
        - Do NOT attribute capabilities to a specific platform/vendor/hosting provider unless explicitly stated in EVIDENCE.
        - If evidence does not mention TLS/encryption mechanisms, do NOT mention TLS, "platform-level encryption", "key management", or similar implementation details.
        - Prefer: "encrypted in transit and at rest using industry-standard mechanisms" + "details available upon request".
    
    
    Write a concise, professional answer aligned to the QUESTION and grounded in EVIDENCE. If the evidence is insufficient for a claim, state that details can be provided during due diligence or subject to contract.
    """.strip()


def build_template_safe_prompt(q: EnrichedQuestion) -> str:
    template = DOMAIN_TEMPLATES.get(q.domain, DOMAIN_TEMPLATES["General"])
    
    return f"""
    You are generating a tender response where there is no strong historical match.
    Use the provided safe template as a baseline, adapt it to the question, and keep language cautious.
    Do NOT invent certifications, certificate numbers, audit results, uptime SLAs, or legal compliance claims.
    Do not mention specific external standards bodies (e.g., NIST) and similar in context, unless the question explicitly mentions them.
    
    Question:
    {q.question}
    
    Detected Domain: {q.domain}
    Risk Hints: {q.risk_hints}
    
    Safe Baseline Template:
    {template}
    
    Write a professional response. If specifics are required, state they can be provided during due diligence or agreed contractually.
    """.strip()