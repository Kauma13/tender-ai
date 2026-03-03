# app/pipeline/guardrails.py
import re
from typing import List, Tuple

## High Risk - "CERTIFICATION_CLAIM", "SLA_PROMISE", "LEGAL_COMPLIANCE"

def apply_guardrails(question: str, draft: str, risk_hints: List[str]) -> tuple[str, List[str]]:
    """
    Modify answer if it risks fabricating claims.
    Returns (safe_answer, added_flags)

    Deterministic safety layer.
    - If question is high-risk, prevent fabricated specifics.
    - Adds flags when it detects dangerous assertions.
    """
    flags: List[str] = []

    q = question.lower()
    a = (draft or "").strip()

    # Guardrail 1: Certification claim - avoid inventing certificate numbers/dates
    if "CERTIFICATION_CLAIM" in risk_hints:
        # if question asks for certificate number/expiry -> must be cautious
        if re.search(r"(certificate number|expiry|expiration|valid until)", q):
            flags.append("NEEDS_EVIDENCE")
            flags.append("HIGH_RISK")
            a = (
                "We can provide relevant security certifications and supporting evidence as part of the formal due diligence process "
                "and subject to confidentiality. Certification details (e.g., certificate number and expiry) are provided through "
                "official documentation channels upon request."
            )

    # Guardrail 2: SLA promise - avoid hard guarantees unless supported
    if "SLA_PROMISE" in risk_hints:
        # remove/avoid hard guarantee language
        if re.search(r"\b(guarantee|guaranteed|100%|always)\b", a, flags=re.IGNORECASE):
            flags.append("HIGH_RISK")
            a = re.sub(r"\bguarantee(d)?\b", "target", a, flags=re.IGNORECASE)

        # If question explicitly asks for uptime commitment, be conditional
        if "uptime" in q or "availability" in q:
            flags.append("NEEDS_CONTRACTUAL_CONFIRMATION")
            a = (
                "Availability and service levels are agreed based on the selected hosting and support plan and are formalised contractually. "
                "Uptime is measured using agreed monitoring windows and reported through service reporting. Planned maintenance exclusions and "
                "measurement definitions are documented in the service agreement."
            )

    return a, sorted(set(flags))