# app/pipeline/soften.py
from pydantic import BaseModel, Field
from app.core.llm import get_llm

class CautiousRewrite(BaseModel):
    rewritten_answer: str
    removed_or_softened_claims: list[str] = []

def llm_soften_answer(question: str, domain: str, answer: str) -> CautiousRewrite:

    system = (
        "You are a senior tender writer. Rewrite answers to be cautious and non-committal when evidence is limited. "
        "Do not add new facts."
    )

    user = f"""
        Rewrite the answer to be safe for a tender response with limited evidence.
        
        Hard constraints (must follow):
        - Do NOT use commitment language like: "we will", "we are committed to", "we are developing", "we are integrating", "we guarantee".
        - Prefer conditional phrasing: "can", "may", "subject to", "where required", "can be assessed during discovery", "available on request".
        - Do NOT introduce any new capabilities, certifications, SLAs, audits, or legal compliance claims.
        - Keep it concise and professional.
        
        Return:
        1) rewritten_answer
        2) removed_or_softened_claims: list the exact phrases you softened/removed.
        
        Question:
        {question}
        
        Domain:
        {domain}
        
        Original Answer:
        {answer}
        """.strip()

    structured = get_llm.with_structured_output(CautiousRewrite)
    return structured.invoke([("system", system), ("user", user)])