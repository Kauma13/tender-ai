'''Purpose: Adds run statistics to state.summary, such as:
    total questions
    number of errors
    number routed historical vs template
    count of risk-flagged questions
    status -  completed/partial'''

from app.models.state import TenderState

def summarise_node(state: TenderState) -> TenderState:
    s = state.summary

    s.total = len(state.questions)
    s.errors = sum(1 for x in state.enriched if x.error)

    s.routed_historical = sum(1 for x in state.enriched if x.route == "HISTORICAL_GUIDED")
    s.routed_template = sum(1 for x in state.enriched if x.route == "TEMPLATE_SAFE")
    s.flagged_risk = sum(1 for x in state.enriched if x.risk_hints)

    # verifier/quality outcomes (based on flags)
    s.requires_sme_review = sum(1 for x in state.enriched if "REQUIRES_SME_REVIEW" in (x.flags or []))
    s.unsupported = sum(1 for x in state.enriched if "UNSUPPORTED_BY_EVIDENCE" in (x.flags or []))
    s.inconsistent = sum(1 for x in state.enriched if "INCONSISTENT_WITH_EVIDENCE" in (x.flags or []))
    s.overclaim = sum(1 for x in state.enriched if "OVERCLAIM" in (x.flags or []))

    # readiness split
    s.ready_to_submit = sum(
        1 for x in state.enriched
        if not x.error and "REQUIRES_SME_REVIEW" not in (x.flags or [])
    )
    s.needs_review = s.total - s.ready_to_submit

    confs = [float(x.confidence) for x in state.enriched if x.confidence is not None]
    if confs:
        s.confidence_min = round(min(confs), 2)
        s.confidence_avg = round(sum(confs) / len(confs), 2)
        s.confidence_max = round(max(confs), 2)
    else:
        s.confidence_min = s.confidence_avg = s.confidence_max = 0.0

    s.status = "PARTIAL" if s.errors > 0 else "COMPLETED"
    return state
