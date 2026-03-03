'''Purpose: Converts similarity into discrete route decisions.'''

def match_strength(top_sim: float, strong: float, weak: float) -> str:
    if top_sim >= strong:
        return "STRONG"
    if top_sim >= weak:
        return "WEAK"
    return "NONE"

def route_from_strength(strength: str) -> str:
    return "HISTORICAL_GUIDED" if strength in {"STRONG", "WEAK"} else "TEMPLATE_SAFE"
