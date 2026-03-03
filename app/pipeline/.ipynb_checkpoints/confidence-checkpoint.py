# app/pipeline/confidence.py
from typing import List
import math

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def confidence_from_similarity(
    s1: float,
    match_strength: str,
    flags: list[str],
    k: float = 16.0,
    t_strong: float = 0.70,
    t_weak: float = 0.60,
) -> float:
    ''' Penalise if evidence missing / needs evidence'''
    
    # Evidence mapping (calibrated curve)
    t = t_strong if match_strength == "STRONG" else t_weak
    conf = sigmoid(k * (s1 - t))  # 0..1

    # Risk caps (deterministic)
    if "NEEDS_EVIDENCE" in flags:
        conf = min(conf, 0.40)
    if "HIGH_RISK" in flags:
        conf = min(conf, 0.30)

    # Route / evidence bands (deterministic, easy to explain)
    if match_strength == "STRONG":
        conf = max(conf, 0.65)
        conf = min(conf, 0.90)
    elif match_strength == "WEAK":
        conf = max(conf, 0.40)
        conf = min(conf, 0.75)
    else:  # NONE
        conf = max(conf, 0.20)
        conf = min(conf, 0.45)

    # Final clamp
    conf = max(0.05, min(conf, 0.95))
    return round(conf, 2)