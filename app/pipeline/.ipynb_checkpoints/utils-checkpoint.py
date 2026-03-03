''' Common deterministic utilities. Used for stable Chroma document IDs.'''

import hashlib

def stable_id(*parts: str) -> str:
    raw = "||".join(p.strip() for p in parts if p is not None)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()