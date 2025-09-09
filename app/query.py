import re
from typing import Optional, List

EARBUD_SYNS = ["earbuds", "earbud", "tws", "true wireless", "buds"]

def _parse_price(text: str) -> Optional[float]:
    t = text.lower().replace(",", "").strip()
    # under/below/up to 5000 / 5k
    m = re.search(r"(under|below|<=|less than|up to|upto)\s*₹?\s*([0-9]+(?:\.[0-9]+)?k?)", t)
    if not m:
        m = re.search(r"₹?\s*([0-9]+(?:\.[0-9]+)?k?)\s*(?:or )?(?:less|below|under)", t)
    if not m:
        m = re.search(r"₹?\s*([0-9]+(?:\.[0-9]+)?)\s*k\b", t)  # e.g. 5k
    if m:
        val = m.group(2) if len(m.groups()) >= 2 else m.group(1)
        return float(val[:-1]) * 1000 if val.endswith("k") else float(val)
    # fallback: any standalone number that looks like a budget
    m = re.search(r"₹?\s*([4-9][0-9]{2,4})\b", t)
    return float(m.group(1)) if m else None

def _parse_category(text: str) -> Optional[str]:
    t = text.lower()
    for s in EARBUD_SYNS:
        if s in t:
            return "Earbuds"
    return None

def _must_terms(text: str) -> List[str]:
    t = text.lower()
    terms = []
    if any(x in t for x in ["mic", "microphone", "calls", "calling"]): terms.append("mic")
    if "anc" in t or "noise cancel" in t: terms.append("anc")
    if "bluetooth" in t or "bt" in t: terms.append("bluetooth")
    return terms

def parse_query(text: str):
    return {
        "max_price": _parse_price(text),
        "category": _parse_category(text),
        "terms": _must_terms(text),
    }
