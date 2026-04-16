_OFF_TOPIC = {
    # food & lifestyle
    "recipe", "chicken rice", "bubble tea", "hawker", "makan", "restaurant",
    # tech / coding
    "python", "javascript", "html", "coding", "code", "chatgpt", "openai",
    # finance (non-property)
    "crypto", "bitcoin", "stock market", "shares", "forex", "etf",
    # politics
    "pap", "election", "parliament", "minister", "opposition",
    # other property types
    "condo", "condominium", "bto", "ec property", "private property", "landed",
    "penthouse", "villa", "serviced apartment",
    # misc
    "weather", "football", "soccer", "movie", "song",
    "relationship", "girlfriend", "boyfriend",
}

_HDB_KEYWORDS = {
    "hdb", "flat", "resale", "room", "storey", "floor", "sqm", "lease",
    "block", "street", "town", "price", "estimate", "ang mo kio", "bedok",
    "bishan", "bukit", "central", "choa chu kang", "clementi", "geylang",
    "hougang", "jurong", "kallang", "marine parade", "pasir ris", "punggol",
    "queenstown", "sembawang", "sengkang", "serangoon", "tampines",
    "toa payoh", "woodlands", "yishun", "whampoa",
    "model a", "dbss", "maisonette", "executive", "standard", "improved",
}


def quick_topic_check(message: str) -> str:
    """
    Returns: 'hdb' | 'off_topic' | 'ambiguous'
    'ambiguous' means pass to LLM for nuanced decision.
    """
    lower = message.lower()
    tokens = set(lower.split())

    has_hdb = bool(_HDB_KEYWORDS & tokens) or any(kw in lower for kw in _HDB_KEYWORDS)
    has_off = bool(_OFF_TOPIC & tokens) or any(kw in lower for kw in _OFF_TOPIC)

    if has_hdb and not has_off:
        return "hdb"
    if has_off and not has_hdb:
        return "off_topic"
    return "ambiguous"
