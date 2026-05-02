import threading
from cachetools import TTLCache

DEFAULT_PARAMS = {
    "town": None,
    "block": None,
    "storey_range": None,
    "floor_area_sqm": None,
    "street_name": None,
}

REQUIRED_PARAMS = {"town", "block", "storey_range", "floor_area_sqm"}

# Gemini sometimes uses alternate keys — map into our schema before /predict.
_PARAM_ALIASES = {
    "floor_area": "floor_area_sqm",
    "sqm": "floor_area_sqm",
    "street": "street_name",
    "road": "street_name",
}


def _coerce_merged_value(key: str, val):
    """Best-effort types for backend PredictRequest (avoids 422 from bad LLM JSON)."""
    if val is None:
        return None
    if key == "floor_area_sqm":
        if isinstance(val, bool):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val.replace(",", "").strip())
            except ValueError:
                return None
        return None
    if key in ("town", "block", "storey_range", "street_name"):
        if isinstance(val, str):
            s = val.strip()
            return s if s else None
        return str(val).strip() if val is not None else None
    return val


class SessionCache:
    """
    Thread-safe in-process TTL cache for Telegram conversation sessions.
    maxsize=500: supports ~500 concurrent active sessions.
    ttl=3600:    sessions expire after 1 hour of inactivity.
    """

    def __init__(self, maxsize: int = 500, ttl: int = 3600):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    def _default_state(self) -> dict:
        return {
            "history": [],
            "collected_params": dict(DEFAULT_PARAMS),
        }

    def get(self, chat_id: int) -> dict:
        with self._lock:
            if chat_id not in self._cache:
                self._cache[chat_id] = self._default_state()
            return self._cache[chat_id]

    def clear(self, chat_id: int) -> None:
        with self._lock:
            self._cache.pop(chat_id, None)

    def append_history(self, chat_id: int, role: str, content: str) -> None:
        with self._lock:
            state = self._cache.setdefault(chat_id, self._default_state())
            state["history"].append({"role": role, "content": content})
            # Keep last 30 turns to cap memory usage
            state["history"] = state["history"][-30:]

    def merge_params(self, chat_id: int, new_params: dict) -> None:
        """Merge extracted params — null values do NOT overwrite existing values."""
        with self._lock:
            state = self._cache.setdefault(chat_id, self._default_state())
            for key, val in new_params.items():
                key = _PARAM_ALIASES.get(key, key)
                if key not in state["collected_params"]:
                    continue
                coerced = _coerce_merged_value(key, val)
                if coerced is not None:
                    state["collected_params"][key] = coerced

    def is_complete(self, chat_id: int) -> bool:
        """True when all required params are present and satisfy backend validation."""
        with self._lock:
            state = self._cache.get(chat_id, self._default_state())
            params = state["collected_params"]
            for k in REQUIRED_PARAMS:
                v = params.get(k)
                if v is None:
                    return False
                if k in ("town", "block", "storey_range"):
                    if not isinstance(v, str) or not v.strip():
                        return False
                if k == "floor_area_sqm":
                    if not isinstance(v, (int, float)):
                        return False
                    fv = float(v)
                    if not (20.0 <= fv <= 300.0):
                        return False
            return True
