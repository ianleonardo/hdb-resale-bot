import threading
from cachetools import TTLCache

DEFAULT_PARAMS = {
    "town": None, "flat_type": None, "flat_model": None,
    "storey_range": None, "floor_area_sqm": None,
    "remaining_lease_years": None, "street_name": None, "block": None,
}


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
                if val is not None and key in state["collected_params"]:
                    state["collected_params"][key] = val

    def is_complete(self, chat_id: int) -> bool:
        with self._lock:
            state = self._cache.get(chat_id, self._default_state())
            return all(v is not None for v in state["collected_params"].values())
