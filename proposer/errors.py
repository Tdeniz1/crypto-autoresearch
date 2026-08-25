class RungError(RuntimeError):
    """A rung failed for a non-quota reason (auth, network, bad output)."""
    def __init__(self, msg: str, kind: str = "error"):
        super().__init__(msg)
        self.kind = kind


class RungQuotaError(RungError):
    """429 / quota / rate-limit / credits exhausted."""
    def __init__(self, msg: str):
        super().__init__(msg, kind="quota")


class RungUnavailable(RungError):
    """Rung cannot be used at all this session (no key, binary missing)."""
    def __init__(self, msg: str):
        super().__init__(msg, kind="unavailable")


QUOTA_MARKERS = ("429", "rate limit", "rate_limit", "ratelimit", "quota", "too many requests",
                 "usage limit", "overloaded", "insufficient credits", "credit", "capacity",
                 "resource_exhausted", "limit reached", "out of extra usage")


def looks_like_quota(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in QUOTA_MARKERS)
