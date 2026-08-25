"""Push delivery. ntfy.sh (free, no account: subscribe to a topic in the ntfy app) + macOS banner."""
from __future__ import annotations

import platform
import subprocess

import httpx

from live import config

PRIORITY = {"signal": "high", "halt": "urgent", "reminder": "default", "resume": "default", "report": "default"}
TAGS = {"signal": "chart_with_upwards_trend", "halt": "rotating_light", "reminder": "hourglass",
        "resume": "white_check_mark", "report": "clipboard"}


def ascii_only(s: str) -> str:
    """HTTP header values must be ASCII; emoji go through ntfy tags instead."""
    return " ".join(s.encode("ascii", "ignore").decode().split())


def push_ntfy(title: str, body: str, kind: str = "signal") -> tuple[bool, str]:
    if not config.NTFY_TOPIC:
        return False, "NTFY_TOPIC not set"
    try:
        r = httpx.post(f"{config.NTFY_SERVER.rstrip('/')}/{config.NTFY_TOPIC}", content=body.encode("utf-8"),
                       headers={"Title": ascii_only(title), "Priority": PRIORITY.get(kind, "default"),
                                "Tags": TAGS.get(kind, ""), "Markdown": "no"}, timeout=15)
        return r.status_code < 300, f"ntfy {r.status_code}"
    except Exception as e:  # noqa: BLE001 — delivery must never raise
        return False, f"ntfy error: {type(e).__name__}: {e}"


def push_macos(title: str, body: str, kind: str = "signal") -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "not macOS"
    safe = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')  # noqa: E731
    sound = ' sound name "Glass"' if kind in ("signal", "halt") else ""
    try:
        subprocess.run(["osascript", "-e", f'display notification "{safe(body[:200])}" with title "{safe(title)}"{sound}'],
                       capture_output=True, timeout=10, check=False)
        return True, "macos ok"
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"macos error: {e}"


CHANNELS = {"ntfy": push_ntfy, "macos": push_macos}


def push(title: str, body: str, kind: str = "signal", channels: list[str] | None = None) -> dict[str, str]:
    """Deliver on every configured channel; never raises. Returns {channel: status}."""
    results = {}
    for ch in channels or config.NOTIFY_CHANNELS:
        fn = CHANNELS.get(ch)
        if fn is None:
            results[ch] = "unknown channel"
            continue
        try:
            ok, msg = fn(title, body, kind)
        except Exception as e:  # noqa: BLE001 — a broken channel must not lose the alert
            ok, msg = False, f"{type(e).__name__}: {e}"
        results[ch] = ("ok: " if ok else "FAILED: ") + msg
    return results
