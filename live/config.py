"""Live alerting configuration. Independent of harness/ (deployable on its own).

Env overrides are read at import time so Railway variables work without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env is optional; Railway injects real env vars
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "live"
LOG_DIR = ROOT / "logs"
ALERTS_LOG = LOG_DIR / "alerts.jsonl"
OUTCOMES_LOG = LOG_DIR / "alert_outcomes.jsonl"
LIVE_BARS_DIR = LOG_DIR / "live_bars"
STATE_FILE = LOG_DIR / "alerter_state.json"
PROMOTED_FILE = LIVE_DIR / "promoted.json"
PROMOTED_DIR = LIVE_DIR / "promoted"
REPORT_DIR = ROOT / "reports"

EXCHANGE = os.environ.get("ALERTER_EXCHANGE", "binanceus")
LOOKBACK_BARS = int(os.environ.get("ALERTER_LOOKBACK_BARS", "1500"))   # history fed to the strategy each run

# ---- sizing -----------------------------------------------------------------
RISK_FRACTION = float(os.environ.get("ALERTER_RISK_FRACTION", "0.10"))    # equity fraction per |position| == 1
ACCOUNT_EQUITY_USD = float(os.environ.get("ALERTER_EQUITY_USD", "10000"))  # fallback when no broker balance
MIN_POSITION_DELTA = 0.05                                                  # ignore changes smaller than this
FEE_BPS = 10.0            # used by alerter_report for designed-vs-taken PnL
SLIPPAGE_BPS = 10.0

# ---- health -----------------------------------------------------------------
STALE_GRACE_MULT = 1.0     # halt if no new closed bar within timeframe * (1 + this) after the last one
ERROR_STREAK_HALT = 3      # consecutive fetch errors before halting
DD_HALT_MULT = 1.0         # halt when live drawdown > backtested max drawdown * this
REMIND_AFTER_BARS = 1      # nag about un-acked alerts older than this many bars

# ---- delivery ---------------------------------------------------------------
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NOTIFY_CHANNELS = [c for c in os.environ.get("ALERTER_CHANNELS", "ntfy,macos").split(",") if c]

# ---- modes ------------------------------------------------------------------
# Same code path; only the credential env names and the tag differ. Orders are never placed in either mode.
MODES = {
    "paper": {"key_env": "ALPACA_PAPER_API_KEY", "secret_env": "ALPACA_PAPER_API_SECRET",
              "base_url": "https://paper-api.alpaca.markets"},
    "live": {"key_env": "ALPACA_LIVE_API_KEY", "secret_env": "ALPACA_LIVE_API_SECRET",
             "base_url": "https://api.alpaca.markets"},
}
