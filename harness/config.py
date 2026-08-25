"""Frozen harness configuration.

Everything the research loop is NOT allowed to touch lives here. The loop
may only edit strategy.py. Edit this file by hand, between sessions.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FROZEN_DIR = DATA_DIR / "frozen"
LOG_DIR = ROOT / "logs"
CURVE_DIR = LOG_DIR / "curves"
REPORT_DIR = ROOT / "reports"
SESSION_LOG = LOG_DIR / "session.jsonl"
LEDGER_LOG = LOG_DIR / "ledger.jsonl"
STRATEGY_FILE = ROOT / "strategy.py"
PROGRAM_FILE = ROOT / "program.md"

# ---------------------------------------------------------------- data ------
# The holdout slice lives OUTSIDE the project tree. prepare.py writes it to the
# directory named by this env var. The runner asserts the var is UNSET.
HOLDOUT_ENV_VAR = "CRYPTO_HOLDOUT_DIR"
EXCHANGES = ["binanceus", "coinbase", "kraken"]  # tried in order by prepare.py
SYMBOLS = ["BTC/USDT", "ETH/USDT"]              # only prepare.py / holdout know this
TIMEFRAMES = ["1h", "15m"]
SPLIT = (0.70, 0.15, 0.15)                       # train / validation / holdout, chronological
BARS_PER_YEAR = {"1h": 24 * 365, "15m": 96 * 365, "4h": 6 * 365, "1d": 365}
# Which frozen series ids the ratchet evaluates. None = every series in the manifest.
EVAL_SERIES: list[str] | None = None

# --------------------------------------------------------- execution costs ---
# Conservative: Binance-tier taker fee plus fat slippage, both per side, in bps.
FEE_BPS = 10.0
SLIPPAGE_BPS = 10.0

# ---------------------------------------------------------------- metric ----
MIN_TRADES = 30            # per evaluated series; below this the metric is -inf
DD_FULL_PENALTY = 0.50     # max drawdown at which the penalty hits its floor
DD_PENALTY_FLOOR = 0.05    # penalty multiplier never goes below this
MAX_ABS_POSITION = 1.0     # positions are clipped to [-1, 1]

# ----------------------------------------------------------- determinism ----
SEED = 1337

# --------------------------------------------------------------- canaries ---
CANARY_SHUFFLES = 3               # permutations averaged for the shuffled-returns canary
LEAK_SHUFFLE_ABS_SHARPE = 1.0     # shuffled Sharpe above this AND ...
LEAK_SHUFFLE_FRAC = 0.5           # ... above this fraction of the real Sharpe => LEAK_SUSPECTED
TIMING_RETAIN_FRAC = 0.25         # delayed Sharpe below this fraction of real => TIMING_FRAGILE
COST_INVERTED_SHARPE = 0.5        # inverted Sharpe above this while real > 0 => COST_MODEL_BROKEN
COST_IDENTITY_TOL = 1e-9          # |net_orig + net_inv + 2*cost| tolerance
REJECT_ON_LEAK = True             # LEAK_SUSPECTED never becomes the new best

# ------------------------------------------------------------------ scan ----
SCAN_MAX_LINES = 400
SCAN_ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
SCAN_YEAR_RANGE = (2009, 2035)

# ---------------------------------------------------------------- proposer --
PROPOSER_RUNGS = ["openrouter", "claude_code", "anthropic"]
RETRIES_PER_RUNG = 2
BACKOFF_BASE_SEC = float(os.environ.get("CAR_BACKOFF_BASE_SEC", "3.0"))
RUNG_COOLDOWN_SEC = 900           # a demoted rung is retried after this many seconds
MAX_CONSECUTIVE_PROPOSER_FAILURES = 5
PROPOSER_MAX_OUTPUT_TOKENS = 6000
LOG_SUMMARY_ROWS = 15

# Rung 1 — OpenRouter free models, rotated on retry.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS = [
    "qwen/qwen3-coder:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/devstral-small-2505:free",
]
OPENROUTER_TIMEOUT_SEC = 180

# Rung 2 — Claude Code headless.
CLAUDE_CODE_BIN = "claude"
CLAUDE_CODE_MODEL = "sonnet"      # alias accepted by the CLI; keep the overnight run cheap
CLAUDE_CODE_MAX_TURNS = 4
CLAUDE_CODE_TIMEOUT_SEC = 600
CLAUDE_CODE_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"  # from `claude setup-token`

# Rung 3 — Anthropic API direct.
ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_MODEL_OPTIONS = ["claude-haiku-4-5", "claude-sonnet-5"]
RUNG3_SESSION_USD_CAP = 1.00      # hard cap; the session stops cleanly when reached
# USD per 1M tokens: input, output, cache write, cache read
PRICES_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
}

# ------------------------------------------------------------------ report --
REPORT_MODEL = "claude-opus-5"    # separately configurable; only report.py uses it
REPORT_MAX_TOKENS = 4000

# --------------------------------------------------------------- dashboard --
DASHBOARD_PORT = 8765
DASHBOARD_TOP_N = 15
DASHBOARD_PLATEAU_EXPERIMENTS = 25   # best-so-far flat for this many => "search exhausted"
FLAG_TRADE_FLOOR_PROXIMITY = 0.20    # within 20% of MIN_TRADES
FLAG_TIME_IN_MARKET_MIN = 0.05
FLAG_TURNOVER_MAX = 400.0            # annualized position turnover
FLAG_MDD_MAX = 0.35
FLAG_IMPROVEMENT_STD_WINDOW = 20


def eval_series_ids(manifest: dict) -> list[str]:
    ids = [s["id"] for s in manifest["series"]]
    if EVAL_SERIES is None:
        return ids
    missing = [s for s in EVAL_SERIES if s not in ids]
    if missing:
        raise ValueError(f"EVAL_SERIES not in manifest: {missing}")
    return list(EVAL_SERIES)
