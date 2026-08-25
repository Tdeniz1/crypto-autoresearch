"""Rung 2 — Claude Code headless (`claude -p`), Read/Write tools only.

The subprocess runs in a throw-away sandbox directory (NOT the repo), with a
minimal environment that never contains the holdout env var. When
CLAUDE_CODE_OAUTH_TOKEN is set (from `claude setup-token`) we run with --bare so
the unattended overnight run does not depend on the keychain; otherwise we fall
back to the interactive login and log a warning.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from harness import config
from proposer.errors import RungError, RungQuotaError, RungUnavailable, looks_like_quota
from proposer.prompt import extract_strategy
from proposer.result import RungResult

NAME = "claude_code"
_warned_no_token = False


def available() -> str | None:
    if shutil.which(os.environ.get("CAR_CLAUDE_BIN", config.CLAUDE_CODE_BIN)) is None:
        return f"`{config.CLAUDE_CODE_BIN}` binary not on PATH"
    return None


def _env() -> dict:
    keep = {"PATH", "HOME", "TERM", "LANG", "LC_ALL", "TMPDIR", "USER", "SHELL"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    tok = os.environ.get(config.CLAUDE_CODE_TOKEN_ENV)
    if tok:
        env[config.CLAUDE_CODE_TOKEN_ENV] = tok
    env.pop(config.HOLDOUT_ENV_VAR, None)
    return env


def propose(static_context: str, dynamic_context: str, attempt: int = 0) -> RungResult:
    global _warned_no_token
    reason = available()
    if reason:
        raise RungUnavailable(reason)
    bare = bool(os.environ.get(config.CLAUDE_CODE_TOKEN_ENV))
    if not bare and not _warned_no_token:
        print(f"WARN rung 2: {config.CLAUDE_CODE_TOKEN_ENV} not set; using interactive login "
              f"(run `claude setup-token` for unattended runs)")
        _warned_no_token = True

    with tempfile.TemporaryDirectory(prefix="car-proposer-") as tmp:
        sandbox = Path(tmp)
        (sandbox / "strategy.py").write_text("")
        prompt = (
            static_context + "\n\n" + dynamic_context +
            "\n\nINSTRUCTIONS FOR THIS SESSION: You are in an empty sandbox directory. "
            "Use the Write tool exactly once to write the complete new strategy.py to ./strategy.py "
            "in this directory, then reply with the single word DONE. Do not read other files; "
            "everything you need is above."
        )
        cmd = [os.environ.get("CAR_CLAUDE_BIN", config.CLAUDE_CODE_BIN), "-p",
               "--output-format", "json",
               "--max-turns", str(config.CLAUDE_CODE_MAX_TURNS),
               "--allowedTools", "Read", "Write",
               "--model", config.CLAUDE_CODE_MODEL]
        if bare:
            cmd.insert(2, "--bare")
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=sandbox,
                                  env=_env(), timeout=config.CLAUDE_CODE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as e:
            raise RungError(f"claude -p timed out after {config.CLAUDE_CODE_TIMEOUT_SEC}s") from e
        except OSError as e:
            raise RungError(f"claude -p failed to launch: {e}") from e

        stdout = proc.stdout.strip()
        try:
            data = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            data = {}
        if not data:
            msg = (proc.stderr or stdout or "")[:300]
            if looks_like_quota(msg):
                raise RungQuotaError(f"claude -p quota: {msg}")
            raise RungError(f"claude -p exit {proc.returncode}, no JSON: {msg}")

        result_text = str(data.get("result") or "")
        if data.get("is_error") or data.get("subtype", "success") != "success" and not result_text:
            msg = result_text or json.dumps(data)[:300]
            if looks_like_quota(msg) or data.get("api_error_status") == 429:
                raise RungQuotaError(f"claude -p quota/rate-limit: {msg[:200]}")
            raise RungError(f"claude -p error: {msg[:200]}")

        written = (sandbox / "strategy.py").read_text() if (sandbox / "strategy.py").exists() else ""
        source = written if "def strategy" in written else extract_strategy(result_text)
        if not source:
            raise RungError("claude -p produced no strategy.py")

        usage = data.get("usage") or {}
        model_usage = data.get("modelUsage") or {}
        # modelUsage may list helper models (haiku for titles etc.); attribute to the one that did the work
        model = config.CLAUDE_CODE_MODEL
        if model_usage:
            model = max(model_usage, key=lambda k: ((model_usage[k] or {}).get("costUSD") or 0,
                                                    (model_usage[k] or {}).get("outputTokens") or 0))
        return RungResult(
            source=source, rung=NAME, model=model,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cost_usd=float(data.get("total_cost_usd") or 0.0),
            extra={"session_id": data.get("session_id"), "num_turns": data.get("num_turns"),
                   "duration_ms": data.get("duration_ms"), "bare": bare},
        )
