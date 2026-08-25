"""The scalar the ratchet compares. Exactly one float comes out of score().

score = mean_over_series(net Sharpe) * drawdown_penalty(worst max-drawdown)
        or -inf if ANY series has fewer than config.MIN_TRADES trades.

Also here: deflated Sharpe (Bailey & Lopez de Prado 2014) against the number
of trials the ledger has recorded, for multiple-testing accounting.
"""
from __future__ import annotations

import math

from harness import config

NEG_INF = float("-inf")
_EULER_GAMMA = 0.5772156649015329


def drawdown_penalty(mdd: float) -> float:
    pen = 1.0 - min(max(mdd, 0.0), 1.0) / config.DD_FULL_PENALTY
    return max(config.DD_PENALTY_FLOOR, pen)


def score(per_series_stats: list[dict]) -> float:
    """One float. Sharpe is net of fees and slippage by construction (engine)."""
    if not per_series_stats:
        return NEG_INF
    if min(s["trades"] for s in per_series_stats) < config.MIN_TRADES:
        return NEG_INF
    sharpes = [float(s["sharpe"]) for s in per_series_stats]
    mean_sharpe = sum(sharpes) / len(sharpes)
    worst_dd = max(float(s["max_drawdown"]) for s in per_series_stats)
    pen = drawdown_penalty(worst_dd)
    if not math.isfinite(mean_sharpe):
        return NEG_INF
    # penalty always makes the number WORSE: shrink gains, amplify losses
    val = mean_sharpe * pen if mean_sharpe >= 0 else mean_sharpe / pen
    return float(val)


# ---------------------------------------------------------------------------
# Deflated Sharpe ratio
# ---------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the inverse normal CDF."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """E[max SR] over n_trials i.i.d. trials with the given SR variance (per-bar units)."""
    if n_trials < 2 or trial_sharpe_var <= 0:
        return 0.0
    sd = math.sqrt(trial_sharpe_var)
    return sd * ((1 - _EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
                 + _EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(sharpe_annual: float, n_bars: int, bars_per_year: float,
                    n_trials: int, trial_sharpes_annual: list[float],
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Returns raw and deflated annualized Sharpe plus the DSR probability.

    trial_sharpes_annual: annualized Sharpe of every trial so far (multiple-testing set).
    """
    if not math.isfinite(sharpe_annual) or n_bars < 3 or bars_per_year <= 0:
        return {"raw": sharpe_annual, "deflated": NEG_INF, "sr0": 0.0, "dsr_prob": 0.0, "n_trials": n_trials}
    k = math.sqrt(bars_per_year)
    sr = sharpe_annual / k
    finite = [t / k for t in trial_sharpes_annual if math.isfinite(t)]
    var = 0.0
    if len(finite) >= 2:
        m = sum(finite) / len(finite)
        var = sum((t - m) ** 2 for t in finite) / (len(finite) - 1)
    sr0 = expected_max_sharpe(max(n_trials, len(finite)), var)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    se = math.sqrt(max(denom, 1e-12) / (n_bars - 1))
    z = (sr - sr0) / se if se > 0 else 0.0
    return {
        "raw": float(sharpe_annual),
        "deflated": float((sr - sr0) * k),
        "sr0": float(sr0 * k),
        "dsr_prob": float(norm_cdf(z)),
        "n_trials": int(max(n_trials, len(finite))),
    }
