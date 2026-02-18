"""Phase 1a entry degradation test per Gate 1 item 7.

Applies randomized entry degradation to a fraction of IS trades and
evaluates whether performance survives within pre-registered tolerances.

Degradation modes:
  - delayed_entry: move entry forward by N bars, re-walk trade
  - adverse_fill: worsen entry price by a dollar amount, recompute R
  - combined: both delay and adverse fill

Phase 1a exploratory tolerances (may tighten at Gate 5):
  - total_r_pct <= 25%
  - win_rate_pp <= 5 pp
  - mar_pct <= 30%

See docs/notes/Task11c_assumptions.md for design rationale.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from ctl.simulator import TradeResult

logger = logging.getLogger(__name__)

# Phase 1a exploratory tolerances (Gate 1 item 7).
DEFAULT_TOTAL_R_TOLERANCE = 25.0       # max % degradation
DEFAULT_WIN_RATE_TOLERANCE = 5.0       # max pp degradation
DEFAULT_MAR_TOLERANCE = 30.0           # max % degradation

DegradationMode = Literal["delayed_entry", "adverse_fill", "combined"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Aggregate metrics for a set of trade results."""

    n_trades: int
    total_r: float
    avg_r: float
    win_rate: float
    max_dd_r: float
    mar_proxy: float


@dataclass
class DegradationDeltas:
    """Degradation deltas vs baseline."""

    total_r_pct: float     # (degraded - baseline) / |baseline| * 100
    win_rate_pp: float     # (degraded - baseline) * 100
    mar_pct: float         # (degraded_mar - baseline_mar) / |baseline_mar| * 100


@dataclass
class DegradationReport:
    """Complete entry degradation report."""

    mode: str
    degradation_pct: float
    seed: int
    n_total: int
    n_degraded: int
    n_excluded: int        # trades lost due to degradation (risk <= 0, etc.)
    baseline: AggregateMetrics
    degraded: AggregateMetrics
    deltas: DegradationDeltas
    total_r_pass: bool
    win_rate_pass: bool
    mar_pass: bool

    @property
    def all_passed(self) -> bool:
        return self.total_r_pass and self.win_rate_pass and self.mar_pass

    @property
    def verdict(self) -> str:
        return "PASS" if self.all_passed else "FAIL"

    def summary(self) -> str:
        lines = [
            "=== Entry Degradation Report ===",
            f"Mode: {self.mode}, degraded {self.n_degraded}/{self.n_total} "
            f"trades ({self.degradation_pct:.0%}), seed={self.seed}",
            f"Excluded (unviable after degradation): {self.n_excluded}",
            "",
            f"Baseline: total_r={self.baseline.total_r:.3f}, "
            f"win_rate={self.baseline.win_rate:.1%}, "
            f"MAR={self.baseline.mar_proxy:.3f}",
            f"Degraded: total_r={self.degraded.total_r:.3f}, "
            f"win_rate={self.degraded.win_rate:.1%}, "
            f"MAR={self.degraded.mar_proxy:.3f}",
            "",
            f"total_r delta: {self.deltas.total_r_pct:+.1f}% "
            f"(tolerance: {DEFAULT_TOTAL_R_TOLERANCE}%) "
            f"{'PASS' if self.total_r_pass else 'FAIL'}",
            f"win_rate delta: {self.deltas.win_rate_pp:+.1f}pp "
            f"(tolerance: {DEFAULT_WIN_RATE_TOLERANCE}pp) "
            f"{'PASS' if self.win_rate_pass else 'FAIL'}",
            f"MAR delta: {self.deltas.mar_pct:+.1f}% "
            f"(tolerance: {DEFAULT_MAR_TOLERANCE}%) "
            f"{'PASS' if self.mar_pass else 'FAIL'}",
            "",
            f"Overall verdict: {self.verdict}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_drawdown_r(r_values: List[float]) -> float:
    """Max drawdown from per-trade R-multiples."""
    if not r_values:
        return 0.0
    cumulative = np.cumsum(r_values)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown))


def _compute_aggregate(r_values: List[float]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of TheoreticalR values."""
    n = len(r_values)
    if n == 0:
        return AggregateMetrics(
            n_trades=0, total_r=0.0, avg_r=0.0,
            win_rate=0.0, max_dd_r=0.0, mar_proxy=0.0,
        )
    total_r = sum(r_values)
    avg_r = total_r / n
    win_rate = sum(1 for r in r_values if r > 0) / n
    max_dd = _max_drawdown_r(r_values)
    mar = total_r / max_dd if max_dd > 0 else float("inf")
    return AggregateMetrics(
        n_trades=n, total_r=total_r, avg_r=avg_r,
        win_rate=win_rate, max_dd_r=max_dd, mar_proxy=mar,
    )


def _compute_deltas(
    baseline: AggregateMetrics,
    degraded: AggregateMetrics,
) -> DegradationDeltas:
    """Compute degradation deltas (same convention as slippage stress)."""
    # Total R % change.
    if abs(baseline.total_r) > 1e-9:
        total_r_pct = (
            (degraded.total_r - baseline.total_r)
            / abs(baseline.total_r) * 100
        )
    else:
        total_r_pct = 0.0 if abs(degraded.total_r) < 1e-9 else float("-inf")

    # Win rate pp.
    win_rate_pp = (degraded.win_rate - baseline.win_rate) * 100

    # MAR % change.
    b_mar = baseline.mar_proxy
    d_mar = degraded.mar_proxy
    if b_mar == float("inf") and d_mar == float("inf"):
        mar_pct = 0.0
    elif b_mar == float("inf") or abs(b_mar) < 1e-9:
        mar_pct = 0.0 if d_mar == float("inf") else float("-inf")
    else:
        mar_pct = (d_mar - b_mar) / abs(b_mar) * 100

    return DegradationDeltas(
        total_r_pct=total_r_pct,
        win_rate_pp=win_rate_pp,
        mar_pct=mar_pct,
    )


def _select_degraded_indices(
    n: int,
    degradation_pct: float,
    seed: int,
) -> np.ndarray:
    """Deterministically select which trades to degrade."""
    k = int(n * degradation_pct)
    k = min(k, n)
    if k == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)


# ---------------------------------------------------------------------------
# Degradation: adverse fill (price worsening only)
# ---------------------------------------------------------------------------

def degrade_adverse_fill(
    trade: TradeResult,
    adverse_amount: float,
) -> Optional[TradeResult]:
    """Worsen entry price; recompute R-multiples.

    Parameters
    ----------
    trade : TradeResult
    adverse_amount : float
        Dollar amount to add to entry price (worsens long entry).

    Returns
    -------
    Degraded TradeResult, or None if risk becomes <= 0.
    """
    new_entry = trade.entry_price + adverse_amount
    new_risk = new_entry - trade.stop_price
    if new_risk <= 0:
        return None

    # Recompute exit price for stop exits (stop fill worsens too).
    if trade.exit_reason == "Stop":
        new_exit = trade.exit_price  # stop fill is independent of entry
    else:
        new_exit = trade.exit_price  # TP/Open: limit order, unchanged

    # Recompute R-multiples.
    actual_r = (new_exit - new_entry) / new_risk

    # Theoretical R (perfect thirds).
    tp_levels = [trade.tp1, trade.tp2, trade.tp3]
    tp_hit = [trade.tp1_hit, trade.tp2_hit, trade.tp3_hit]
    parts = []
    for i in range(3):
        if tp_hit[i]:
            parts.append((tp_levels[i] - new_entry) / new_risk)
        else:
            parts.append((new_exit - new_entry) / new_risk)
    theoretical_r = sum(p / 3.0 for p in parts)

    result = copy.copy(trade)
    result.entry_price = new_entry
    result.risk_per_unit = new_risk
    result.r_multiple_actual = actual_r
    result.theoretical_r = theoretical_r
    result.mfe_r = (trade.entry_price + trade.mfe_r * (trade.entry_price - trade.stop_price) - new_entry) / new_risk
    result.mae_r = (new_entry - (trade.entry_price - trade.mae_r * (trade.entry_price - trade.stop_price))) / new_risk
    result.trade_outcome = "Win" if tp_hit[0] else "Loss"
    if trade.exit_reason == "Open":
        result.trade_outcome = "Open"

    return result


# ---------------------------------------------------------------------------
# Degradation: delayed entry (re-walk from later bar)
# ---------------------------------------------------------------------------

def degrade_delayed_entry(
    trade: TradeResult,
    data_arrays: Dict[str, np.ndarray],
    delay_bars: int = 1,
    slippage: float = 0.0,
) -> Optional[TradeResult]:
    """Re-simulate trade from a later entry bar.

    Parameters
    ----------
    trade : TradeResult
    data_arrays : dict with keys "opens", "highs", "lows", "closes", "dates", "n_bars"
    delay_bars : int
    slippage : float
        Original slippage per side.

    Returns
    -------
    Degraded TradeResult, or None if delayed bar is beyond data or risk <= 0.
    """
    new_entry_idx = trade.entry_bar_idx + delay_bars
    n_bars = data_arrays["n_bars"]

    if new_entry_idx >= n_bars:
        return None

    opens = data_arrays["opens"]
    highs = data_arrays["highs"]
    lows = data_arrays["lows"]
    closes = data_arrays["closes"]
    dates = data_arrays["dates"]

    new_entry_price = float(opens[new_entry_idx]) + slippage
    new_risk = new_entry_price - trade.stop_price
    if new_risk <= 0:
        return None

    tp_levels = [trade.tp1, trade.tp2, trade.tp3]

    # Re-walk from new entry bar.
    highest_high = float(highs[new_entry_idx])
    lowest_low = float(lows[new_entry_idx])
    day1_fail = bool(lows[new_entry_idx] < trade.stop_price)
    tp_hit = [False, False, False]

    exit_idx = None
    exit_price_final = 0.0
    exit_reason = ""
    same_bar_collision = False
    exit_on_last_bar = False

    for bar in range(new_entry_idx, n_bars):
        h = float(highs[bar])
        lo = float(lows[bar])
        c = float(closes[bar])

        if h > highest_high:
            highest_high = h
        if lo < lowest_low:
            lowest_low = lo

        if bar == new_entry_idx:
            continue

        # TP evaluation.
        tp_hit_this_bar = -1
        for tp_idx in (2, 1, 0):
            if not tp_hit[tp_idx] and h >= tp_levels[tp_idx]:
                tp_hit_this_bar = tp_idx
                break

        # Stop evaluation.
        stop_breached = c < trade.stop_price

        # Collision (tp_wins default).
        if tp_hit_this_bar >= 0 and stop_breached:
            same_bar_collision = True
            stop_breached = False  # tp_wins

        if tp_hit_this_bar >= 0:
            for j in range(tp_hit_this_bar + 1):
                tp_hit[j] = True
            if all(tp_hit):
                exit_idx = bar
                exit_price_final = tp_levels[2]
                exit_reason = "TP3"
                break

        if stop_breached:
            next_bar = bar + 1
            if next_bar < n_bars:
                exit_idx = next_bar
                exit_price_final = float(opens[next_bar]) - slippage
            else:
                exit_idx = bar
                exit_price_final = c
                exit_on_last_bar = True
            exit_reason = "Stop"
            break

    if exit_idx is None:
        exit_idx = n_bars - 1
        exit_price_final = float(closes[n_bars - 1])
        exit_reason = "Open"
        exit_on_last_bar = True

    actual_r = (exit_price_final - new_entry_price) / new_risk

    # Theoretical R (perfect thirds).
    parts = []
    for i in range(3):
        if tp_hit[i]:
            parts.append((tp_levels[i] - new_entry_price) / new_risk)
        else:
            parts.append((exit_price_final - new_entry_price) / new_risk)
    theoretical_r = sum(p / 3.0 for p in parts)

    mfe_r = (highest_high - new_entry_price) / new_risk
    mae_r = (new_entry_price - lowest_low) / new_risk

    trade_outcome = "Win" if tp_hit[0] else "Loss"
    if exit_reason == "Open":
        trade_outcome = "Open"

    result = copy.copy(trade)
    result.entry_bar_idx = new_entry_idx
    result.entry_date = dates[new_entry_idx] if dates is not None else None
    result.entry_price = new_entry_price
    result.risk_per_unit = new_risk
    result.exit_bar_idx = exit_idx
    result.exit_date = dates[exit_idx] if dates is not None else None
    result.exit_price = exit_price_final
    result.exit_reason = exit_reason
    result.r_multiple_actual = actual_r
    result.theoretical_r = theoretical_r
    result.mfe_r = mfe_r
    result.mae_r = mae_r
    result.day1_fail = day1_fail
    result.same_bar_collision = same_bar_collision
    result.exit_on_last_bar = exit_on_last_bar
    result.trade_outcome = trade_outcome
    result.hold_bars = exit_idx - new_entry_idx
    result.tp1_hit = tp_hit[0]
    result.tp2_hit = tp_hit[1]
    result.tp3_hit = tp_hit[2]

    return result


# ---------------------------------------------------------------------------
# Main entry degradation runner
# ---------------------------------------------------------------------------

def run_entry_degradation(
    results: List[TradeResult],
    data_by_symbol: Optional[Dict[str, "pd.DataFrame"]] = None,
    mode: DegradationMode = "adverse_fill",
    degradation_pct: float = 0.30,
    seed: int = 42,
    delay_bars: int = 1,
    adverse_fill_fraction: float = 0.1,
    slippage: float = 0.0,
    total_r_tolerance: float = DEFAULT_TOTAL_R_TOLERANCE,
    win_rate_tolerance: float = DEFAULT_WIN_RATE_TOLERANCE,
    mar_tolerance: float = DEFAULT_MAR_TOLERANCE,
) -> DegradationReport:
    """Run entry degradation test on IS trade results.

    Parameters
    ----------
    results : list of TradeResult
        Baseline IS trade results.
    data_by_symbol : dict of symbol -> DataFrame, optional
        Required for delayed_entry and combined modes (need OHLCV to re-walk).
    mode : "adverse_fill", "delayed_entry", or "combined"
    degradation_pct : float
        Fraction of trades to degrade (default 0.30).
    seed : int
        Random seed for reproducibility.
    delay_bars : int
        Bars to delay entry (default 1).
    adverse_fill_fraction : float
        Adverse fill as fraction of risk_per_unit (default 0.10).
    slippage : float
        Original slippage per side from baseline simulation.
    total_r_tolerance : float
        Max % degradation in total R (default 25%).
    win_rate_tolerance : float
        Max pp degradation in win rate (default 5).
    mar_tolerance : float
        Max % degradation in MAR (default 30%).

    Returns
    -------
    DegradationReport with baseline, degraded metrics, deltas, and pass/fail.
    """
    import pandas as pd  # deferred to avoid circular imports at module level

    n = len(results)
    baseline_r = [t.theoretical_r for t in results]
    baseline_metrics = _compute_aggregate(baseline_r)

    if n == 0:
        empty = _compute_aggregate([])
        deltas = DegradationDeltas(0.0, 0.0, 0.0)
        return DegradationReport(
            mode=mode, degradation_pct=degradation_pct, seed=seed,
            n_total=0, n_degraded=0, n_excluded=0,
            baseline=empty, degraded=empty, deltas=deltas,
            total_r_pass=True, win_rate_pass=True, mar_pass=True,
        )

    # Select trades to degrade.
    degraded_indices = set(_select_degraded_indices(n, degradation_pct, seed))

    # Prepare data arrays for delayed entry.
    data_arrays_cache: Dict[str, Dict] = {}
    if mode in ("delayed_entry", "combined") and data_by_symbol is not None:
        for sym, df in data_by_symbol.items():
            data_arrays_cache[sym] = {
                "opens": df["Open"].values.astype(float),
                "highs": df["High"].values.astype(float),
                "lows": df["Low"].values.astype(float),
                "closes": df["Close"].values.astype(float),
                "dates": df["Date"].values if "Date" in df.columns else None,
                "n_bars": len(df),
            }

    degraded_r: List[float] = []
    n_excluded = 0
    n_actually_degraded = 0

    for i, trade in enumerate(results):
        if i not in degraded_indices:
            degraded_r.append(trade.theoretical_r)
            continue

        n_actually_degraded += 1
        degraded_trade: Optional[TradeResult] = None

        if mode == "adverse_fill":
            amount = trade.risk_per_unit * adverse_fill_fraction
            degraded_trade = degrade_adverse_fill(trade, amount)

        elif mode == "delayed_entry":
            arrays = data_arrays_cache.get(trade.symbol)
            if arrays is None:
                # No data for re-walk — keep original.
                degraded_r.append(trade.theoretical_r)
                continue
            degraded_trade = degrade_delayed_entry(
                trade, arrays, delay_bars, slippage,
            )

        elif mode == "combined":
            # First delay, then adverse fill on the delayed result.
            arrays = data_arrays_cache.get(trade.symbol)
            if arrays is None:
                degraded_r.append(trade.theoretical_r)
                continue
            delayed = degrade_delayed_entry(
                trade, arrays, delay_bars, slippage,
            )
            if delayed is not None:
                amount = delayed.risk_per_unit * adverse_fill_fraction
                degraded_trade = degrade_adverse_fill(delayed, amount)
            else:
                degraded_trade = None

        if degraded_trade is None:
            n_excluded += 1
        else:
            degraded_r.append(degraded_trade.theoretical_r)

    degraded_metrics = _compute_aggregate(degraded_r)
    deltas = _compute_deltas(baseline_metrics, degraded_metrics)

    # Pass/fail against tolerances (degradation is negative, check magnitude).
    total_r_pass = abs(min(deltas.total_r_pct, 0.0)) <= total_r_tolerance
    win_rate_pass = abs(min(deltas.win_rate_pp, 0.0)) <= win_rate_tolerance
    mar_pass = abs(min(deltas.mar_pct, 0.0)) <= mar_tolerance

    report = DegradationReport(
        mode=mode,
        degradation_pct=degradation_pct,
        seed=seed,
        n_total=n,
        n_degraded=n_actually_degraded,
        n_excluded=n_excluded,
        baseline=baseline_metrics,
        degraded=degraded_metrics,
        deltas=deltas,
        total_r_pass=total_r_pass,
        win_rate_pass=win_rate_pass,
        mar_pass=mar_pass,
    )

    logger.info("Entry degradation (%s): %s", mode, report.verdict)
    return report
