"""Slippage stress testing framework per Phase Gate Checklist v2.

Applies slippage scenarios (0/1/2/3 ticks per side) to baseline trade
results and evaluates whether the edge survives execution costs.

Gate 1 item 8: profitable at 2 ticks per side.
Kill criterion: edge evaporates at 2 ticks → REJECT.
Tracker v3 Task 4b: record total R, win rate, avg R, MAR at each level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ctl.simulator import TradeResult
from ctl.universe import EQUITY_TICK, TICK_VALUES


# ---------------------------------------------------------------------------
# Per-scenario metrics
# ---------------------------------------------------------------------------

@dataclass
class ScenarioMetrics:
    """Aggregate metrics for one slippage scenario."""

    ticks: int
    n_trades: int
    n_excluded: int             # trades where risk <= 0 after slippage
    total_r: float              # sum of adjusted TheoreticalR
    avg_r: float                # mean adjusted TheoreticalR
    win_rate: float             # fraction with adjusted TheoreticalR > 0
    total_actual_r: float       # sum of adjusted actual R
    max_drawdown_r: float       # max DD in cumulative TheoreticalR
    mar_proxy: float            # total_r / max_drawdown_r


# ---------------------------------------------------------------------------
# Degradation vs baseline
# ---------------------------------------------------------------------------

@dataclass
class DegradationResult:
    """Degradation of one scenario vs baseline."""

    ticks: int
    total_r_pct: float          # % change in total R
    win_rate_pp: float          # change in win rate (percentage points)
    mar_pct: float              # % change in MAR proxy


# ---------------------------------------------------------------------------
# Stress report
# ---------------------------------------------------------------------------

@dataclass
class StressReport:
    """Full slippage stress report for Gate 1 review."""

    scenarios: List[ScenarioMetrics]
    degradations: List[DegradationResult]
    profitable_at_2_ticks: bool
    profitable_at_3_ticks: Optional[bool]   # None if 3 ticks not tested
    gate_pass: bool                          # Gate 1 item 8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tick_value(
    symbol: str,
    tick_values: Dict[str, float],
    equity_tick: float,
) -> float:
    """Look up tick value for a symbol."""
    return tick_values.get(symbol, equity_tick)


def _max_drawdown_r(r_values: List[float]) -> float:
    """Max drawdown from a sequence of per-trade R-multiples."""
    if not r_values:
        return 0.0
    cumulative = np.cumsum(r_values)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown))


def _adjust_trade(
    trade: TradeResult,
    slippage_per_side: float,
    baseline_slippage: float,
) -> Optional[Dict]:
    """Adjust a single trade for a new slippage level.

    Returns a dict with adjusted R-multiples, or None if the trade
    becomes unviable (risk <= 0).
    """
    delta = slippage_per_side - baseline_slippage

    new_entry = trade.entry_price + delta
    new_risk = new_entry - trade.stop_price

    if new_risk <= 0:
        return None

    # Adjust exit price based on exit type.
    if trade.exit_reason == "Stop":
        new_exit = trade.exit_price - delta
    else:
        # TP or Open: limit-order fill or snapshot — no slippage adjustment.
        new_exit = trade.exit_price

    # Actual R.
    actual_r = (new_exit - new_entry) / new_risk

    # Theoretical R (perfect thirds per Spec §7).
    tp_levels = [trade.tp1, trade.tp2, trade.tp3]
    tp_hit = [trade.tp1_hit, trade.tp2_hit, trade.tp3_hit]

    parts: List[float] = []
    for i in range(3):
        if tp_hit[i]:
            parts.append((tp_levels[i] - new_entry) / new_risk)
        else:
            parts.append((new_exit - new_entry) / new_risk)

    theoretical_r = sum(p / 3.0 for p in parts)

    return {
        "actual_r": actual_r,
        "theoretical_r": theoretical_r,
    }


# ---------------------------------------------------------------------------
# Scenario computation
# ---------------------------------------------------------------------------

def compute_scenario_metrics(
    results: List[TradeResult],
    n_ticks: int,
    tick_values: Dict[str, float] | None = None,
    equity_tick: float = EQUITY_TICK,
    baseline_slippage: float = 0.0,
) -> ScenarioMetrics:
    """Compute aggregate metrics for one slippage scenario.

    Parameters
    ----------
    results : list of TradeResult
        Baseline trade results.
    n_ticks : int
        Number of ticks of slippage per side for this scenario.
    tick_values : dict, optional
        Symbol -> tick value in dollars.  Defaults to universe TICK_VALUES.
    equity_tick : float
        Default tick value for equities/ETFs (default $0.01).
    baseline_slippage : float
        Slippage already baked into the baseline results (default 0).
    """
    if tick_values is None:
        tick_values = TICK_VALUES

    adjusted_theoretical: List[float] = []
    adjusted_actual: List[float] = []
    n_excluded = 0

    for trade in results:
        tv = _get_tick_value(trade.symbol, tick_values, equity_tick)
        slippage = tv * n_ticks
        adj = _adjust_trade(trade, slippage, baseline_slippage)

        if adj is None:
            n_excluded += 1
            continue

        adjusted_theoretical.append(adj["theoretical_r"])
        adjusted_actual.append(adj["actual_r"])

    n_trades = len(adjusted_theoretical)

    if n_trades == 0:
        return ScenarioMetrics(
            ticks=n_ticks, n_trades=0, n_excluded=n_excluded,
            total_r=0.0, avg_r=0.0, win_rate=0.0,
            total_actual_r=0.0, max_drawdown_r=0.0, mar_proxy=0.0,
        )

    total_r = float(np.sum(adjusted_theoretical))
    avg_r = float(np.mean(adjusted_theoretical))
    win_rate = float(np.mean([r > 0 for r in adjusted_theoretical]))
    total_actual_r = float(np.sum(adjusted_actual))
    max_dd = _max_drawdown_r(adjusted_theoretical)
    mar = total_r / max_dd if max_dd > 0 else float("inf")

    return ScenarioMetrics(
        ticks=n_ticks,
        n_trades=n_trades,
        n_excluded=n_excluded,
        total_r=total_r,
        avg_r=avg_r,
        win_rate=win_rate,
        total_actual_r=total_actual_r,
        max_drawdown_r=max_dd,
        mar_proxy=mar,
    )


# ---------------------------------------------------------------------------
# Degradation computation
# ---------------------------------------------------------------------------

def compute_degradation(
    baseline: ScenarioMetrics,
    stressed: ScenarioMetrics,
) -> DegradationResult:
    """Compute degradation of a stressed scenario vs baseline.

    Returns
    -------
    DegradationResult with:
      - total_r_pct: (stressed - baseline) / |baseline| * 100
      - win_rate_pp: (stressed - baseline) in percentage points
      - mar_pct: (stressed - baseline) / |baseline| * 100
    """
    # Total R % change.
    if abs(baseline.total_r) > 1e-9:
        total_r_pct = (
            (stressed.total_r - baseline.total_r)
            / abs(baseline.total_r)
            * 100
        )
    else:
        total_r_pct = 0.0 if abs(stressed.total_r) < 1e-9 else float("-inf")

    # Win rate change in percentage points.
    win_rate_pp = (stressed.win_rate - baseline.win_rate) * 100

    # MAR % change.
    if (
        baseline.mar_proxy != float("inf")
        and abs(baseline.mar_proxy) > 1e-9
    ):
        mar_pct = (
            (stressed.mar_proxy - baseline.mar_proxy)
            / abs(baseline.mar_proxy)
            * 100
        )
    elif stressed.mar_proxy == float("inf") and baseline.mar_proxy == float("inf"):
        mar_pct = 0.0
    else:
        mar_pct = float("-inf")

    return DegradationResult(
        ticks=stressed.ticks,
        total_r_pct=total_r_pct,
        win_rate_pp=win_rate_pp,
        mar_pct=mar_pct,
    )


# ---------------------------------------------------------------------------
# Full stress test
# ---------------------------------------------------------------------------

def run_stress_test(
    results: List[TradeResult],
    tick_values: Dict[str, float] | None = None,
    tick_levels: List[int] | None = None,
    equity_tick: float = EQUITY_TICK,
    baseline_slippage: float = 0.0,
) -> StressReport:
    """Run slippage stress test across multiple tick levels.

    Parameters
    ----------
    results : list of TradeResult
        Baseline trade results (assumed at baseline_slippage level).
    tick_values : dict, optional
        Symbol -> tick value.  Defaults to universe TICK_VALUES.
    tick_levels : list of int, optional
        Tick levels to test (default [0, 1, 2, 3] per Tracker v3).
    equity_tick : float
        Default tick for equities/ETFs.
    baseline_slippage : float
        Slippage already baked into baseline results (default 0).

    Returns
    -------
    StressReport
    """
    if tick_levels is None:
        tick_levels = [0, 1, 2, 3]

    scenarios: List[ScenarioMetrics] = []
    for ticks in tick_levels:
        sm = compute_scenario_metrics(
            results, ticks, tick_values, equity_tick, baseline_slippage,
        )
        scenarios.append(sm)

    # Baseline is the first scenario (should be 0 ticks).
    baseline = scenarios[0]

    degradations: List[DegradationResult] = []
    for sm in scenarios[1:]:
        degradations.append(compute_degradation(baseline, sm))

    # Gate check.
    profitable_2 = False
    profitable_3: Optional[bool] = None

    for sm in scenarios:
        if sm.ticks == 2:
            profitable_2 = sm.total_r > 0
        if sm.ticks == 3:
            profitable_3 = sm.total_r > 0

    return StressReport(
        scenarios=scenarios,
        degradations=degradations,
        profitable_at_2_ticks=profitable_2,
        profitable_at_3_ticks=profitable_3,
        gate_pass=profitable_2,
    )
