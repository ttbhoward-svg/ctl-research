"""Tests for slippage stress testing framework."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.simulator import TradeResult
from ctl.slippage_stress import (
    DegradationResult,
    ScenarioMetrics,
    StressReport,
    _adjust_trade,
    _max_drawdown_r,
    compute_degradation,
    compute_scenario_metrics,
    run_stress_test,
)


# ---------------------------------------------------------------------------
# Helpers — create trades with controlled fields
# ---------------------------------------------------------------------------

def _trade(
    symbol: str = "/PA",
    entry_price: float = 105.0,
    stop_price: float = 100.0,
    exit_price: float = 115.0,
    exit_reason: str = "TP3",
    tp1: float = 110.0,
    tp2: float = 112.0,
    tp3: float = 115.0,
    tp1_hit: bool = True,
    tp2_hit: bool = True,
    tp3_hit: bool = True,
) -> TradeResult:
    """Create a TradeResult with the fields relevant to slippage stress."""
    risk = entry_price - stop_price
    return TradeResult(
        symbol=symbol,
        timeframe="daily",
        entry_price=entry_price,
        stop_price=stop_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        risk_per_unit=risk,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp3_hit=tp3_hit,
    )


# =========================================================================
# _adjust_trade: per-trade slippage adjustment
# =========================================================================

class TestAdjustTrade:
    def test_zero_delta_unchanged(self):
        """No slippage delta -> R values unchanged."""
        t = _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0)
        adj = _adjust_trade(t, slippage_per_side=0.0, baseline_slippage=0.0)

        assert adj is not None
        risk = 5.0
        expected_theo = ((110 - 105) / risk + (112 - 105) / risk
                         + (115 - 105) / risk) / 3.0
        assert abs(adj["theoretical_r"] - expected_theo) < 1e-9

    def test_tp_exit_unaffected_by_slippage(self):
        """TP exits stay at TP level — only entry changes."""
        t = _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0,
                    exit_reason="TP3")
        adj = _adjust_trade(t, slippage_per_side=5.0, baseline_slippage=0.0)

        assert adj is not None
        # New entry = 110, new risk = 10.  Exit stays at 115 (TP3).
        expected_actual = (115.0 - 110.0) / 10.0  # 0.5
        assert abs(adj["actual_r"] - expected_actual) < 1e-9

    def test_stop_exit_worsened(self):
        """Stop exit price decreases with slippage delta."""
        t = _trade(entry_price=5000.0, stop_price=4990.0,
                    exit_price=4988.0, exit_reason="Stop",
                    tp1=5010.0, tp2=5015.0, tp3=5020.0,
                    tp1_hit=False, tp2_hit=False, tp3_hit=False,
                    symbol="/ES")
        adj = _adjust_trade(t, slippage_per_side=12.50, baseline_slippage=0.0)

        assert adj is not None
        # new_entry=5012.50, new_risk=22.50, new_exit=4975.50
        expected = (4975.50 - 5012.50) / 22.50
        assert abs(adj["actual_r"] - expected) < 1e-6

    def test_risk_increases_with_slippage(self):
        """Higher slippage -> wider risk -> different R."""
        t = _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0)

        adj_0 = _adjust_trade(t, 0.0, 0.0)
        adj_5 = _adjust_trade(t, 5.0, 0.0)

        # With larger risk, same TP levels produce smaller R.
        assert adj_0["theoretical_r"] > adj_5["theoretical_r"]

    def test_unviable_trade_returns_none(self):
        """Trade excluded when risk <= 0 (entry at or below stop)."""
        # For longs, slippage moves entry HIGHER (away from stop), so it
        # widens risk — slippage alone cannot make a long trade unviable.
        # Unviable trades arise from data edge cases: entry == stop or
        # entry < stop (bad data).

        # Case 1: entry == stop -> risk = 0 -> None.
        t_zero_risk = _trade(entry_price=100.0, stop_price=100.0)
        assert _adjust_trade(t_zero_risk, 0.0, 0.0) is None

        # Case 2: entry below stop (bad data) -> risk < 0 -> None.
        t_negative_risk = _trade(entry_price=99.0, stop_price=100.0)
        assert _adjust_trade(t_negative_risk, 0.0, 0.0) is None

        # Case 3: valid trade remains viable even under heavy slippage.
        t_valid = _trade(entry_price=100.5, stop_price=100.0)
        # slippage=1.0 -> new_entry=101.5, risk=1.5 > 0 -> still viable.
        assert _adjust_trade(t_valid, 1.0, 0.0) is not None
        # Even extreme slippage: new_entry=200.5, risk=100.5 > 0.
        assert _adjust_trade(t_valid, 100.0, 0.0) is not None

    def test_open_exit_unaffected(self):
        """'Open' exit (trade still running) not adjusted for slippage."""
        t = _trade(entry_price=105.0, stop_price=100.0, exit_price=108.0,
                    exit_reason="Open",
                    tp1_hit=False, tp2_hit=False, tp3_hit=False)
        adj = _adjust_trade(t, slippage_per_side=5.0, baseline_slippage=0.0)

        assert adj is not None
        # new_entry=110, exit stays 108, new_risk=10
        expected = (108.0 - 110.0) / 10.0  # -0.2
        assert abs(adj["actual_r"] - expected) < 1e-9

    def test_tp1_hit_then_stopped(self):
        """TP1 hit + stop exit: TP1 third uses TP level, other thirds use
        adjusted stop exit."""
        t = _trade(entry_price=105.0, stop_price=100.0,
                    exit_price=98.0, exit_reason="Stop",
                    tp1=110.0, tp2=112.0, tp3=115.0,
                    tp1_hit=True, tp2_hit=False, tp3_hit=False)
        adj = _adjust_trade(t, slippage_per_side=5.0, baseline_slippage=0.0)

        assert adj is not None
        # new_entry=110, new_risk=10, new_exit=98-5=93 (stop adjusted)
        # theoretical_r = ( (110-110)/10 + (93-110)/10 + (93-110)/10 ) / 3
        #               = ( 0 + -1.7 + -1.7 ) / 3 = -1.133
        expected = (0.0 + (-17.0 / 10.0) + (-17.0 / 10.0)) / 3.0
        assert abs(adj["theoretical_r"] - expected) < 1e-6


# =========================================================================
# _max_drawdown_r
# =========================================================================

class TestMaxDrawdown:
    def test_no_drawdown(self):
        """All positive R -> no drawdown."""
        assert _max_drawdown_r([1.0, 2.0, 0.5]) == 0.0

    def test_simple_drawdown(self):
        """Single dip produces measurable drawdown."""
        # Cumulative: [1, -1, 0, -0.5], peak: [1, 1, 1, 1]
        # Drawdown: [0, 2, 1, 1.5] -> max = 2.0
        assert abs(_max_drawdown_r([1.0, -2.0, 1.0, -0.5]) - 2.0) < 1e-9

    def test_empty_list(self):
        assert _max_drawdown_r([]) == 0.0

    def test_all_losses(self):
        """Steady losses -> drawdown equals total loss magnitude."""
        # Cumulative: [-1, -2, -3], peak: [0, 0, 0] (starts at peak 0?
        # No — np.maximum.accumulate([-1, -2, -3]) = [-1, -1, -1]
        # Drawdown: [-1 - (-1), -1 - (-2), -1 - (-3)] = [0, 1, 2]
        assert abs(_max_drawdown_r([-1.0, -1.0, -1.0]) - 2.0) < 1e-9


# =========================================================================
# compute_scenario_metrics
# =========================================================================

class TestScenarioMetrics:
    def test_zero_ticks_matches_baseline(self):
        """0-tick scenario should reproduce baseline R values."""
        t = _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0)
        tick_vals = {"/PA": 5.0}

        sm = compute_scenario_metrics([t], n_ticks=0, tick_values=tick_vals)

        risk = 5.0
        expected_theo = ((110 - 105) / risk + (112 - 105) / risk
                         + (115 - 105) / risk) / 3.0
        assert abs(sm.total_r - expected_theo) < 1e-9
        assert sm.n_trades == 1
        assert sm.n_excluded == 0

    def test_higher_ticks_lower_total_r(self):
        """Increasing slippage should reduce total R."""
        trades = [
            _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0),
        ]
        tick_vals = {"/PA": 5.0}

        sm0 = compute_scenario_metrics(trades, 0, tick_vals)
        sm1 = compute_scenario_metrics(trades, 1, tick_vals)
        sm2 = compute_scenario_metrics(trades, 2, tick_vals)

        assert sm0.total_r > sm1.total_r > sm2.total_r

    def test_excluded_trades_counted(self):
        """Trades with risk <= 0 after slippage are excluded."""
        # entry exactly at stop: risk = 0 even at 0 ticks.
        t = _trade(entry_price=100.0, stop_price=100.0)
        sm = compute_scenario_metrics([t], 0, {"/PA": 5.0})

        assert sm.n_trades == 0
        assert sm.n_excluded == 1

    def test_win_rate_decreases_with_slippage(self):
        """Higher slippage turns some winning trades negative."""
        trades = [
            _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0),
        ]
        tick_vals = {"/PA": 5.0}

        sm0 = compute_scenario_metrics(trades, 0, tick_vals)
        sm2 = compute_scenario_metrics(trades, 2, tick_vals)

        assert sm0.win_rate >= sm2.win_rate

    def test_mixed_symbols(self):
        """Different tick values per symbol are applied correctly."""
        t_pa = _trade(symbol="/PA", entry_price=105.0, stop_price=100.0,
                       exit_price=115.0)
        t_xle = _trade(symbol="XLE", entry_price=85.0, stop_price=83.0,
                        exit_price=88.0, tp1=87.0, tp2=88.0, tp3=89.0,
                        tp1_hit=True, tp2_hit=True, tp3_hit=False)
        tick_vals = {"/PA": 5.0}  # XLE uses equity_tick default

        sm = compute_scenario_metrics(
            [t_pa, t_xle], n_ticks=1, tick_values=tick_vals, equity_tick=0.01,
        )
        # Both trades should be included (neither has risk <= 0).
        assert sm.n_trades == 2

    def test_mar_proxy_computation(self):
        """MAR proxy = total_r / max_drawdown."""
        # Two trades: one win, one loss.
        t_win = _trade(entry_price=105.0, stop_price=100.0, exit_price=115.0)
        t_loss = _trade(entry_price=105.0, stop_price=100.0,
                         exit_price=94.0, exit_reason="Stop",
                         tp1_hit=False, tp2_hit=False, tp3_hit=False)
        tick_vals = {"/PA": 5.0}

        sm = compute_scenario_metrics([t_win, t_loss], 0, tick_vals)
        assert sm.mar_proxy != float("inf")
        # MAR = total_r / max_dd.  max_dd > 0 because of the loss.
        assert sm.mar_proxy == pytest.approx(
            sm.total_r / sm.max_drawdown_r, abs=1e-6,
        )

    def test_empty_results(self):
        """No trades -> zero metrics."""
        sm = compute_scenario_metrics([], 0, {"/PA": 5.0})
        assert sm.n_trades == 0
        assert sm.total_r == 0.0


# =========================================================================
# compute_degradation
# =========================================================================

class TestDegradation:
    def test_degradation_total_r(self):
        """Total R % change is computed correctly."""
        baseline = ScenarioMetrics(
            ticks=0, n_trades=10, n_excluded=0,
            total_r=20.0, avg_r=2.0, win_rate=0.8,
            total_actual_r=18.0, max_drawdown_r=4.0, mar_proxy=5.0,
        )
        stressed = ScenarioMetrics(
            ticks=2, n_trades=10, n_excluded=0,
            total_r=14.0, avg_r=1.4, win_rate=0.6,
            total_actual_r=12.0, max_drawdown_r=6.0, mar_proxy=14.0 / 6.0,
        )
        deg = compute_degradation(baseline, stressed)

        # total_r_pct = (14 - 20) / 20 * 100 = -30%
        assert abs(deg.total_r_pct - (-30.0)) < 1e-9
        # win_rate_pp = (0.6 - 0.8) * 100 = -20
        assert abs(deg.win_rate_pp - (-20.0)) < 1e-9
        # mar_pct = (2.333 - 5.0) / 5.0 * 100 = -53.33
        expected_mar_pct = ((14.0 / 6.0) - 5.0) / 5.0 * 100
        assert abs(deg.mar_pct - expected_mar_pct) < 1e-6

    def test_degradation_zero_baseline(self):
        """Zero baseline total R -> pct is 0 if stressed is also zero."""
        baseline = ScenarioMetrics(
            ticks=0, n_trades=5, n_excluded=0,
            total_r=0.0, avg_r=0.0, win_rate=0.5,
            total_actual_r=0.0, max_drawdown_r=0.0, mar_proxy=float("inf"),
        )
        stressed = ScenarioMetrics(
            ticks=1, n_trades=5, n_excluded=0,
            total_r=0.0, avg_r=0.0, win_rate=0.4,
            total_actual_r=0.0, max_drawdown_r=0.0, mar_proxy=float("inf"),
        )
        deg = compute_degradation(baseline, stressed)
        assert deg.total_r_pct == 0.0

    def test_degradation_ticks_field(self):
        """DegradationResult.ticks matches the stressed scenario."""
        baseline = ScenarioMetrics(
            ticks=0, n_trades=5, n_excluded=0,
            total_r=10.0, avg_r=2.0, win_rate=0.8,
            total_actual_r=10.0, max_drawdown_r=2.0, mar_proxy=5.0,
        )
        stressed = ScenarioMetrics(
            ticks=3, n_trades=5, n_excluded=0,
            total_r=5.0, avg_r=1.0, win_rate=0.6,
            total_actual_r=5.0, max_drawdown_r=3.0, mar_proxy=5.0 / 3.0,
        )
        deg = compute_degradation(baseline, stressed)
        assert deg.ticks == 3


# =========================================================================
# run_stress_test: end-to-end
# =========================================================================

class TestRunStressTest:
    def test_gate_pass_profitable_at_2_ticks(self):
        """If total R > 0 at 2 ticks, gate passes."""
        # Use an equity trade — tiny tick value keeps edge alive.
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=83.0,
                    exit_price=89.0, tp1=87.0, tp2=88.0, tp3=89.0),
        ]
        report = run_stress_test(
            trades, tick_values={}, equity_tick=0.01,
        )
        assert report.gate_pass is True
        assert report.profitable_at_2_ticks is True

    def test_gate_fail_unprofitable_at_2_ticks(self):
        """If total R <= 0 at 2 ticks, gate fails."""
        # /PA trade with tight risk — slippage kills the edge.
        trades = [
            _trade(symbol="/PA", entry_price=105.0, stop_price=100.0,
                    exit_price=115.0),
        ]
        report = run_stress_test(
            trades, tick_values={"/PA": 5.0},
        )
        # At 2 ticks: slippage=$10.  new_entry=115, risk=15.
        # All TPs at or below new entry → negative R.
        assert report.gate_pass is False
        assert report.profitable_at_2_ticks is False

    def test_report_has_all_scenarios(self):
        """Default tick_levels [0,1,2,3] -> 4 scenarios."""
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=80.0,
                    exit_price=95.0, tp1=90.0, tp2=92.0, tp3=95.0),
        ]
        report = run_stress_test(trades, tick_values={}, equity_tick=0.01)

        assert len(report.scenarios) == 4
        assert [s.ticks for s in report.scenarios] == [0, 1, 2, 3]

    def test_report_has_degradation_for_non_baseline(self):
        """Degradation list has entries for scenarios 1, 2, 3."""
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=80.0,
                    exit_price=95.0, tp1=90.0, tp2=92.0, tp3=95.0),
        ]
        report = run_stress_test(trades, tick_values={}, equity_tick=0.01)

        assert len(report.degradations) == 3
        assert [d.ticks for d in report.degradations] == [1, 2, 3]

    def test_profitable_at_3_ticks_tracked(self):
        """profitable_at_3_ticks is populated when 3 ticks is tested."""
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=80.0,
                    exit_price=95.0, tp1=90.0, tp2=92.0, tp3=95.0),
        ]
        report = run_stress_test(trades, tick_values={}, equity_tick=0.01)

        assert report.profitable_at_3_ticks is not None

    def test_custom_tick_levels(self):
        """Custom tick_levels respected."""
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=80.0,
                    exit_price=95.0, tp1=90.0, tp2=92.0, tp3=95.0),
        ]
        report = run_stress_test(
            trades, tick_values={}, equity_tick=0.01,
            tick_levels=[0, 2, 4],
        )
        assert len(report.scenarios) == 3
        assert [s.ticks for s in report.scenarios] == [0, 2, 4]
        # 3 ticks not tested -> None.
        assert report.profitable_at_3_ticks is None

    def test_total_r_monotonically_decreases(self):
        """Total R should decrease or stay constant with more slippage."""
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=80.0,
                    exit_price=95.0, tp1=90.0, tp2=92.0, tp3=95.0),
        ]
        report = run_stress_test(trades, tick_values={}, equity_tick=0.01)
        totals = [s.total_r for s in report.scenarios]

        for i in range(1, len(totals)):
            assert totals[i] <= totals[i - 1] + 1e-9

    def test_degradation_values_negative(self):
        """Degradation pct should be non-positive (slippage only hurts)."""
        trades = [
            _trade(symbol="XLE", entry_price=85.0, stop_price=80.0,
                    exit_price=95.0, tp1=90.0, tp2=92.0, tp3=95.0),
        ]
        report = run_stress_test(trades, tick_values={}, equity_tick=0.01)

        for deg in report.degradations:
            assert deg.total_r_pct <= 1e-9  # non-positive
