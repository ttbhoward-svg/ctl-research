"""Unit tests for Phase 1a entry degradation test."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.entry_degradation import (
    AggregateMetrics,
    DegradationDeltas,
    DegradationReport,
    _compute_aggregate,
    _compute_deltas,
    _select_degraded_indices,
    degrade_adverse_fill,
    degrade_delayed_entry,
    run_entry_degradation,
)
from ctl.simulator import TradeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    symbol: str = "/ES",
    entry_price: float = 100.0,
    stop_price: float = 95.0,
    exit_price: float = 110.0,
    exit_reason: str = "TP1",
    tp1: float = 105.0,
    tp2: float = 110.0,
    tp3: float = 115.0,
    tp1_hit: bool = True,
    tp2_hit: bool = False,
    tp3_hit: bool = False,
    entry_bar_idx: int = 10,
    exit_bar_idx: int = 15,
    theoretical_r: float | None = None,
) -> TradeResult:
    risk = entry_price - stop_price
    if theoretical_r is None:
        parts = []
        for level, hit in [(tp1, tp1_hit), (tp2, tp2_hit), (tp3, tp3_hit)]:
            if hit:
                parts.append((level - entry_price) / risk)
            else:
                parts.append((exit_price - entry_price) / risk)
        theoretical_r = sum(p / 3.0 for p in parts)

    actual_r = (exit_price - entry_price) / risk if risk != 0 else 0.0

    return TradeResult(
        symbol=symbol,
        timeframe="daily",
        entry_price=entry_price,
        stop_price=stop_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        tp1=tp1, tp2=tp2, tp3=tp3,
        tp4=120.0, tp5=125.0,
        tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
        risk_per_unit=risk,
        r_multiple_actual=actual_r,
        theoretical_r=theoretical_r,
        mfe_r=2.0,
        mae_r=0.5,
        entry_bar_idx=entry_bar_idx,
        exit_bar_idx=exit_bar_idx,
        trade_outcome="Win" if tp1_hit else "Loss",
        hold_bars=exit_bar_idx - entry_bar_idx,
        swing_high=115.0,
        entry_date=pd.Timestamp("2020-01-15"),
        exit_date=pd.Timestamp("2020-01-22"),
        trigger_date=pd.Timestamp("2020-01-14"),
        trigger_bar_idx=9,
    )


def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    """Simple uptrend OHLCV for delayed entry re-walk."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 95.0 + np.arange(n) * 0.5
    return pd.DataFrame({
        "Date": dates,
        "Open": base,
        "High": base + 2.0,
        "Low": base - 1.0,
        "Close": base + 1.0,
        "Volume": np.full(n, 1000.0),
    })


# ---------------------------------------------------------------------------
# Tests: selection
# ---------------------------------------------------------------------------

class TestSelectDegradedIndices:
    def test_correct_count(self):
        idx = _select_degraded_indices(100, 0.30, seed=42)
        assert len(idx) == 30

    def test_deterministic(self):
        a = _select_degraded_indices(100, 0.30, seed=42)
        b = _select_degraded_indices(100, 0.30, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_selection(self):
        a = _select_degraded_indices(100, 0.30, seed=42)
        b = _select_degraded_indices(100, 0.30, seed=99)
        assert not np.array_equal(a, b)

    def test_zero_pct_empty(self):
        idx = _select_degraded_indices(100, 0.0, seed=42)
        assert len(idx) == 0

    def test_full_pct(self):
        idx = _select_degraded_indices(10, 1.0, seed=42)
        assert len(idx) == 10

    def test_no_duplicates(self):
        idx = _select_degraded_indices(50, 0.5, seed=42)
        assert len(set(idx)) == len(idx)


# ---------------------------------------------------------------------------
# Tests: adverse fill
# ---------------------------------------------------------------------------

class TestAdverseFill:
    def test_worsens_entry(self):
        trade = _make_trade(entry_price=100.0, stop_price=95.0)
        degraded = degrade_adverse_fill(trade, adverse_amount=1.0)
        assert degraded is not None
        assert degraded.entry_price == 101.0
        assert degraded.risk_per_unit == pytest.approx(6.0)  # 101 - 95

    def test_theoretical_r_decreases(self):
        trade = _make_trade()
        degraded = degrade_adverse_fill(trade, adverse_amount=1.0)
        assert degraded is not None
        assert degraded.theoretical_r < trade.theoretical_r

    def test_risk_zero_returns_none(self):
        # Degenerate trade: stop above entry. risk = entry - stop < 0.
        # new_risk = (entry + adverse) - stop. Returns None when <= 0.
        trade = _make_trade(entry_price=99.0, stop_price=100.0, theoretical_r=0.0)
        # new_entry=99.5, new_risk=99.5-100=-0.5 <= 0 => None.
        assert degrade_adverse_fill(trade, adverse_amount=0.5) is None
        # new_entry=101.01, new_risk=101.01-100=1.01 > 0 => viable.
        assert degrade_adverse_fill(trade, adverse_amount=2.01) is not None

    def test_preserves_exit_price_for_tp(self):
        trade = _make_trade(exit_reason="TP1", exit_price=105.0)
        degraded = degrade_adverse_fill(trade, adverse_amount=1.0)
        assert degraded is not None
        assert degraded.exit_price == 105.0  # TP: limit order, unchanged

    def test_preserves_exit_price_for_stop(self):
        trade = _make_trade(exit_reason="Stop", exit_price=94.0)
        degraded = degrade_adverse_fill(trade, adverse_amount=1.0)
        assert degraded is not None
        assert degraded.exit_price == 94.0  # stop fill independent of entry


# ---------------------------------------------------------------------------
# Tests: delayed entry
# ---------------------------------------------------------------------------

class TestDelayedEntry:
    def test_entry_bar_shifted(self):
        trade = _make_trade(entry_bar_idx=10)
        df = _make_ohlcv(30)
        arrays = {
            "opens": df["Open"].values.astype(float),
            "highs": df["High"].values.astype(float),
            "lows": df["Low"].values.astype(float),
            "closes": df["Close"].values.astype(float),
            "dates": df["Date"].values,
            "n_bars": len(df),
        }
        degraded = degrade_delayed_entry(trade, arrays, delay_bars=1)
        assert degraded is not None
        assert degraded.entry_bar_idx == 11

    def test_new_entry_price_from_open(self):
        trade = _make_trade(entry_bar_idx=10)
        df = _make_ohlcv(30)
        arrays = {
            "opens": df["Open"].values.astype(float),
            "highs": df["High"].values.astype(float),
            "lows": df["Low"].values.astype(float),
            "closes": df["Close"].values.astype(float),
            "dates": df["Date"].values,
            "n_bars": len(df),
        }
        degraded = degrade_delayed_entry(trade, arrays, delay_bars=1)
        assert degraded is not None
        expected_entry = float(df["Open"].iloc[11])
        assert degraded.entry_price == pytest.approx(expected_entry)

    def test_beyond_data_returns_none(self):
        trade = _make_trade(entry_bar_idx=28)
        df = _make_ohlcv(30)
        arrays = {
            "opens": df["Open"].values.astype(float),
            "highs": df["High"].values.astype(float),
            "lows": df["Low"].values.astype(float),
            "closes": df["Close"].values.astype(float),
            "dates": df["Date"].values,
            "n_bars": len(df),
        }
        degraded = degrade_delayed_entry(trade, arrays, delay_bars=5)
        assert degraded is None  # 28 + 5 = 33 >= 30

    def test_deterministic(self):
        trade = _make_trade(entry_bar_idx=5)
        df = _make_ohlcv(30)
        arrays = {
            "opens": df["Open"].values.astype(float),
            "highs": df["High"].values.astype(float),
            "lows": df["Low"].values.astype(float),
            "closes": df["Close"].values.astype(float),
            "dates": df["Date"].values,
            "n_bars": len(df),
        }
        r1 = degrade_delayed_entry(trade, arrays, delay_bars=1)
        r2 = degrade_delayed_entry(trade, arrays, delay_bars=1)
        assert r1.theoretical_r == r2.theoretical_r
        assert r1.entry_price == r2.entry_price


# ---------------------------------------------------------------------------
# Tests: aggregate metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_basic(self):
        m = _compute_aggregate([1.0, 2.0, -0.5])
        assert m.n_trades == 3
        assert m.total_r == pytest.approx(2.5)
        assert m.avg_r == pytest.approx(2.5 / 3)
        assert m.win_rate == pytest.approx(2.0 / 3)

    def test_empty(self):
        m = _compute_aggregate([])
        assert m.n_trades == 0
        assert m.total_r == 0.0

    def test_all_wins(self):
        m = _compute_aggregate([1.0, 1.0, 1.0])
        assert m.win_rate == 1.0
        assert m.max_dd_r == 0.0


# ---------------------------------------------------------------------------
# Tests: deltas
# ---------------------------------------------------------------------------

class TestDeltas:
    def test_degradation_negative(self):
        baseline = AggregateMetrics(
            n_trades=10, total_r=10.0, avg_r=1.0,
            win_rate=0.8, max_dd_r=2.0, mar_proxy=5.0,
        )
        degraded = AggregateMetrics(
            n_trades=10, total_r=8.0, avg_r=0.8,
            win_rate=0.7, max_dd_r=3.0, mar_proxy=2.67,
        )
        d = _compute_deltas(baseline, degraded)
        assert d.total_r_pct == pytest.approx(-20.0)
        assert d.win_rate_pp == pytest.approx(-10.0)
        assert d.mar_pct == pytest.approx((2.67 - 5.0) / 5.0 * 100)

    def test_zero_baseline(self):
        baseline = AggregateMetrics(
            n_trades=10, total_r=0.0, avg_r=0.0,
            win_rate=0.5, max_dd_r=0.0, mar_proxy=0.0,
        )
        degraded = AggregateMetrics(
            n_trades=10, total_r=0.0, avg_r=0.0,
            win_rate=0.5, max_dd_r=0.0, mar_proxy=0.0,
        )
        d = _compute_deltas(baseline, degraded)
        assert d.total_r_pct == 0.0
        assert d.win_rate_pp == 0.0


# ---------------------------------------------------------------------------
# Tests: pass/fail logic
# ---------------------------------------------------------------------------

class TestPassFail:
    def test_within_tolerance_passes(self):
        trades = [_make_trade(theoretical_r=1.0) for _ in range(20)]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=0.3,
            seed=42, adverse_fill_fraction=0.01,  # tiny degradation
        )
        assert report.total_r_pass
        assert report.win_rate_pass
        assert report.mar_pass

    def test_severe_degradation_fails(self):
        trades = [
            _make_trade(
                entry_price=100.0, stop_price=99.5,  # tight risk
                tp1=101.0, tp2=102.0, tp3=103.0,
                exit_price=101.0, exit_reason="TP1",
                tp1_hit=True,
            )
            for _ in range(20)
        ]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=1.0,
            seed=42, adverse_fill_fraction=0.5,  # huge adverse fill
        )
        # With 50% adverse fill on tight risk, many trades become unviable.
        assert report.n_excluded > 0 or not report.total_r_pass

    def test_report_structure(self):
        trades = [_make_trade() for _ in range(10)]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=0.3, seed=42,
        )
        assert isinstance(report, DegradationReport)
        assert report.mode == "adverse_fill"
        assert report.degradation_pct == 0.3
        assert report.seed == 42
        assert report.n_total == 10
        assert report.verdict in ("PASS", "FAIL")

    def test_summary_string(self):
        trades = [_make_trade() for _ in range(10)]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=0.3, seed=42,
        )
        s = report.summary()
        assert "Entry Degradation Report" in s
        assert "adverse_fill" in s


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_results(self):
        report = run_entry_degradation([], mode="adverse_fill")
        assert report.n_total == 0
        assert report.all_passed

    def test_single_trade(self):
        trades = [_make_trade()]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=1.0, seed=42,
        )
        assert report.n_total == 1
        assert report.n_degraded == 1

    def test_all_trades_degraded(self):
        trades = [_make_trade() for _ in range(5)]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=1.0,
            seed=42, adverse_fill_fraction=0.05,
        )
        assert report.n_degraded == 5

    def test_zero_degradation_matches_baseline(self):
        trades = [_make_trade(theoretical_r=1.5) for _ in range(10)]
        report = run_entry_degradation(
            trades, mode="adverse_fill", degradation_pct=0.0, seed=42,
        )
        assert report.n_degraded == 0
        assert report.degraded.total_r == pytest.approx(report.baseline.total_r)
        assert report.deltas.total_r_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: delayed entry with data
# ---------------------------------------------------------------------------

class TestDelayedEntryIntegration:
    def test_delayed_entry_mode(self):
        df = _make_ohlcv(30)
        trades = [_make_trade(entry_bar_idx=5, symbol="/ES")]
        report = run_entry_degradation(
            trades,
            data_by_symbol={"/ES": df},
            mode="delayed_entry",
            degradation_pct=1.0,
            seed=42,
        )
        assert report.n_degraded == 1
        assert isinstance(report, DegradationReport)

    def test_combined_mode(self):
        df = _make_ohlcv(30)
        trades = [_make_trade(entry_bar_idx=5, symbol="/ES")]
        report = run_entry_degradation(
            trades,
            data_by_symbol={"/ES": df},
            mode="combined",
            degradation_pct=1.0,
            seed=42,
        )
        assert report.n_degraded == 1

    def test_deterministic_full_run(self):
        df = _make_ohlcv(30)
        trades = [_make_trade(entry_bar_idx=i, symbol="/ES") for i in range(3, 8)]
        r1 = run_entry_degradation(
            trades, data_by_symbol={"/ES": df},
            mode="delayed_entry", degradation_pct=0.5, seed=42,
        )
        r2 = run_entry_degradation(
            trades, data_by_symbol={"/ES": df},
            mode="delayed_entry", degradation_pct=0.5, seed=42,
        )
        assert r1.degraded.total_r == r2.degraded.total_r
        assert r1.deltas.total_r_pct == r2.deltas.total_r_pct
