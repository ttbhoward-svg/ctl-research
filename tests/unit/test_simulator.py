"""Tests for the trade simulator.

Each test targets a specific execution convention or edge case.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.b1_detector import B1Trigger
from ctl.simulator import (
    SimConfig,
    TradeResult,
    _compute_theoretical_r,
    results_to_dataframe,
    simulate_all,
    simulate_trade,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal DataFrames with controlled price paths
# ---------------------------------------------------------------------------

def _dates(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _flat_df(n: int, price: float = 100.0) -> pd.DataFrame:
    """Flat price bars — useful as a canvas to surgically set individual bars."""
    return pd.DataFrame({
        "Date": _dates(n),
        "Open": [price] * n,
        "High": [price + 1.0] * n,
        "Low": [price - 1.0] * n,
        "Close": [price] * n,
        "Volume": [10000.0] * n,
    })


def _trigger(
    trigger_idx: int = 5,
    entry_idx: int = 8,
    stop: float = 95.0,
    tp1: float = 103.0,
    tp2: float = 105.0,
    tp3: float = 107.0,
    symbol: str = "/PA",
    timeframe: str = "daily",
) -> B1Trigger:
    """Create a confirmed trigger with explicit levels."""
    return B1Trigger(
        trigger_bar_idx=trigger_idx,
        trigger_date=pd.Timestamp("2024-01-09"),
        symbol=symbol,
        timeframe=timeframe,
        slope_20=10.0,
        bars_of_air=8,
        ema10_at_trigger=100.0,
        atr14_at_trigger=3.0,
        stop_price=stop,
        swing_high=107.0,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        tp4=115.0,
        tp5=125.0,
        confirmed=True,
        confirmation_bar_idx=entry_idx - 1,
        entry_bar_idx=entry_idx,
        entry_date=pd.Timestamp("2024-01-12"),
        entry_price=100.0,
    )


# ---------------------------------------------------------------------------
# Stop fill: close-based trigger, next-bar open fill
# ---------------------------------------------------------------------------

class TestStopFill:
    def test_stop_triggers_on_close_below(self):
        """Close < Stop on bar X -> exit at Open[X+1]."""
        df = _flat_df(20, price=100.0)
        # Bar 12: close drops below stop.
        df.at[12, "Close"] = 93.0
        df.at[12, "Low"] = 92.0
        # Bar 13: open at 94 (gap scenario).
        df.at[13, "Open"] = 94.0

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.exit_reason == "Stop"
        assert result.exit_bar_idx == 13
        assert result.exit_price == 94.0  # Open[13], no slippage

    def test_stop_with_slippage(self):
        """Stop fill should subtract slippage_per_side."""
        df = _flat_df(20, price=100.0)
        df.at[12, "Close"] = 93.0
        df.at[13, "Open"] = 94.0

        trig = _trigger(entry_idx=10, stop=95.0)
        config = SimConfig(slippage_per_side=0.50)
        result = simulate_trade(trig, df, config)

        assert result is not None
        # Entry price = Open[10] + slippage = 100 + 0.5 = 100.5
        assert result.entry_price == 100.5
        # Exit price = Open[13] - slippage = 94 - 0.5 = 93.5
        assert result.exit_price == 93.5

    def test_gap_through_stop(self):
        """Gap open well below stop — exit at that open, not at stop level."""
        df = _flat_df(20, price=100.0)
        df.at[12, "Close"] = 93.0  # breach
        df.at[13, "Open"] = 80.0   # severe gap down

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.exit_price == 80.0  # NOT clamped to 95
        assert result.exit_reason == "Stop"

    def test_stop_on_last_bar(self):
        """Stop breach on final bar -> fallback to Close of that bar."""
        df = _flat_df(15, price=100.0)
        df.at[14, "Close"] = 90.0  # last bar breaches

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.exit_on_last_bar is True
        assert result.exit_reason == "Stop"
        assert result.exit_price == 90.0

    def test_close_exactly_at_stop_no_breach(self):
        """Close == Stop is NOT a breach (spec: Close < StopPrice)."""
        df = _flat_df(20, price=100.0)
        df.at[12, "Close"] = 95.0  # exactly at stop

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        # Should NOT have stopped on bar 12.
        assert result.exit_reason != "Stop" or result.exit_bar_idx != 13


# ---------------------------------------------------------------------------
# TP fill: high-based, at level exactly
# ---------------------------------------------------------------------------

class TestTPFill:
    def test_tp1_fills_at_level(self):
        """High >= TP1 -> partial exit at TP1 price, trade continues."""
        df = _flat_df(30, price=100.0)
        # Bar 15: high reaches TP1.
        df.at[15, "High"] = 103.5

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.tp1_hit is True
        # Trade should continue (not exit at TP1 alone unless TP3 also hit).

    def test_tp3_full_exit(self):
        """High >= TP3 -> all out at TP3."""
        df = _flat_df(30, price=100.0)
        df.at[15, "High"] = 108.0  # above TP3=107

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.exit_reason == "TP3"
        assert result.exit_price == 107.0  # at level, not at High
        assert result.tp1_hit is True
        assert result.tp2_hit is True
        assert result.tp3_hit is True

    def test_multi_tp_same_bar_highest_wins(self):
        """If High exceeds TP3 in one bar (skipping TP1, TP2), exit at TP3."""
        df = _flat_df(20, price=100.0)
        # No prior TP hits. Bar 12: huge spike.
        df.at[12, "High"] = 120.0

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.exit_reason == "TP3"
        assert result.exit_price == 107.0
        assert all([result.tp1_hit, result.tp2_hit, result.tp3_hit])

    def test_tp1_then_stop(self):
        """TP1 hit on bar A, then stopped on bar B -> Win outcome."""
        df = _flat_df(30, price=100.0)
        # Bar 14: TP1 hit.
        df.at[14, "High"] = 104.0
        # Bar 20: stop breach.
        df.at[20, "Close"] = 93.0
        df.at[21, "Open"] = 94.0

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.tp1_hit is True
        assert result.exit_reason == "Stop"
        assert result.trade_outcome == "Win"  # TP1 was hit -> Win


# ---------------------------------------------------------------------------
# Same-bar collision: TP hit AND stop breach
# ---------------------------------------------------------------------------

class TestCollision:
    def test_tp_wins_default(self):
        """Default collision rule: TP wins over stop."""
        df = _flat_df(20, price=100.0)
        # Bar 12: High reaches TP3 AND Close drops below stop.
        df.at[12, "High"] = 108.0
        df.at[12, "Close"] = 90.0
        df.at[12, "Low"] = 89.0

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df, SimConfig(collision_rule="tp_wins"))

        assert result is not None
        assert result.exit_reason == "TP3"
        assert result.same_bar_collision is True
        assert result.exit_price == 107.0

    def test_stop_wins_alternative(self):
        """Alternative collision rule: stop wins over TP."""
        df = _flat_df(20, price=100.0)
        df.at[12, "High"] = 108.0
        df.at[12, "Close"] = 90.0
        df.at[13, "Open"] = 91.0

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df, SimConfig(collision_rule="stop_wins"))

        assert result is not None
        assert result.exit_reason == "Stop"
        assert result.same_bar_collision is True
        assert result.exit_price == 91.0  # next-bar open

    def test_collision_flag_logged(self):
        """same_bar_collision flag is True when both conditions met."""
        df = _flat_df(20, price=100.0)
        df.at[12, "High"] = 104.0  # hits TP1
        df.at[12, "Close"] = 93.0  # breaches stop

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df, SimConfig(collision_rule="tp_wins"))

        assert result is not None
        assert result.same_bar_collision is True


# ---------------------------------------------------------------------------
# R-multiple and TheoreticalR
# ---------------------------------------------------------------------------

class TestRMultiple:
    def test_full_stop_loss_r(self):
        """Stopped before TP1 -> negative R."""
        df = _flat_df(20, price=100.0)
        df.at[12, "Close"] = 93.0
        df.at[13, "Open"] = 94.0

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        # R = (94 - 100) / (100 - 95) = -6/5 = -1.2
        assert abs(result.r_multiple_actual - (-1.2)) < 0.01

    def test_theoretical_r_all_stopped(self):
        """All thirds stopped -> TheoreticalR = actual R."""
        df = _flat_df(20, price=100.0)
        df.at[12, "Close"] = 93.0
        df.at[13, "Open"] = 94.0

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert abs(result.theoretical_r - result.r_multiple_actual) < 0.01

    def test_theoretical_r_tp3_hit(self):
        """All three TPs hit -> TheoreticalR = (TP1_R + TP2_R + TP3_R) / 3."""
        df = _flat_df(20, price=100.0)
        df.at[12, "High"] = 108.0  # hits all three TPs

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        risk = 100.0 - 95.0  # 5.0
        expected = ((103.0 - 100) / risk + (105.0 - 100) / risk + (107.0 - 100) / risk) / 3.0
        assert abs(result.theoretical_r - expected) < 0.01

    def test_theoretical_r_tp1_then_stop(self):
        """TP1 hit, then stopped -> 1/3 at TP1 R, 2/3 at stop R."""
        tp1_r = (103.0 - 100.0) / 5.0       # 0.6
        stop_exit_r = (94.0 - 100.0) / 5.0   # -1.2
        expected = (tp1_r + stop_exit_r + stop_exit_r) / 3.0

        df = _flat_df(30, price=100.0)
        df.at[14, "High"] = 104.0  # TP1 hit
        df.at[20, "Close"] = 93.0  # stop breach
        df.at[21, "Open"] = 94.0

        trig = _trigger(entry_idx=10, tp1=103.0, tp2=105.0, tp3=107.0, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert abs(result.theoretical_r - expected) < 0.01


# ---------------------------------------------------------------------------
# MFE / MAE / Day1Fail
# ---------------------------------------------------------------------------

class TestMFEMAE:
    def test_mfe_captures_highest_high(self):
        df = _flat_df(20, price=100.0)
        df.at[12, "High"] = 110.0  # highest point
        df.at[15, "Close"] = 93.0  # stop
        df.at[16, "Open"] = 94.0

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        # MFE = (110 - 100) / 5 = 2.0
        assert abs(result.mfe_r - 2.0) < 0.01

    def test_mae_captures_lowest_low(self):
        df = _flat_df(20, price=100.0)
        df.at[11, "Low"] = 93.0    # deepest drawdown
        df.at[15, "High"] = 108.0  # TP3 exit

        trig = _trigger(entry_idx=10, tp3=107.0, stop=90.0)
        result = simulate_trade(trig, df)

        assert result is not None
        # MAE = (100 - 93) / (100 - 90) = 0.7
        assert abs(result.mae_r - 0.7) < 0.01

    def test_day1_fail_flag(self):
        """Low of entry bar < StopPrice -> Day1Fail = True."""
        df = _flat_df(20, price=100.0)
        df.at[10, "Low"] = 94.0   # entry bar low below stop

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.day1_fail is True

    def test_day1_no_fail(self):
        """Low of entry bar >= StopPrice -> Day1Fail = False."""
        df = _flat_df(20, price=100.0)
        df.at[10, "Low"] = 96.0   # above stop

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.day1_fail is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unconfirmed_trigger_returns_none(self):
        trig = _trigger()
        trig.confirmed = False
        result = simulate_trade(trig, _flat_df(20))
        assert result is None

    def test_entry_beyond_data_returns_none(self):
        df = _flat_df(10)
        trig = _trigger(entry_idx=15)  # beyond data
        result = simulate_trade(trig, df)
        assert result is None

    def test_degenerate_risk_returns_none(self):
        """Entry at or below stop -> risk_per_unit <= 0 -> skip."""
        df = _flat_df(20, price=100.0)
        df.at[10, "Open"] = 90.0  # entry open below stop

        trig = _trigger(entry_idx=10, stop=95.0)
        result = simulate_trade(trig, df)
        assert result is None

    def test_trade_open_at_end_of_data(self):
        """No stop or TP hit -> trade still open."""
        df = _flat_df(20, price=100.0)
        # No breach, no TP hit (all highs at 101, TPs at 103+).
        trig = _trigger(entry_idx=10, stop=95.0, tp1=103.0)
        result = simulate_trade(trig, df)

        assert result is not None
        assert result.exit_reason == "Open"
        assert result.exit_on_last_bar is True
        assert result.trade_outcome == "Open"

    def test_simulate_all_batch(self):
        """simulate_all processes multiple triggers."""
        df = _flat_df(30, price=100.0)
        df.at[15, "High"] = 108.0  # TP3 for first trade

        t1 = _trigger(entry_idx=10, tp3=107.0, stop=95.0)
        t2 = _trigger(entry_idx=20, tp3=107.0, stop=95.0)
        t2.trigger_bar_idx = 17
        t2.trigger_date = pd.Timestamp("2024-01-25")

        results = simulate_all([t1, t2], df)
        assert len(results) == 2

    def test_results_to_dataframe(self):
        df = _flat_df(20, price=100.0)
        df.at[15, "High"] = 108.0
        trig = _trigger(entry_idx=10, tp3=107.0, stop=95.0)
        results = simulate_all([trig], df)
        out = results_to_dataframe(results)
        assert len(out) == 1
        assert "TheoreticalR" in out.columns
        assert "ExitReason" in out.columns
        assert "HoldBars" in out.columns
