"""Tests for B1 confluence flags (Spec §8)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.b1_detector import (
    B1Params,
    _clean_pullback,
    _fib_confluence,
    _gap_fill_below,
    _multi_year_highs,
    _single_bar_pullback,
    _volume_declining,
    _wr_divergence,
    compute_indicators,
    detect_triggers,
)


# =========================================================================
# WR Divergence (§8.1)
# =========================================================================

class TestWRDivergence:
    def test_positive_divergence(self):
        """Lower low + higher WR = bullish divergence -> True."""
        n = 20
        lows = np.full(n, 110.0)
        ema10 = np.full(n, 100.0)
        wr = np.full(n, -50.0)

        # Prior touch at bar 5: Low <= EMA10, WR = -80.
        lows[5] = 95.0
        ema10[5] = 100.0
        wr[5] = -80.0

        # Air bars 6-14: Low > EMA10.
        # Trigger at bar 15: Low <= EMA10, WR = -50, Low <= Low[5].
        lows[15] = 90.0
        ema10[15] = 100.0
        wr[15] = -50.0  # higher WR than -80

        assert _wr_divergence(lows, wr, ema10, 15) is True

    def test_no_prior_touch(self):
        """All bars have Low > EMA10 -> no prior touch -> False."""
        n = 20
        lows = np.full(n, 110.0)
        ema10 = np.full(n, 100.0)
        wr = np.full(n, -50.0)
        lows[19] = 95.0  # trigger bar touches, but no *prior* touch

        assert _wr_divergence(lows, wr, ema10, 19) is False

    def test_higher_low_no_divergence(self):
        """Price made a higher low (not divergence) -> False."""
        n = 20
        lows = np.full(n, 110.0)
        ema10 = np.full(n, 100.0)
        wr = np.full(n, -50.0)

        # Prior touch at bar 5.
        lows[5] = 90.0
        ema10[5] = 100.0
        wr[5] = -80.0

        # Trigger at bar 15: Low HIGHER than prior touch.
        lows[15] = 95.0
        ema10[15] = 100.0
        wr[15] = -50.0

        assert _wr_divergence(lows, wr, ema10, 15) is False

    def test_same_wr_no_divergence(self):
        """WR equal (not higher) -> False."""
        n = 20
        lows = np.full(n, 110.0)
        ema10 = np.full(n, 100.0)
        wr = np.full(n, -50.0)

        lows[5] = 90.0
        ema10[5] = 100.0
        wr[5] = -50.0

        lows[15] = 85.0
        ema10[15] = 100.0
        wr[15] = -50.0  # equal, not higher

        assert _wr_divergence(lows, wr, ema10, 15) is False

    def test_nan_wr_returns_false(self):
        """NaN WilliamsR at trigger -> False."""
        n = 20
        lows = np.full(n, 110.0)
        ema10 = np.full(n, 100.0)
        wr = np.full(n, np.nan)

        lows[5] = 90.0
        ema10[5] = 100.0
        lows[15] = 85.0
        ema10[15] = 100.0

        assert _wr_divergence(lows, wr, ema10, 15) is False


# =========================================================================
# Clean Pullback (§8.2)
# =========================================================================

class TestCleanPullback:
    def test_positive(self):
        """Three bars of lower highs + lower lows -> True."""
        highs = np.array([0, 0, 0, 0, 0, 0, 0, 30.0, 25.0, 20.0, 0], dtype=float)
        lows = np.array([0, 0, 0, 0, 0, 0, 0, 28.0, 23.0, 18.0, 0], dtype=float)
        # At n=10: N-1=9(H=20,L=18), N-2=8(H=25,L=23), N-3=7(H=30,L=28)
        # 20<25<30 and 18<23<28 -> True
        assert _clean_pullback(highs, lows, 10) is True

    def test_highs_not_declining(self):
        """Highs don't monotonically decline -> False."""
        highs = np.array([0, 0, 0, 0, 0, 0, 0, 30.0, 25.0, 26.0, 0], dtype=float)
        lows = np.array([0, 0, 0, 0, 0, 0, 0, 28.0, 23.0, 18.0, 0], dtype=float)
        # H[N-1]=26 < H[N-2]=25 is False
        assert _clean_pullback(highs, lows, 10) is False

    def test_lows_not_declining(self):
        """Lows don't monotonically decline -> False."""
        highs = np.array([0, 0, 0, 0, 0, 0, 0, 30.0, 25.0, 20.0, 0], dtype=float)
        lows = np.array([0, 0, 0, 0, 0, 0, 0, 28.0, 23.0, 24.0, 0], dtype=float)
        assert _clean_pullback(highs, lows, 10) is False

    def test_insufficient_bars(self):
        """n < 3 -> False."""
        highs = np.array([10.0, 9.0])
        lows = np.array([8.0, 7.0])
        assert _clean_pullback(highs, lows, 2) is False

    def test_equal_highs_fail(self):
        """Equal highs (ties) -> strict inequality fails -> False."""
        highs = np.array([0, 0, 0, 0, 0, 0, 0, 30.0, 30.0, 20.0, 0], dtype=float)
        lows = np.array([0, 0, 0, 0, 0, 0, 0, 28.0, 23.0, 18.0, 0], dtype=float)
        assert _clean_pullback(highs, lows, 10) is False


# =========================================================================
# Volume Declining (§8.3)
# =========================================================================

class TestVolumeDeclining:
    def test_positive(self):
        """Recent avg < prior avg -> True."""
        # Bars: ... [N-6..N-4] prior, [N-3..N-1] recent
        vols = np.array([0, 0, 0, 0, 1000, 1200, 1100, 500, 600, 400, 0], dtype=float)
        # n=10: recent=[N-1=400, N-2=600, N-3=500] avg=500
        #       prior=[N-4=1100, N-5=1200, N-6=1000] avg=1100
        assert _volume_declining(vols, 10) is True

    def test_negative(self):
        """Recent avg >= prior avg -> False."""
        vols = np.array([0, 0, 0, 0, 500, 600, 400, 1000, 1200, 1100, 0], dtype=float)
        assert _volume_declining(vols, 10) is False

    def test_insufficient_bars(self):
        """n < 6 -> False."""
        vols = np.array([100, 200, 300, 400, 500], dtype=float)
        assert _volume_declining(vols, 5) is False

    def test_nan_volume(self):
        """NaN in volume -> False."""
        vols = np.array([0, 0, 0, 0, np.nan, 100, 100, 50, 50, 50, 0], dtype=float)
        assert _volume_declining(vols, 10) is False


# =========================================================================
# Gap Fill Below (§8.4)
# =========================================================================

class TestGapFillBelow:
    def test_unfilled_gap_near_stop(self):
        """Unfilled gap-down within 2% below stop -> True."""
        n = 20
        opens = np.full(n, 105.0)
        closes = np.full(n, 106.0)
        highs = np.full(n, 107.0)

        # Gap down at bar 10: Open[10] < Close[9].
        closes[9] = 102.0
        opens[10] = 99.0  # gap from 102 to 99

        # Stop price = 100. Gap bottom = 99 (within 2% below 100).
        # Gap has NOT been filled: no bar from 10..19 has High >= 102 (gap_top).
        # Keep all highs < 102.
        for i in range(10, n):
            highs[i] = 101.0

        assert _gap_fill_below(opens, closes, highs, 100.0, 19, 100) is True

    def test_filled_gap(self):
        """Gap exists but has been filled -> False."""
        n = 20
        opens = np.full(n, 105.0)
        closes = np.full(n, 106.0)
        highs = np.full(n, 107.0)

        closes[9] = 102.0
        opens[10] = 99.0

        # Gap is filled: bar 15 has High >= gap_top (102).
        highs[15] = 103.0

        assert _gap_fill_below(opens, closes, highs, 100.0, 19, 100) is False

    def test_gap_too_far_below(self):
        """Gap bottom more than 2% below stop -> False."""
        n = 20
        opens = np.full(n, 105.0)
        closes = np.full(n, 106.0)
        highs = np.full(n, 101.0)

        closes[9] = 102.0
        opens[10] = 95.0  # gap bottom = 95, stop = 100, 95 < 100*0.98=98

        assert _gap_fill_below(opens, closes, highs, 100.0, 19, 100) is False

    def test_no_gap_down(self):
        """No gap-down bars -> False."""
        n = 20
        opens = np.full(n, 100.0)
        closes = np.full(n, 101.0)
        highs = np.full(n, 102.0)
        # Open always >= Close of prior bar -> no gap down.

        assert _gap_fill_below(opens, closes, highs, 100.0, 19, 100) is False


# =========================================================================
# Multi-Year Highs (§8.5)
# =========================================================================

class TestMultiYearHighs:
    def test_at_yearly_high(self):
        """SwingHigh == yearly high -> True (100% >= 95%)."""
        highs = np.linspace(80, 120, 300)
        swing_high = 120.0  # same as the max
        assert _multi_year_highs(highs, swing_high, 299) is True

    def test_within_five_percent(self):
        """SwingHigh at 96% of yearly high -> True."""
        highs = np.full(300, 100.0)
        highs[100] = 200.0  # yearly high = 200
        swing_high = 192.0  # 96% of 200
        assert _multi_year_highs(highs, swing_high, 299) is True

    def test_below_five_percent(self):
        """SwingHigh at 90% of yearly high -> False."""
        highs = np.full(300, 100.0)
        highs[100] = 200.0
        swing_high = 180.0  # 90% of 200
        assert _multi_year_highs(highs, swing_high, 299) is False

    def test_insufficient_bars(self):
        """n < 1 -> False."""
        highs = np.array([100.0])
        assert _multi_year_highs(highs, 100.0, 0) is False


# =========================================================================
# Single Bar Pullback (§8.7)
# =========================================================================

class TestSingleBarPullback:
    def test_positive(self):
        """Bar N-1 is the swing lookback high -> True."""
        highs = np.full(30, 100.0)
        highs[19] = 150.0  # N-1 is the swing high
        assert _single_bar_pullback(highs, 20, 20) is True

    def test_negative(self):
        """Bar N-1 is NOT the swing lookback high -> False."""
        highs = np.full(30, 100.0)
        highs[10] = 150.0  # bar 10 is the high, not N-1
        assert _single_bar_pullback(highs, 20, 20) is False

    def test_insufficient_bars(self):
        """n < 1 -> False."""
        highs = np.array([100.0])
        assert _single_bar_pullback(highs, 0, 20) is False


# =========================================================================
# Fib Confluence (§8.6)
# =========================================================================

class TestFibConfluence:
    def _make_weekly(self, n=50, high_val=200.0, low_val=100.0):
        """Weekly data with known swing high/low."""
        dates = pd.date_range("2019-01-04", periods=n, freq="W-FRI")
        highs = np.full(n, 150.0)
        lows = np.full(n, 120.0)
        # Set extremes in the window.
        highs[20] = high_val
        lows[10] = low_val
        return dates.values, highs, lows

    def test_positive(self):
        """TP1 within 1% of weekly fib 618 -> True."""
        dates, highs, lows = self._make_weekly(high_val=200.0, low_val=100.0)
        # HTF fib 618 = 100 + (200-100)*0.618 = 161.8
        tp1 = 161.8
        trigger = pd.Timestamp(dates[40])
        assert _fib_confluence(tp1, dates, highs, lows, trigger, 20) is True

    def test_negative(self):
        """TP1 far from weekly fib -> False."""
        dates, highs, lows = self._make_weekly(high_val=200.0, low_val=100.0)
        tp1 = 140.0  # well away from 161.8
        trigger = pd.Timestamp(dates[40])
        assert _fib_confluence(tp1, dates, highs, lows, trigger, 20) is False

    def test_trigger_before_data(self):
        """Trigger date before all weekly bars -> False."""
        dates, highs, lows = self._make_weekly()
        trigger = pd.Timestamp("2018-01-01")
        assert _fib_confluence(100.0, dates, highs, lows, trigger, 20) is False

    def test_zero_range(self):
        """Weekly swing range = 0 -> False."""
        n = 50
        dates = pd.date_range("2019-01-04", periods=n, freq="W-FRI").values
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        trigger = pd.Timestamp(dates[40])
        assert _fib_confluence(100.0, dates, highs, lows, trigger, 20) is False


# =========================================================================
# Integration: confluence flags on trigger output
# =========================================================================

def _make_perfect_b1() -> pd.DataFrame:
    """Minimal DataFrame that produces a confirmed B1 trigger.

    Duplicated from test_b1_detector.py for independence.
    """
    total = 213
    dates = pd.bdate_range("2019-01-02", periods=total)

    warmup_n = 200
    base = np.linspace(100, 160, warmup_n)
    air_base = np.linspace(160, 185, 10)
    tail_base = np.array([175.0, 180.0, 181.0])
    full_base = np.concatenate([base, air_base, tail_base])

    close = full_base.copy()
    high = full_base + 3.0
    low = full_base - 1.0
    opn = full_base.copy()
    vol = np.full(total, 50000.0)

    for i in range(warmup_n, warmup_n + 10):
        low[i] = full_base[i] + 0.5
        close[i] = full_base[i] + 1.5
        high[i] = full_base[i] + 4.0
        opn[i] = full_base[i] + 1.0

    trigger_idx = 210
    low[trigger_idx] = 168.0
    close[trigger_idx] = 175.0
    high[trigger_idx] = 178.0
    opn[trigger_idx] = 176.0

    confirm_idx = 211
    close[confirm_idx] = 180.0
    high[confirm_idx] = 182.0
    low[confirm_idx] = 177.0
    opn[confirm_idx] = 178.0

    entry_idx = 212
    opn[entry_idx] = 181.0
    close[entry_idx] = 182.0
    high[entry_idx] = 183.0
    low[entry_idx] = 180.0

    return pd.DataFrame({
        "Date": dates[:total],
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


class TestConfluenceIntegration:
    def test_confluence_fields_on_trigger(self):
        """Triggers have all 7 confluence flag fields with correct types."""
        df = _make_perfect_b1()
        params = B1Params()
        compute_indicators(df, params)
        triggers = detect_triggers(df, "/PA", "daily", params)

        assert len(triggers) >= 1
        for t in triggers:
            assert isinstance(t.wr_divergence, bool)
            assert isinstance(t.clean_pullback, bool)
            assert isinstance(t.volume_declining, bool)
            assert isinstance(t.gap_fill_below, bool)
            assert isinstance(t.multi_year_highs, bool)
            assert isinstance(t.single_bar_pullback, bool)
            # fib_confluence is None without weekly data.
            assert t.fib_confluence is None

    def test_existing_fields_unchanged(self):
        """Adding confluence flags doesn't break existing trigger fields."""
        df = _make_perfect_b1()
        params = B1Params()
        compute_indicators(df, params)
        triggers = detect_triggers(df, "/PA", "daily", params)
        confirmed = [t for t in triggers if t.confirmed]

        assert len(confirmed) >= 1
        t = confirmed[0]
        assert t.entry_price is not None
        assert t.stop_price > 0
        assert t.tp1 > t.stop_price
        assert t.slope_20 > 0
        assert t.bars_of_air >= 6
