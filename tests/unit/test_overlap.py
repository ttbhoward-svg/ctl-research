"""Unit tests for overlap window computation and enforcement (H.5)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.overlap import align_to_overlap, compute_overlap_window, validate_min_overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(start: str, periods: int) -> pd.DataFrame:
    """Build a simple OHLCV frame starting at *start* with *periods* bdays."""
    dates = pd.bdate_range(start, periods=periods)
    base = 100.0 + np.arange(periods, dtype=float)
    return pd.DataFrame({
        "Date": dates,
        "Open": base - 0.2,
        "High": base + 1.0,
        "Low": base - 1.0,
        "Close": base,
        "Volume": np.full(periods, 50000.0),
    })


# ---------------------------------------------------------------------------
# Tests: compute_overlap_window
# ---------------------------------------------------------------------------

class TestComputeOverlapWindow:
    def test_full_overlap(self):
        df = _make_ohlcv("2020-01-01", 50)
        start, end, n = compute_overlap_window(df, df.copy())
        assert n == 50
        assert start == df["Date"].min()
        assert end == df["Date"].max()

    def test_partial_overlap(self):
        df_a = _make_ohlcv("2020-01-01", 60)
        df_b = _make_ohlcv("2020-02-01", 60)
        start, end, n = compute_overlap_window(df_a, df_b)
        assert n > 0
        assert n < 60
        assert start >= df_b["Date"].min()
        assert end <= df_a["Date"].max()

    def test_no_overlap(self):
        df_a = _make_ohlcv("2020-01-01", 20)
        df_b = _make_ohlcv("2021-01-01", 20)
        start, end, n = compute_overlap_window(df_a, df_b)
        assert n == 0
        assert pd.isna(start)
        assert pd.isna(end)

    def test_single_bar_overlap(self):
        df_a = _make_ohlcv("2020-01-01", 30)
        last_date = df_a["Date"].iloc[-1]
        df_b = _make_ohlcv(str(last_date.date()), 10)
        start, end, n = compute_overlap_window(df_a, df_b)
        assert n == 1
        assert start == last_date
        assert end == last_date


# ---------------------------------------------------------------------------
# Tests: align_to_overlap
# ---------------------------------------------------------------------------

class TestAlignToOverlap:
    def test_both_trimmed(self):
        df_a = _make_ohlcv("2020-01-01", 60)
        df_b = _make_ohlcv("2020-02-01", 60)
        a_out, b_out = align_to_overlap(df_a, df_b)
        assert len(a_out) > 0
        assert len(b_out) > 0
        # Both should start at or after the later start date.
        assert a_out["Date"].min() >= df_b["Date"].min()
        assert b_out["Date"].min() >= df_b["Date"].min()
        # Both should end at or before the earlier end date.
        assert a_out["Date"].max() <= df_a["Date"].max()
        assert b_out["Date"].max() <= df_a["Date"].max()

    def test_no_overlap_returns_empty(self):
        df_a = _make_ohlcv("2020-01-01", 20)
        df_b = _make_ohlcv("2021-01-01", 20)
        a_out, b_out = align_to_overlap(df_a, df_b)
        assert len(a_out) == 0
        assert len(b_out) == 0

    def test_non_mutating(self):
        df_a = _make_ohlcv("2020-01-01", 60)
        df_b = _make_ohlcv("2020-02-01", 60)
        orig_len_a = len(df_a)
        orig_len_b = len(df_b)
        align_to_overlap(df_a, df_b)
        assert len(df_a) == orig_len_a
        assert len(df_b) == orig_len_b

    def test_preserves_columns(self):
        df_a = _make_ohlcv("2020-01-01", 50)
        df_b = _make_ohlcv("2020-01-01", 50)
        a_out, b_out = align_to_overlap(df_a, df_b)
        assert list(a_out.columns) == list(df_a.columns)
        assert list(b_out.columns) == list(df_b.columns)


# ---------------------------------------------------------------------------
# Tests: validate_min_overlap
# ---------------------------------------------------------------------------

class TestValidateMinOverlap:
    def test_passes_when_sufficient(self):
        # Should not raise.
        validate_min_overlap(50, 20)

    def test_raises_when_insufficient(self):
        with pytest.raises(ValueError, match="Insufficient overlap"):
            validate_min_overlap(5, 20)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestOverlapEdgeCases:
    def test_identical_date_ranges(self):
        df = _make_ohlcv("2020-01-01", 40)
        start, end, n = compute_overlap_window(df, df.copy())
        assert n == 40
        a_out, b_out = align_to_overlap(df, df.copy())
        assert len(a_out) == 40
        assert len(b_out) == 40

    def test_one_empty_frame(self):
        df = _make_ohlcv("2020-01-01", 40)
        empty = df.iloc[0:0].copy()
        start, end, n = compute_overlap_window(df, empty)
        assert n == 0
        assert pd.isna(start)
        a_out, b_out = align_to_overlap(df, empty)
        assert len(a_out) == 0
        assert len(b_out) == 0
