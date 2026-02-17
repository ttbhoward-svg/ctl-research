"""Tests for the indicator module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.indicators import atr, ema, slope_pct, sma, williams_r


def _close_series(n: int = 50, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 + rng.standard_normal(n).cumsum())


# --- EMA -------------------------------------------------------------------

def test_ema_seed_equals_first_value():
    s = _close_series()
    result = ema(s, 10)
    assert result.iloc[0] == s.iloc[0]


def test_ema_length_matches_input():
    s = _close_series()
    assert len(ema(s, 10)) == len(s)


def test_ema_matches_manual_calculation():
    """Verify against the recursive formula from Spec §1.3."""
    s = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0])
    mult = 2.0 / (3 + 1)  # period=3
    expected = [10.0]
    for i in range(1, len(s)):
        expected.append(s.iloc[i] * mult + expected[-1] * (1 - mult))
    result = ema(s, 3)
    np.testing.assert_allclose(result.values, expected, atol=1e-10)


# --- SMA -------------------------------------------------------------------

def test_sma_first_values_nan():
    s = _close_series(10)
    result = sma(s, 5)
    assert result.iloc[:4].isna().all()
    assert not np.isnan(result.iloc[4])


def test_sma_value_correct():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert result.iloc[2] == 2.0  # (1+2+3)/3
    assert result.iloc[4] == 4.0  # (3+4+5)/3


# --- ATR -------------------------------------------------------------------

def test_atr_positive():
    rng = np.random.default_rng(7)
    n = 50
    close = pd.Series(100.0 + rng.standard_normal(n).cumsum())
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    result = atr(high, low, close, 14)
    # After warmup, ATR should be positive.
    assert (result.iloc[14:] > 0).all()


def test_atr_length():
    s = pd.Series([100.0] * 20)
    h = s + 1
    lo = s - 1
    result = atr(h, lo, s, 14)
    assert len(result) == 20


# --- Williams %R -----------------------------------------------------------

def test_williams_r_range():
    """Williams %R should be between -100 and 0."""
    rng = np.random.default_rng(3)
    n = 50
    close = pd.Series(100.0 + rng.standard_normal(n).cumsum())
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    result = williams_r(high, low, close, 10)
    valid = result.dropna()
    assert (valid >= -100).all()
    assert (valid <= 0).all()


def test_williams_r_at_high():
    """If close == highest high over period, %R should be 0."""
    high = pd.Series([10.0] * 10 + [20.0])
    low = pd.Series([5.0] * 10 + [15.0])
    close = pd.Series([8.0] * 10 + [20.0])
    result = williams_r(high, low, close, 10)
    # Last bar: close=20=highest high -> %R = 0
    assert result.iloc[-1] == 0.0


# --- Slope -----------------------------------------------------------------

def test_slope_pct_positive_trend():
    # Steadily rising EMA: slope should be positive.
    ema_series = pd.Series(np.linspace(100, 120, 30))
    result = slope_pct(ema_series, 20)
    assert result.iloc[-1] > 0


def test_slope_pct_nan_at_start():
    ema_series = pd.Series(np.linspace(100, 120, 30))
    result = slope_pct(ema_series, 20)
    assert np.isnan(result.iloc[0])
