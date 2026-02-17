"""Tests for the data sanitiser module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.data_sanitizer import Issue, SanitiserReport, sanitise_dataframe


def _make_df(n: int = 10, start: str = "2024-01-02") -> pd.DataFrame:
    """Create a clean OHLCV DataFrame for testing."""
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = low + rng.uniform(0, 1, n) * (high - low)
    vol = rng.integers(1000, 50000, n).astype(float)
    return pd.DataFrame({
        "Date": dates,
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


def test_clean_data_produces_no_errors():
    df = _make_df()
    issues = sanitise_dataframe(df, "TEST", "daily")
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 0


def test_detects_high_lt_low():
    df = _make_df()
    # Corrupt one row: swap High and Low.
    df.at[3, "High"], df.at[3, "Low"] = df.at[3, "Low"], df.at[3, "High"]
    issues = sanitise_dataframe(df, "TEST", "daily")
    hl_errors = [i for i in issues if i.check == "high_lt_low"]
    assert len(hl_errors) >= 1


def test_detects_duplicate_dates():
    df = _make_df()
    df.at[5, "Date"] = df.at[4, "Date"]
    issues = sanitise_dataframe(df, "TEST", "daily")
    dupe_issues = [i for i in issues if i.check == "duplicate_dates"]
    assert len(dupe_issues) >= 1


def test_detects_zero_volume():
    df = _make_df()
    df.at[2, "Volume"] = 0.0
    df.at[7, "Volume"] = -1.0
    issues = sanitise_dataframe(df, "TEST", "daily")
    vol_issues = [i for i in issues if i.check == "zero_volume"]
    assert len(vol_issues) >= 1


def test_detects_price_jump():
    df = _make_df()
    # Inject a 20% jump.
    df.at[5, "Close"] = df.at[4, "Close"] * 1.25
    df.at[5, "High"] = max(df.at[5, "High"], df.at[5, "Close"])
    issues = sanitise_dataframe(df, "TEST", "daily")
    jump_issues = [i for i in issues if i.check == "price_jump"]
    assert len(jump_issues) >= 1


def test_detects_nan_values():
    df = _make_df()
    df.at[4, "Close"] = np.nan
    issues = sanitise_dataframe(df, "TEST", "daily")
    nan_issues = [i for i in issues if i.check == "nan_close"]
    assert len(nan_issues) >= 1


def test_detects_date_gap():
    df = _make_df(20)
    # Create a 10-day gap by removing rows.
    df = df.drop(index=[5, 6, 7, 8, 9]).reset_index(drop=True)
    issues = sanitise_dataframe(df, "TEST", "daily")
    gap_issues = [i for i in issues if i.check == "date_gap"]
    assert len(gap_issues) >= 1


def test_empty_dataframe():
    df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    issues = sanitise_dataframe(df, "TEST", "daily")
    assert any(i.check == "empty" for i in issues)


def test_report_summary():
    report = SanitiserReport(symbol="TEST")
    report.issues.append(Issue("TEST", "daily", "error", "test", "test issue"))
    report.issues.append(Issue("TEST", "daily", "warning", "test2", "test warning"))
    assert report.error_count == 1
    assert report.warning_count == 1
    assert not report.is_clean
    assert "1 errors" in report.summary()
