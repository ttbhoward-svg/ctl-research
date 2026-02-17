"""Tests for the data loader module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.data_loader import OHLCV_COLS, load_tradestation_csv, save_processed, load_processed


def _write_csv(path: Path, n: int = 20) -> pd.DataFrame:
    """Write a synthetic TradeStation-style CSV and return the expected DataFrame."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(99)
    close = 100.0 + rng.standard_normal(n).cumsum()
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = low + rng.uniform(0, 1, n) * (high - low)
    vol = rng.integers(1000, 50000, n).astype(float)
    df = pd.DataFrame({
        "Date": dates.strftime("%m/%d/%Y"),  # TS date format
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })
    df.to_csv(path, index=False)
    return df


def test_load_tradestation_csv():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "PA_daily.csv"
        _write_csv(p)
        result = load_tradestation_csv(p, "daily")
        assert list(result.columns) == OHLCV_COLS
        assert len(result) == 20
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])
        assert result["Date"].is_monotonic_increasing


def test_load_handles_case_insensitive_columns():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        df = pd.DataFrame({
            "DATE": pd.bdate_range("2024-01-02", periods=5).strftime("%Y-%m-%d"),
            "OPEN": [1.0] * 5,
            "HIGH": [2.0] * 5,
            "LOW": [0.5] * 5,
            "CLOSE": [1.5] * 5,
            "VOL": [100.0] * 5,
        })
        df.to_csv(p, index=False)
        result = load_tradestation_csv(p, "daily")
        assert list(result.columns) == OHLCV_COLS
        assert len(result) == 5


def test_save_and_load_processed():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        dates = pd.bdate_range("2024-01-02", periods=10)
        df = pd.DataFrame({
            "Date": dates,
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
            "Close": range(10),
            "Volume": range(10),
        })
        data = {"/PA": {"daily": df}}
        written = save_processed(data, out_dir)
        assert len(written) == 1
        assert written[0].name == "PA_daily.parquet"

        loaded = load_processed("/PA", "daily", out_dir)
        assert len(loaded) == 10
        assert list(loaded.columns) == OHLCV_COLS


def test_removes_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        dates = ["2024-01-02"] * 3 + ["2024-01-03", "2024-01-04"]
        df = pd.DataFrame({
            "Date": dates,
            "Open": [1.0] * 5,
            "High": [2.0] * 5,
            "Low": [0.5] * 5,
            "Close": [1.5] * 5,
            "Volume": [100.0] * 5,
        })
        df.to_csv(p, index=False)
        result = load_tradestation_csv(p, "daily")
        assert len(result) == 3  # duplicates removed
