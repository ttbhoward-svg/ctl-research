"""Unit tests for COT data loader and feature computation."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.cot_loader import compute_cot_features, load_cot_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cot_raw(symbol: str = "/ES", n_weeks: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic weekly COT data for one symbol."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-05", periods=n_weeks, freq="W-FRI")
    # Simulate a slowly drifting commercial net position.
    net = np.cumsum(rng.normal(0, 500, size=n_weeks)) + 50000
    return pd.DataFrame({
        "publication_date": dates,
        "symbol": symbol,
        "commercial_net": net,
    })


def _make_multi_symbol(symbols=("/ES", "/CL"), n_weeks=60, seed=42):
    """Generate COT data for multiple symbols."""
    frames = []
    for i, sym in enumerate(symbols):
        frames.append(_make_cot_raw(sym, n_weeks, seed=seed + i))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: load_cot_csv
# ---------------------------------------------------------------------------

class TestLoadCotCSV:
    def test_loads_valid_csv(self):
        df = _make_cot_raw()
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        result = load_cot_csv(path)
        assert len(result) == 60
        assert list(result.columns) == ["publication_date", "symbol", "commercial_net"]
        assert result["publication_date"].dtype == "datetime64[ns]"
        path.unlink()

    def test_missing_column_raises(self):
        df = pd.DataFrame({"publication_date": ["2024-01-05"], "commercial_net": [100]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        with pytest.raises(ValueError, match="symbol"):
            load_cot_csv(path)
        path.unlink()

    def test_deduplicates_by_symbol_date(self):
        df = _make_cot_raw(n_weeks=10)
        # Duplicate the first row.
        dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            dup.to_csv(f, index=False)
            path = Path(f.name)
        result = load_cot_csv(path)
        assert len(result) == 10
        path.unlink()

    def test_sorted_by_symbol_then_date(self):
        df = _make_multi_symbol()
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.sample(frac=1, random_state=1).to_csv(f, index=False)
            path = Path(f.name)
        result = load_cot_csv(path)
        # Verify sorted.
        syms = result["symbol"].values
        dates = result["publication_date"].values
        for i in range(1, len(result)):
            if syms[i] == syms[i - 1]:
                assert dates[i] >= dates[i - 1]
        path.unlink()


# ---------------------------------------------------------------------------
# Tests: compute_cot_features
# ---------------------------------------------------------------------------

class TestComputeCotFeatures:
    def test_delta_is_4_week_diff(self):
        df = _make_cot_raw(n_weeks=10)
        result = compute_cot_features(df)
        # First 4 rows should be NaN (insufficient lookback).
        assert result["cot_20d_delta"].iloc[:4].isna().all()
        # Row 4 = commercial_net[4] - commercial_net[0].
        expected = df["commercial_net"].iloc[4] - df["commercial_net"].iloc[0]
        assert abs(result["cot_20d_delta"].iloc[4] - expected) < 1e-6

    def test_zscore_needs_52_weeks(self):
        df = _make_cot_raw(n_weeks=51)
        result = compute_cot_features(df)
        # All z-scores should be NaN (insufficient window).
        assert result["cot_zscore_1y"].isna().all()

    def test_zscore_valid_after_52_weeks(self):
        df = _make_cot_raw(n_weeks=60)
        result = compute_cot_features(df)
        # First valid z-score at row 51 (0-indexed).
        assert result["cot_zscore_1y"].iloc[51:].notna().all()
        # Z-score should have mean ~0 and std ~1 within the window.
        zscores = result["cot_zscore_1y"].dropna()
        assert abs(zscores.mean()) < 2.0  # generous bound for 8 data points

    def test_zscore_zero_std_returns_nan(self):
        """Constant commercial_net for 52 weeks => std=0 => z-score=NaN."""
        df = pd.DataFrame({
            "publication_date": pd.date_range("2024-01-05", periods=55, freq="W-FRI"),
            "symbol": "/ES",
            "commercial_net": 50000.0,  # constant
        })
        result = compute_cot_features(df)
        # All z-scores should be NaN because std is 0.
        assert result["cot_zscore_1y"].isna().all()

    def test_multi_symbol_independent(self):
        """Features are computed per-symbol, not cross-symbol."""
        df = _make_multi_symbol(n_weeks=60)
        result = compute_cot_features(df)
        es = result[result["symbol"] == "/ES"]
        cl = result[result["symbol"] == "/CL"]
        # Delta at row 4 should use that symbol's own data.
        es_expected = es["commercial_net"].iloc[4] - es["commercial_net"].iloc[0]
        assert abs(es["cot_20d_delta"].iloc[4] - es_expected) < 1e-6
        cl_expected = cl["commercial_net"].iloc[4] - cl["commercial_net"].iloc[0]
        assert abs(cl["cot_20d_delta"].iloc[4] - cl_expected) < 1e-6

    def test_output_columns(self):
        df = _make_cot_raw(n_weeks=10)
        result = compute_cot_features(df)
        expected_cols = {"publication_date", "symbol", "commercial_net",
                         "cot_20d_delta", "cot_zscore_1y"}
        assert set(result.columns) == expected_cols
