"""Unit tests for VIX regime loader."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.vix_loader import compute_vix_regime, load_vix_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vix(n_days: int = 30, base: float = 18.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily VIX data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-06-01", periods=n_days)
    vix = base + rng.normal(0, 3, size=n_days)
    return pd.DataFrame({"date": dates, "vix_close": vix})


# ---------------------------------------------------------------------------
# Tests: load_vix_csv
# ---------------------------------------------------------------------------

class TestLoadVixCSV:
    def test_loads_valid_csv(self):
        df = _make_vix()
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        result = load_vix_csv(path)
        assert len(result) == 30
        assert "date" in result.columns
        assert "vix_close" in result.columns
        assert result["date"].dtype == "datetime64[ns]"
        path.unlink()

    def test_missing_column_raises(self):
        df = pd.DataFrame({"date": ["2024-06-01"], "something_else": [18.0]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        with pytest.raises(ValueError, match="vix_close"):
            load_vix_csv(path)
        path.unlink()

    def test_deduplicates_by_date(self):
        df = _make_vix(n_days=5)
        dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            dup.to_csv(f, index=False)
            path = Path(f.name)
        result = load_vix_csv(path)
        assert len(result) == 5
        path.unlink()

    def test_optional_vix3m_column(self):
        df = _make_vix(n_days=5)
        df["vix3m_close"] = df["vix_close"] + 2.0
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        result = load_vix_csv(path)
        assert "vix3m_close" in result.columns
        path.unlink()


# ---------------------------------------------------------------------------
# Tests: compute_vix_regime
# ---------------------------------------------------------------------------

class TestComputeVixRegime:
    def test_low_vol_below_20(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"]), "vix_close": [15.0]})
        result = compute_vix_regime(df)
        assert result["vix_regime"].iloc[0] == True  # noqa: E712

    def test_elevated_above_20(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"]), "vix_close": [25.0]})
        result = compute_vix_regime(df)
        assert result["vix_regime"].iloc[0] == False  # noqa: E712

    def test_exactly_20_is_not_low(self):
        """VIX_Regime = True only if STRICTLY < 20."""
        df = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"]), "vix_close": [20.0]})
        result = compute_vix_regime(df)
        assert result["vix_regime"].iloc[0] == False  # noqa: E712

    def test_nan_vix_regime_is_none(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"]), "vix_close": [np.nan]})
        result = compute_vix_regime(df)
        assert result["vix_regime"].iloc[0] is None

    def test_mixed_values(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
            "vix_close": [15.0, 25.0, np.nan],
        })
        result = compute_vix_regime(df)
        assert result["vix_regime"].iloc[0] == True   # noqa: E712
        assert result["vix_regime"].iloc[1] == False   # noqa: E712
        assert result["vix_regime"].iloc[2] is None
