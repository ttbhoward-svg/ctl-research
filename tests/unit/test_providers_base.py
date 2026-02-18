"""Unit tests for data provider abstraction layer (Data Cutover Task A)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.providers.base import (
    CANONICAL_COLUMNS,
    CLOSE_TYPES,
    ROLL_METHODS,
    SESSION_TYPES,
    DataProvider,
    ProviderMeta,
    normalize_to_canonical,
    validate_canonical,
)
from ctl.providers.databento_provider import DatabentoProvider
from ctl.providers.norgate_provider import NorgateProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_ohlcv(n: int = 20) -> pd.DataFrame:
    """Minimal raw OHLCV DataFrame (pre-normalisation)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100.0 + np.arange(n, dtype=float) * 0.5
    return pd.DataFrame({
        "Date": dates,
        "Open": base,
        "High": base + 2.0,
        "Low": base - 1.0,
        "Close": base + 1.0,
        "Volume": np.full(n, 50000.0),
    })


def _make_canonical(n: int = 20, **overrides) -> pd.DataFrame:
    """Canonical-schema DataFrame for validation tests."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    base = 100.0 + np.arange(n, dtype=float) * 0.5
    df = pd.DataFrame({
        "timestamp": dates,
        "Open": base,
        "High": base + 2.0,
        "Low": base - 1.0,
        "Close": base + 1.0,
        "Volume": np.full(n, 50000.0),
        "symbol": "/ES",
        "timeframe": "1D",
        "provider": "test",
        "session_type": "electronic",
        "roll_method": "back_adjusted",
        "close_type": "settlement",
    })
    for k, v in overrides.items():
        df[k] = v
    return df


class _StubProvider(DataProvider):
    """Concrete provider for testing the abstract interface."""

    def __init__(self, data: pd.DataFrame):
        self._data = data

    @property
    def name(self) -> str:
        return "stub"

    def get_ohlcv(self, symbol, timeframe, start, end):
        return self._data.copy()


# ---------------------------------------------------------------------------
# Tests: ProviderMeta
# ---------------------------------------------------------------------------

class TestProviderMeta:
    def test_valid_meta(self):
        meta = ProviderMeta(
            provider="test", session_type="electronic",
            roll_method="back_adjusted", close_type="settlement",
        )
        assert meta.validate() == []

    def test_invalid_session_type(self):
        meta = ProviderMeta(provider="x", session_type="invalid")
        errors = meta.validate()
        assert len(errors) == 1
        assert "session_type" in errors[0]

    def test_invalid_roll_method(self):
        meta = ProviderMeta(provider="x", roll_method="invalid")
        errors = meta.validate()
        assert any("roll_method" in e for e in errors)

    def test_invalid_close_type(self):
        meta = ProviderMeta(provider="x", close_type="invalid")
        errors = meta.validate()
        assert any("close_type" in e for e in errors)

    def test_all_session_types_valid(self):
        for st in SESSION_TYPES:
            meta = ProviderMeta(provider="x", session_type=st)
            errors = [e for e in meta.validate() if "session_type" in e]
            assert errors == []

    def test_all_roll_methods_valid(self):
        for rm in ROLL_METHODS:
            meta = ProviderMeta(provider="x", roll_method=rm)
            errors = [e for e in meta.validate() if "roll_method" in e]
            assert errors == []

    def test_all_close_types_valid(self):
        for ct in CLOSE_TYPES:
            meta = ProviderMeta(provider="x", close_type=ct)
            errors = [e for e in meta.validate() if "close_type" in e]
            assert errors == []


# ---------------------------------------------------------------------------
# Tests: validate_canonical
# ---------------------------------------------------------------------------

class TestValidateCanonical:
    def test_valid_dataframe(self):
        df = _make_canonical()
        assert validate_canonical(df) == []

    def test_missing_columns(self):
        df = _make_canonical().drop(columns=["Open", "Volume"])
        errors = validate_canonical(df)
        assert any("Missing" in e for e in errors)

    def test_naive_timestamp_fails(self):
        df = _make_canonical()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        errors = validate_canonical(df)
        assert any("UTC" in e for e in errors)

    def test_non_datetime_timestamp_fails(self):
        df = _make_canonical()
        df["timestamp"] = "2024-01-01"
        errors = validate_canonical(df)
        assert any("timestamp" in e for e in errors)

    def test_non_float_price_fails(self):
        df = _make_canonical()
        df["Close"] = df["Close"].astype(int)
        errors = validate_canonical(df)
        assert any("Close" in e for e in errors)

    def test_high_below_low_fails(self):
        df = _make_canonical()
        df.loc[0, "High"] = df.loc[0, "Low"] - 1.0
        errors = validate_canonical(df)
        assert any("High < Low" in e for e in errors)

    def test_high_below_open_fails(self):
        df = _make_canonical()
        df.loc[0, "High"] = df.loc[0, "Open"] - 0.01
        errors = validate_canonical(df)
        assert any("High < Open" in e for e in errors)

    def test_low_above_close_fails(self):
        df = _make_canonical()
        df.loc[0, "Low"] = df.loc[0, "Close"] + 0.01
        errors = validate_canonical(df)
        assert any("Low > Close" in e for e in errors)

    def test_invalid_session_type_fails(self):
        df = _make_canonical(session_type="invalid")
        errors = validate_canonical(df)
        assert any("session_type" in e for e in errors)

    def test_invalid_roll_method_fails(self):
        df = _make_canonical(roll_method="invalid")
        errors = validate_canonical(df)
        assert any("roll_method" in e for e in errors)

    def test_invalid_close_type_fails(self):
        df = _make_canonical(close_type="invalid")
        errors = validate_canonical(df)
        assert any("close_type" in e for e in errors)

    def test_empty_dataframe_valid(self):
        df = _make_canonical(n=0)
        errors = validate_canonical(df)
        # Empty is valid (no bars to check).
        non_ts_errors = [e for e in errors if "timestamp" not in e]
        assert non_ts_errors == []


# ---------------------------------------------------------------------------
# Tests: normalize_to_canonical
# ---------------------------------------------------------------------------

class TestNormalizeToCanonical:
    def test_output_has_all_columns(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/ES", "1D")
        for col in CANONICAL_COLUMNS:
            assert col in df.columns

    def test_output_passes_validation(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/ES", "1D")
        assert validate_canonical(df) == []

    def test_timestamp_is_utc(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/ES", "1D")
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_price_columns_float64(self):
        raw = _make_raw_ohlcv()
        raw["Close"] = raw["Close"].astype(int)  # intentionally wrong
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/ES", "1D")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert df[col].dtype == np.float64

    def test_metadata_stamped(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(
            provider="databento", session_type="electronic",
            roll_method="back_adjusted", close_type="settlement",
        )
        df = normalize_to_canonical(raw, meta, "/GC", "1W")
        assert (df["provider"] == "databento").all()
        assert (df["symbol"] == "/GC").all()
        assert (df["timeframe"] == "1W").all()
        assert (df["session_type"] == "electronic").all()
        assert (df["roll_method"] == "back_adjusted").all()
        assert (df["close_type"] == "settlement").all()

    def test_preserves_row_count(self):
        raw = _make_raw_ohlcv(n=15)
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/CL", "1D")
        assert len(df) == 15

    def test_missing_price_column_raises(self):
        raw = _make_raw_ohlcv().drop(columns=["Close"])
        meta = ProviderMeta(provider="test")
        with pytest.raises(ValueError, match="Missing price columns"):
            normalize_to_canonical(raw, meta, "/ES", "1D")

    def test_missing_timestamp_column_raises(self):
        raw = _make_raw_ohlcv().rename(columns={"Date": "Dt"})
        meta = ProviderMeta(provider="test")
        with pytest.raises(ValueError, match="Timestamp column"):
            normalize_to_canonical(raw, meta, "/ES", "1D")

    def test_invalid_meta_raises(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(provider="test", session_type="bad")
        with pytest.raises(ValueError, match="Invalid provider metadata"):
            normalize_to_canonical(raw, meta, "/ES", "1D")

    def test_already_tz_aware_converted_to_utc(self):
        raw = _make_raw_ohlcv()
        raw["Date"] = raw["Date"].dt.tz_localize("US/Eastern")
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/ES", "1D")
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_column_order_matches_canonical(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(provider="test")
        df = normalize_to_canonical(raw, meta, "/ES", "1D")
        assert list(df.columns) == CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# Tests: DataProvider interface
# ---------------------------------------------------------------------------

class TestDataProviderInterface:
    def test_stub_provider_returns_data(self):
        canonical = _make_canonical(10)
        provider = _StubProvider(canonical)
        result = provider.get_ohlcv("/ES", "1D", "2024-01-01", "2024-12-31")
        assert len(result) == 10
        assert provider.name == "stub"

    def test_get_ohlcv_multi_concatenates(self):
        canonical = _make_canonical(5)
        provider = _StubProvider(canonical)
        result = provider.get_ohlcv_multi(
            ["/ES", "/GC"], "1D", "2024-01-01", "2024-12-31",
        )
        assert len(result) == 10  # 5 + 5

    def test_get_ohlcv_multi_empty_symbols(self):
        canonical = _make_canonical(5)
        provider = _StubProvider(canonical)
        result = provider.get_ohlcv_multi([], "1D", "2024-01-01", "2024-12-31")
        assert len(result) == 0
        assert list(result.columns) == CANONICAL_COLUMNS

    def test_abstract_methods_enforced(self):
        with pytest.raises(TypeError):
            DataProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Tests: provider stubs
# ---------------------------------------------------------------------------

class TestProviderStubs:
    def test_databento_name(self):
        p = DatabentoProvider()
        assert p.name == "databento"

    def test_databento_raises_not_implemented(self):
        p = DatabentoProvider()
        with pytest.raises(NotImplementedError, match="DatabentoProvider"):
            p.get_ohlcv("/ES", "1D", "2024-01-01", "2024-12-31")

    def test_norgate_name(self):
        p = NorgateProvider()
        assert p.name == "norgate"

    def test_norgate_raises_not_implemented(self):
        p = NorgateProvider()
        with pytest.raises(NotImplementedError, match="NorgateProvider"):
            p.get_ohlcv("/ES", "1D", "2024-01-01", "2024-12-31")

    def test_databento_inherits_provider(self):
        assert issubclass(DatabentoProvider, DataProvider)

    def test_norgate_inherits_provider(self):
        assert issubclass(NorgateProvider, DataProvider)


# ---------------------------------------------------------------------------
# Tests: deterministic normalisation
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_input_same_output(self):
        raw = _make_raw_ohlcv()
        meta = ProviderMeta(provider="test")
        df1 = normalize_to_canonical(raw, meta, "/ES", "1D")
        df2 = normalize_to_canonical(raw, meta, "/ES", "1D")
        pd.testing.assert_frame_equal(df1, df2)
