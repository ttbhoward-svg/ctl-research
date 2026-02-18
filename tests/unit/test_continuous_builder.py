"""Unit tests for continuous futures series builder (Data Cutover Task F)."""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.continuous_builder import (
    ADJUSTMENT_CONVENTIONS,
    CONTRACT_RE,
    MONTH_CODES,
    YEAR_MAP,
    CalibrationResult,
    ConventionScore,
    ContractSpec,
    ContinuousResult,
    RollEvent,
    _build_roll_log,
    apply_panama_adjustment,
    build_all,
    build_continuous,
    calibrate_convention,
    detect_rolls,
    load_contract_data,
    parse_contracts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract_df(
    symbol: str,
    start: str,
    n_bars: int,
    base_price: float = 100.0,
    base_volume: float = 10000.0,
    trend: float = 0.1,
    volume_trend: float = 0.0,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for one contract.

    Parameters
    ----------
    symbol : str
        Contract symbol (e.g. ``"ESH5"``).
    start : str
        Start date string.
    n_bars : int
        Number of daily bars.
    base_price : float
        Starting close price.
    base_volume : float
        Starting volume.
    trend : float
        Daily price increment.
    volume_trend : float
        Daily volume increment (can be negative for declining volume).
    """
    dates = pd.bdate_range(start, periods=n_bars)
    close = base_price + np.arange(n_bars) * trend
    volume = np.maximum(base_volume + np.arange(n_bars) * volume_trend, 0)
    return pd.DataFrame({
        "date": [d.date() for d in dates],
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    })


def _two_contract_setup(
    overlap_start: str = "2024-06-01",
    n_overlap: int = 20,
    front_pre: int = 40,
    back_post: int = 40,
    front_base: float = 100.0,
    back_base: float = 105.0,
    front_vol_start: float = 50000.0,
    back_vol_start: float = 10000.0,
    front_vol_trend: float = -1500.0,
    back_vol_trend: float = 2000.0,
):
    """Build two overlapping contracts where the back contract's volume
    overtakes the front during the overlap window.

    Returns (contracts dict, contract_order list).
    """
    # Front contract: starts earlier, volume declines.
    front_start = pd.Timestamp(overlap_start) - pd.tseries.offsets.BDay(front_pre)
    front_n = front_pre + n_overlap
    front_df = _make_contract_df(
        "ESM4",
        str(front_start.date()),
        front_n,
        base_price=front_base,
        base_volume=front_vol_start,
        volume_trend=front_vol_trend,
    )

    # Back contract: starts at overlap, volume rises.
    back_n = n_overlap + back_post
    back_df = _make_contract_df(
        "ESU4",
        overlap_start,
        back_n,
        base_price=back_base,
        base_volume=back_vol_start,
        volume_trend=back_vol_trend,
    )

    contracts = {"ESM4": front_df, "ESU4": back_df}
    order = parse_contracts(["ESM4", "ESU4"])
    return contracts, order


# ---------------------------------------------------------------------------
# Tests: Contract parsing
# ---------------------------------------------------------------------------

class TestContractParsing:
    def test_parse_valid_symbol(self):
        spec = ContractSpec.from_symbol("ESH5")
        assert spec.root == "ES"
        assert spec.month_code == "H"
        assert spec.month == 3
        assert spec.year == 2025
        assert spec.symbol == "ESH5"

    def test_parse_single_letter_root(self):
        # Some commodities have single-letter roots (hypothetical).
        # CONTRACT_RE allows 1-3 uppercase letters.
        spec = ContractSpec.from_symbol("ZH5")
        assert spec.root == "Z"
        assert spec.month == 3

    def test_parse_three_letter_root(self):
        spec = ContractSpec.from_symbol("PAF0")
        assert spec.root == "PA"
        assert spec.month_code == "F"
        assert spec.month == 1
        assert spec.year == 2020

    def test_all_month_codes_valid(self):
        for code, month_num in MONTH_CODES.items():
            spec = ContractSpec.from_symbol(f"ES{code}5")
            assert spec.month == month_num
            assert spec.month_code == code

    def test_all_year_digits_valid(self):
        for digit, year in YEAR_MAP.items():
            spec = ContractSpec.from_symbol(f"ES{digit}".replace(digit, f"H{digit}"))
            assert spec.year == year

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("INVALID")

    def test_lowercase_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("esh5")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("")

    def test_numeric_root_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("12H5")

    def test_sort_key_calculation(self):
        spec = ContractSpec.from_symbol("ESH5")
        assert spec.sort_key == 2025 * 12 + 3


class TestContractOrdering:
    def test_sorted_by_expiration(self):
        specs = parse_contracts(["ESZ4", "ESH5", "ESM4", "ESU4"])
        symbols = [s.symbol for s in specs]
        assert symbols == ["ESM4", "ESU4", "ESZ4", "ESH5"]

    def test_cross_year_ordering(self):
        specs = parse_contracts(["ESZ4", "ESH5"])
        assert specs[0].symbol == "ESZ4"  # Dec 2024 before Mar 2025
        assert specs[1].symbol == "ESH5"

    def test_same_month_different_years(self):
        specs = parse_contracts(["ESH4", "ESH5"])
        assert specs[0].symbol == "ESH4"  # 2024 before 2025
        assert specs[1].symbol == "ESH5"

    def test_single_contract(self):
        specs = parse_contracts(["ESH5"])
        assert len(specs) == 1
        assert specs[0].symbol == "ESH5"

    def test_empty_list(self):
        specs = parse_contracts([])
        assert specs == []

    def test_different_roots_sort_by_date(self):
        specs = parse_contracts(["CLH5", "ESM4"])
        # ESM4 = Jun 2024 (sort_key=24300), CLH5 = Mar 2025 (sort_key=24303)
        assert specs[0].symbol == "ESM4"
        assert specs[1].symbol == "CLH5"


# ---------------------------------------------------------------------------
# Tests: Contract regex
# ---------------------------------------------------------------------------

class TestContractRegex:
    def test_valid_patterns(self):
        assert CONTRACT_RE.match("ESH5") is not None
        assert CONTRACT_RE.match("CLZ0") is not None
        assert CONTRACT_RE.match("PAF9") is not None

    def test_invalid_patterns(self):
        assert CONTRACT_RE.match("") is None
        assert CONTRACT_RE.match("ES") is None
        assert CONTRACT_RE.match("ES5") is None
        assert CONTRACT_RE.match("esh5") is None
        assert CONTRACT_RE.match("ES.FUT") is None
        assert CONTRACT_RE.match("TOOLONGH5") is None  # >3 letter root


# ---------------------------------------------------------------------------
# Tests: Roll detection
# ---------------------------------------------------------------------------

class TestRollDetection:
    def test_no_contracts_returns_empty(self):
        rolls, active = detect_rolls({}, [])
        assert rolls == []
        assert active.empty

    def test_single_contract_no_rolls(self):
        df = _make_contract_df("ESH5", "2024-01-02", 50)
        contracts = {"ESH5": df}
        order = parse_contracts(["ESH5"])
        rolls, active = detect_rolls(contracts, order)
        assert len(rolls) == 0
        assert len(active) == 50

    def test_volume_crossover_triggers_roll(self):
        contracts, order = _two_contract_setup(
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order, consecutive_days=2)
        assert len(rolls) == 1
        assert rolls[0].from_contract == "ESM4"
        assert rolls[0].to_contract == "ESU4"

    def test_roll_date_after_consecutive_days(self):
        # With consecutive_days=2, roll fires on the second day of crossover.
        contracts, order = _two_contract_setup(
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, _ = detect_rolls(contracts, order, consecutive_days=2)
        assert len(rolls) == 1
        # The roll date should be a valid business day.
        assert rolls[0].date is not None

    def test_no_roll_if_volume_never_crosses(self):
        # Both contracts cover the same dates; front always has higher volume.
        front_df = _make_contract_df("ESM4", "2024-06-01", 20,
                                     base_volume=100000, volume_trend=0)
        back_df = _make_contract_df("ESU4", "2024-06-01", 20,
                                    base_volume=1000, volume_trend=0)
        contracts = {"ESM4": front_df, "ESU4": back_df}
        order = parse_contracts(["ESM4", "ESU4"])
        rolls, active = detect_rolls(contracts, order)
        assert len(rolls) == 0

    def test_single_day_crossover_insufficient(self):
        """With consecutive_days=2, a single day of volume crossover
        should NOT trigger a roll."""
        # Build two contracts where volume crosses for exactly 1 day.
        front_df = _make_contract_df("ESM4", "2024-06-01", 10,
                                     base_volume=50000, volume_trend=0)
        back_df = _make_contract_df("ESU4", "2024-06-01", 10,
                                    base_volume=10000, volume_trend=0)
        # Make back volume > front on day 5 only.
        back_df.loc[4, "volume"] = 60000
        contracts = {"ESM4": front_df, "ESU4": back_df}
        order = parse_contracts(["ESM4", "ESU4"])
        rolls, _ = detect_rolls(contracts, order, consecutive_days=2)
        assert len(rolls) == 0

    def test_roll_adjustment_sign(self):
        """Adjustment = to_close - from_close."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=105.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, _ = detect_rolls(contracts, order)
        if rolls:
            r = rolls[0]
            assert abs(r.adjustment - (r.to_close - r.from_close)) < 1e-9

    def test_active_series_contract_column(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        assert "date" in active.columns
        assert "contract" in active.columns
        # All contracts should be from our two symbols.
        assert set(active["contract"].unique()).issubset({"ESM4", "ESU4"})

    def test_active_series_switches_after_roll(self):
        contracts, order = _two_contract_setup(
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        if rolls:
            roll_date = rolls[0].date
            before = active[active["date"] < roll_date]
            after = active[active["date"] >= roll_date]
            # Before roll: front contract.
            if not before.empty:
                assert (before["contract"] == "ESM4").all()
            # After roll: back contract.
            if not after.empty:
                assert (after["contract"] == "ESU4").all()

    def test_three_contract_chain(self):
        """Three contracts should produce up to 2 rolls."""
        dates1 = _make_contract_df("ESH4", "2024-01-02", 60,
                                   base_volume=80000, volume_trend=-2000)
        dates2 = _make_contract_df("ESM4", "2024-02-15", 60,
                                   base_volume=5000, volume_trend=3000)
        dates3 = _make_contract_df("ESU4", "2024-04-01", 60,
                                   base_volume=1000, volume_trend=2000)
        # Ensure second roll: make ESU4 volume eventually overtake ESM4.
        for i in range(len(dates3)):
            dates3.loc[i, "volume"] = 1000 + i * 4000

        contracts = {"ESH4": dates1, "ESM4": dates2, "ESU4": dates3}
        order = parse_contracts(["ESH4", "ESM4", "ESU4"])
        rolls, active = detect_rolls(contracts, order)
        # Should have at least 1 roll (possibly 2).
        assert len(rolls) >= 1
        assert rolls[0].from_contract == "ESH4"
        assert rolls[0].to_contract == "ESM4"


# ---------------------------------------------------------------------------
# Tests: Panama back-adjustment
# ---------------------------------------------------------------------------

class TestPanamaAdjustment:
    def test_no_rolls_no_adjustment(self):
        df = _make_contract_df("ESH5", "2024-01-02", 50)
        contracts = {"ESH5": df}
        active = pd.DataFrame({
            "date": df["date"],
            "contract": "ESH5",
        })
        result = apply_panama_adjustment(contracts, active, [])
        assert (result["adjustment"] == 0.0).all()
        # Prices should match original.
        np.testing.assert_allclose(result["Close"].values, df["close"].values, atol=1e-6)

    def test_single_roll_adjusts_earlier_segment(self):
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=105.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        assert len(rolls) == 1

        result = apply_panama_adjustment(contracts, active, rolls)

        # Post-roll bars should have adjustment = 0.
        roll_date = rolls[0].date
        post_roll = result[result["Date"].dt.date >= roll_date]
        if not post_roll.empty:
            assert (post_roll["adjustment"] == 0.0).all()

        # Pre-roll bars should have non-zero adjustment.
        pre_roll = result[result["Date"].dt.date < roll_date]
        if not pre_roll.empty:
            assert (pre_roll["adjustment"] != 0.0).all()

    def test_adjustment_makes_close_continuous(self):
        """At the roll boundary, the adjusted close should not have a gap
        (the gap equals the adjustment)."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=105.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)

        if len(rolls) == 1:
            roll_date = rolls[0].date
            idx = result.index[result["Date"].dt.date == roll_date]
            if len(idx) > 0:
                roll_idx = idx[0]
                if roll_idx > 0:
                    # Check that the gap at the roll point is small.
                    close_before = result.loc[roll_idx - 1, "Close"]
                    close_at = result.loc[roll_idx, "Close"]
                    # The daily change should be reasonable (no giant gap).
                    daily_change = abs(close_at - close_before)
                    # Without adjustment, gap would be ~5.0 (back_base - front_base).
                    # With adjustment, the gap should be roughly the daily trend.
                    assert daily_change < 3.0  # generous threshold

    def test_volume_not_adjusted(self):
        contracts, order = _two_contract_setup(
            front_vol_start=80000,
            back_vol_start=50000,
            front_vol_trend=0,
            back_vol_trend=0,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        # Volume should always be non-negative integer.
        assert (result["Volume"] >= 0).all()

    def test_empty_active_returns_empty(self):
        result = apply_panama_adjustment({}, pd.DataFrame(columns=["date", "contract"]), [])
        assert result.empty
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "contract", "adjustment"}
        assert set(result.columns) == expected_cols

    def test_ohlc_all_shifted_equally(self):
        """Open, High, Low, Close should all receive the same adjustment."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=110.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)

        pre_roll = result[result["adjustment"] != 0.0]
        if not pre_roll.empty:
            row = pre_roll.iloc[0]
            adj = row["adjustment"]
            sym = row["contract"]
            d = row["Date"].date()
            raw = contracts[sym]
            raw_bar = raw[raw["date"] == d].iloc[0]
            # Each OHLC price = raw - adj
            assert abs(row["Open"] - (raw_bar["open"] - adj)) < 1e-6
            assert abs(row["High"] - (raw_bar["high"] - adj)) < 1e-6
            assert abs(row["Low"] - (raw_bar["low"] - adj)) < 1e-6
            assert abs(row["Close"] - (raw_bar["close"] - adj)) < 1e-6


# ---------------------------------------------------------------------------
# Tests: Output schema and quality
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_continuous_columns(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        expected = {"Date", "Open", "High", "Low", "Close", "Volume", "contract", "adjustment"}
        assert set(result.columns) == expected

    def test_dates_monotonically_increasing(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        dates = result["Date"].values
        assert (np.diff(dates) > np.timedelta64(0)).all()

    def test_no_duplicate_dates(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        assert result["Date"].nunique() == len(result)

    def test_date_is_datetime_type(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_volume_is_integer(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        # Volume values should be whole numbers.
        assert (result["Volume"] == result["Volume"].astype(int)).all()

    def test_high_ge_low(self):
        """Adjusted High should still be >= adjusted Low."""
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        assert (result["High"] >= result["Low"]).all()


# ---------------------------------------------------------------------------
# Tests: RollEvent dataclass
# ---------------------------------------------------------------------------

class TestRollEvent:
    def test_roll_event_fields(self):
        r = RollEvent(
            date=datetime.date(2024, 6, 15),
            from_contract="ESM4",
            to_contract="ESU4",
            from_close=5400.0,
            to_close=5410.0,
            adjustment=10.0,
        )
        assert r.adjustment == 10.0
        assert r.cumulative_adjustment == 0.0  # default

    def test_roll_event_cumulative(self):
        r = RollEvent(
            date=datetime.date(2024, 6, 15),
            from_contract="ESM4",
            to_contract="ESU4",
            from_close=5400.0,
            to_close=5410.0,
            adjustment=10.0,
            cumulative_adjustment=25.0,
        )
        assert r.cumulative_adjustment == 25.0


# ---------------------------------------------------------------------------
# Tests: ContinuousResult
# ---------------------------------------------------------------------------

class TestContinuousResult:
    def test_empty_result(self):
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=pd.DataFrame(),
        )
        assert result.root == "ES"
        assert result.n_contracts == 0
        assert result.n_rolls == 0


# ---------------------------------------------------------------------------
# Tests: build_continuous (integration with synthetic data on disk)
# ---------------------------------------------------------------------------

class TestBuildContinuousSynthetic:
    """Test build_continuous using synthetic CSV files written to tmp_path."""

    def _write_contract_csv(self, tmp_dir: Path, symbol: str, df: pd.DataFrame):
        """Write a synthetic contract DataFrame as a CSV (not zst)."""
        # build_continuous expects .csv.zst — we'll use plain .csv for tests.
        # Since load_contract_data globs for *.csv.zst, we need to adjust.
        # Actually let's write .csv.zst but uncompressed (pandas handles it).
        csv_path = tmp_dir / f"test.{symbol}.csv.zst"
        out = pd.DataFrame({
            "ts_event": pd.to_datetime(df["date"]).astype(str),
            "rtype": 0,
            "publisher_id": 0,
            "instrument_id": 0,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "symbol": symbol,
        })
        out.to_csv(csv_path, index=False)

    def test_build_from_synthetic_files(self, tmp_path):
        """Build continuous series from synthetic CSV files."""
        front = _make_contract_df("ESM4", "2024-03-01", 60,
                                  base_price=5300, base_volume=80000,
                                  volume_trend=-2000)
        back = _make_contract_df("ESU4", "2024-04-15", 60,
                                 base_price=5310, base_volume=5000,
                                 volume_trend=3000)
        self._write_contract_csv(tmp_path, "ESM4", front)
        self._write_contract_csv(tmp_path, "ESU4", back)

        result = build_continuous("ES", tmp_path)
        assert result.root == "ES"
        assert result.n_contracts == 2
        assert not result.continuous.empty
        assert result.n_rolls >= 0

    def test_no_files_returns_empty(self, tmp_path):
        result = build_continuous("ES", tmp_path)
        assert result.continuous.empty
        assert result.n_contracts == 0
        assert result.n_rolls == 0

    def test_single_contract_no_roll(self, tmp_path):
        df = _make_contract_df("ESH5", "2024-11-01", 50,
                               base_price=6000, base_volume=100000)
        self._write_contract_csv(tmp_path, "ESH5", df)
        result = build_continuous("ES", tmp_path)
        assert result.n_contracts == 1
        assert result.n_rolls == 0
        assert len(result.continuous) == 50
        assert (result.continuous["adjustment"] == 0.0).all()

    def test_roll_log_schema(self, tmp_path):
        front = _make_contract_df("ESM4", "2024-03-01", 60,
                                  base_price=5300, base_volume=80000,
                                  volume_trend=-2000)
        back = _make_contract_df("ESU4", "2024-04-15", 60,
                                 base_price=5310, base_volume=5000,
                                 volume_trend=3000)
        self._write_contract_csv(tmp_path, "ESM4", front)
        self._write_contract_csv(tmp_path, "ESU4", back)
        result = build_continuous("ES", tmp_path)
        if not result.roll_log.empty:
            expected_cols = {"date", "from_contract", "to_contract",
                             "from_close", "to_close", "adjustment",
                             "cumulative_adjustment", "active_contract_count"}
            assert set(result.roll_log.columns) == expected_cols


# ---------------------------------------------------------------------------
# Tests: build_all (multi-symbol pipeline)
# ---------------------------------------------------------------------------

class TestBuildAll:
    def _write_contract_csv(self, tmp_dir: Path, symbol: str, df: pd.DataFrame):
        csv_path = tmp_dir / f"test.{symbol}.csv.zst"
        out = pd.DataFrame({
            "ts_event": pd.to_datetime(df["date"]).astype(str),
            "rtype": 0,
            "publisher_id": 0,
            "instrument_id": 0,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "symbol": symbol,
        })
        out.to_csv(csv_path, index=False)

    def test_multi_symbol_outputs(self, tmp_path):
        base_dir = tmp_path / "data"
        out_dir = tmp_path / "output"

        # ES
        es_dir = base_dir / "ES"
        es_dir.mkdir(parents=True)
        front = _make_contract_df("ESH5", "2024-11-01", 40,
                                  base_price=6000, base_volume=100000)
        self._write_contract_csv(es_dir, "ESH5", front)

        # CL
        cl_dir = base_dir / "CL"
        cl_dir.mkdir(parents=True)
        clf = _make_contract_df("CLF5", "2024-11-01", 40,
                                base_price=70, base_volume=80000)
        self._write_contract_csv(cl_dir, "CLF5", clf)

        results = build_all(["ES", "CL"], base_dir, out_dir)
        assert "ES" in results
        assert "CL" in results
        assert (out_dir / "ES_continuous.csv").exists()
        assert (out_dir / "CL_continuous.csv").exists()
        assert (out_dir / "roll_log.csv").exists()

    def test_empty_symbol_dir(self, tmp_path):
        base_dir = tmp_path / "data"
        out_dir = tmp_path / "output"
        pa_dir = base_dir / "PA"
        pa_dir.mkdir(parents=True)

        results = build_all(["PA"], base_dir, out_dir)
        assert "PA" in results
        assert results["PA"].continuous.empty

    def test_roll_log_csv_written(self, tmp_path):
        base_dir = tmp_path / "data"
        out_dir = tmp_path / "output"
        es_dir = base_dir / "ES"
        es_dir.mkdir(parents=True)
        df = _make_contract_df("ESH5", "2024-11-01", 30,
                               base_price=6000, base_volume=100000)
        self._write_contract_csv(es_dir, "ESH5", df)
        build_all(["ES"], base_dir, out_dir)
        rl_path = out_dir / "roll_log.csv"
        assert rl_path.exists()
        rl = pd.read_csv(rl_path)
        assert "root" in rl.columns or rl.empty


# ---------------------------------------------------------------------------
# Tests: MONTH_CODES and YEAR_MAP constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_twelve_month_codes(self):
        assert len(MONTH_CODES) == 12

    def test_month_codes_range(self):
        assert set(MONTH_CODES.values()) == set(range(1, 13))

    def test_year_map_covers_data_range(self):
        assert 2018 in YEAR_MAP.values()
        assert 2026 in YEAR_MAP.values()

    def test_year_map_no_overlap(self):
        # All mapped years should be unique.
        assert len(set(YEAR_MAP.values())) == len(YEAR_MAP)


# ---------------------------------------------------------------------------
# Tests: Adjustment convention (Task F hotfix)
# ---------------------------------------------------------------------------

class TestAdjustmentConvention:
    """Test that the adjustment convention parameter works correctly."""

    def test_subtract_is_default(self):
        """Default convention should be 'subtract' (backward-compatible)."""
        df = _make_contract_df("ESH5", "2024-01-02", 50)
        contracts = {"ESH5": df}
        active = pd.DataFrame({"date": df["date"], "contract": "ESH5"})
        result_default = apply_panama_adjustment(contracts, active, [])
        result_subtract = apply_panama_adjustment(contracts, active, [], convention="subtract")
        pd.testing.assert_frame_equal(result_default, result_subtract)

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown convention"):
            apply_panama_adjustment({}, pd.DataFrame(columns=["date", "contract"]),
                                    [], convention="invalid")

    def test_subtract_lowers_historical_when_new_higher(self):
        """With subtract convention: when new contract is higher, historical
        prices are shifted down."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=110.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls, convention="subtract")
        pre_roll = result[result["adjustment"] != 0.0]
        if not pre_roll.empty:
            row = pre_roll.iloc[0]
            raw = contracts[row["contract"]]
            raw_bar = raw[raw["date"] == row["Date"].date()].iloc[0]
            # Subtract convention: adjusted = raw - adj
            assert row["Close"] < raw_bar["close"]  # shifted down

    def test_add_raises_historical_when_new_higher(self):
        """With add convention: when new contract is higher, historical
        prices are shifted up."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=110.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls, convention="add")
        pre_roll = result[result["adjustment"] != 0.0]
        if not pre_roll.empty:
            row = pre_roll.iloc[0]
            raw = contracts[row["contract"]]
            raw_bar = raw[raw["date"] == row["Date"].date()].iloc[0]
            # Add convention: adjusted = raw + adj
            assert row["Close"] > raw_bar["close"]  # shifted up

    def test_conventions_produce_different_results(self):
        """subtract and add should give different prices when there are rolls."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=110.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls_a, active = detect_rolls(contracts, order)
        # Need fresh roll objects for each call (cumulative state is mutated).
        rolls_b = [RollEvent(date=r.date, from_contract=r.from_contract,
                             to_contract=r.to_contract, from_close=r.from_close,
                             to_close=r.to_close, adjustment=r.adjustment)
                    for r in rolls_a]
        result_sub = apply_panama_adjustment(contracts, active, rolls_a, convention="subtract")
        result_add = apply_panama_adjustment(contracts, active, rolls_b, convention="add")
        # Pre-roll Close values should differ.
        pre_sub = result_sub[result_sub["adjustment"] != 0.0]
        pre_add = result_add[result_add["adjustment"] != 0.0]
        if not pre_sub.empty and not pre_add.empty:
            assert not np.allclose(pre_sub["Close"].values, pre_add["Close"].values)

    def test_post_roll_prices_identical_both_conventions(self):
        """Post-roll prices should be the same regardless of convention
        (adjustment=0 for the latest segment)."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=110.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls_a, active = detect_rolls(contracts, order)
        rolls_b = [RollEvent(date=r.date, from_contract=r.from_contract,
                             to_contract=r.to_contract, from_close=r.from_close,
                             to_close=r.to_close, adjustment=r.adjustment)
                    for r in rolls_a]
        result_sub = apply_panama_adjustment(contracts, active, rolls_a, convention="subtract")
        result_add = apply_panama_adjustment(contracts, active, rolls_b, convention="add")
        post_sub = result_sub[result_sub["adjustment"] == 0.0]
        post_add = result_add[result_add["adjustment"] == 0.0]
        if not post_sub.empty and not post_add.empty:
            np.testing.assert_allclose(
                post_sub["Close"].values, post_add["Close"].values, atol=1e-6
            )

    def test_build_continuous_convention_passthrough(self, tmp_path):
        """build_continuous should accept and record convention."""
        df = _make_contract_df("ESH5", "2024-11-01", 50,
                               base_price=6000, base_volume=100000)
        csv_path = tmp_path / "test.ESH5.csv.zst"
        out = pd.DataFrame({
            "ts_event": pd.to_datetime(df["date"]).astype(str),
            "rtype": 0, "publisher_id": 0, "instrument_id": 0,
            "open": df["open"], "high": df["high"], "low": df["low"],
            "close": df["close"], "volume": df["volume"], "symbol": "ESH5",
        })
        out.to_csv(csv_path, index=False)
        result = build_continuous("ES", tmp_path, convention="add")
        assert result.convention == "add"


# ---------------------------------------------------------------------------
# Tests: ES-like synthetic case (inverted convention)
# ---------------------------------------------------------------------------

class TestESLikeInvertedConvention:
    """Simulate ES-like scenario where the 'add' convention better matches
    a reference series (TradeStation-style back-adjustment)."""

    def _build_es_scenario(self):
        """Build 3-contract ES chain with typical contango (each new
        contract trades higher). The 'add' convention should produce
        prices that match a reference built with additive-up adjustment.
        """
        # ESH4 → ESM4 → ESU4, each ~50 points higher.
        front = _make_contract_df(
            "ESH4", "2024-01-02", 60,
            base_price=4800.0, base_volume=100000, trend=0.5,
            volume_trend=-2000,
        )
        mid = _make_contract_df(
            "ESM4", "2024-02-15", 60,
            base_price=4850.0, base_volume=5000, trend=0.5,
            volume_trend=3000,
        )
        back = _make_contract_df(
            "ESU4", "2024-04-01", 60,
            base_price=4900.0, base_volume=1000, trend=0.5,
            volume_trend=2500,
        )
        # Force second roll: ESU4 volume overtakes ESM4.
        for i in range(len(back)):
            back.loc[i, "volume"] = 1000 + i * 5000
        contracts = {"ESH4": front, "ESM4": mid, "ESU4": back}
        order = parse_contracts(list(contracts.keys()))
        return contracts, order

    def test_add_convention_closer_to_additive_up_reference(self):
        """Build a reference with add convention, verify calibration picks it."""
        contracts, order = self._build_es_scenario()
        rolls, active = detect_rolls(contracts, order)

        # Build "reference" using add convention (simulating TradeStation).
        rolls_ref = [RollEvent(date=r.date, from_contract=r.from_contract,
                               to_contract=r.to_contract, from_close=r.from_close,
                               to_close=r.to_close, adjustment=r.adjustment)
                     for r in rolls]
        ref = apply_panama_adjustment(contracts, active, rolls_ref, convention="add")
        ref = ref.rename(columns={"Date": "Date", "Close": "Close"})

        result = calibrate_convention(contracts, order, ref)
        assert result.recommended == "add"
        assert result.scores["add"].mean_close_diff < 1e-6
        assert result.scores["subtract"].mean_close_diff > 1.0

    def test_calibration_returns_both_scores(self):
        contracts, order = self._build_es_scenario()
        rolls, active = detect_rolls(contracts, order)
        rolls_ref = [RollEvent(date=r.date, from_contract=r.from_contract,
                               to_contract=r.to_contract, from_close=r.from_close,
                               to_close=r.to_close, adjustment=r.adjustment)
                     for r in rolls]
        ref = apply_panama_adjustment(contracts, active, rolls_ref, convention="add")

        result = calibrate_convention(contracts, order, ref)
        assert "subtract" in result.scores
        assert "add" in result.scores
        assert result.scores["add"].overlap_bars > 0
        assert result.scores["subtract"].overlap_bars > 0


# ---------------------------------------------------------------------------
# Tests: CL-like dense monthly rolls
# ---------------------------------------------------------------------------

class TestCLLikeDenseRolls:
    """CL rolls monthly. Test with a dense chain of 6+ contracts."""

    def _build_cl_chain(self, n_contracts: int = 6):
        """Build a CL-like chain with monthly rolls."""
        month_codes = ["F", "G", "H", "J", "K", "M"]
        contracts = {}
        start_price = 70.0
        for i, mc in enumerate(month_codes[:n_contracts]):
            sym = f"CL{mc}4"
            start_date = pd.Timestamp("2024-01-02") + pd.tseries.offsets.BDay(i * 20)
            n_bars = 40
            vol_start = 100000 - i * 15000 if i == 0 else 5000
            vol_trend = -3000 if i == 0 else 4000
            df = _make_contract_df(
                sym, str(start_date.date()), n_bars,
                base_price=start_price + i * 1.5,
                base_volume=max(vol_start, 1000),
                trend=0.05,
                volume_trend=vol_trend if i < n_contracts - 1 else 0,
            )
            # For contracts after the first, ensure volume ramps up to
            # overtake the previous.
            if i > 0:
                for j in range(len(df)):
                    df.loc[j, "volume"] = 2000 + j * 6000
            contracts[sym] = df
        order = parse_contracts(list(contracts.keys()))
        return contracts, order

    def test_multiple_rolls_detected(self):
        contracts, order = self._build_cl_chain(6)
        rolls, active = detect_rolls(contracts, order)
        # Dense chain should produce multiple rolls.
        assert len(rolls) >= 2

    def test_roll_log_has_active_contract_count(self):
        contracts, order = self._build_cl_chain(6)
        rolls, active = detect_rolls(contracts, order)
        roll_log = _build_roll_log(rolls, len(contracts))
        assert "active_contract_count" in roll_log.columns
        if not roll_log.empty:
            # Count should decrease with each roll.
            counts = roll_log["active_contract_count"].tolist()
            assert counts == sorted(counts, reverse=True)

    def test_dense_chain_both_conventions_valid_schema(self):
        contracts, order = self._build_cl_chain(6)
        rolls_a, active = detect_rolls(contracts, order)
        rolls_b = [RollEvent(date=r.date, from_contract=r.from_contract,
                             to_contract=r.to_contract, from_close=r.from_close,
                             to_close=r.to_close, adjustment=r.adjustment)
                    for r in rolls_a]
        for conv, rr in [("subtract", rolls_a), ("add", rolls_b)]:
            result = apply_panama_adjustment(contracts, active, rr, convention=conv)
            assert "Date" in result.columns
            assert "Close" in result.columns
            assert result["Date"].nunique() == len(result)
            assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_dense_chain_monotonic_dates(self):
        contracts, order = self._build_cl_chain(6)
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        dates = result["Date"].values
        assert (np.diff(dates) > np.timedelta64(0)).all()

    def test_calibration_with_dense_rolls(self):
        """Calibration should work with many rolls."""
        contracts, order = self._build_cl_chain(6)
        rolls, active = detect_rolls(contracts, order)
        # Build reference using subtract.
        rolls_ref = [RollEvent(date=r.date, from_contract=r.from_contract,
                               to_contract=r.to_contract, from_close=r.from_close,
                               to_close=r.to_close, adjustment=r.adjustment)
                     for r in rolls]
        ref = apply_panama_adjustment(contracts, active, rolls_ref, convention="subtract")
        result = calibrate_convention(contracts, order, ref)
        assert result.recommended == "subtract"
        assert result.scores["subtract"].mean_close_diff < 1e-6


# ---------------------------------------------------------------------------
# Tests: Roll diagnostics
# ---------------------------------------------------------------------------

class TestRollDiagnostics:
    def test_build_roll_log_schema(self):
        rolls = [
            RollEvent(
                date=datetime.date(2024, 3, 15),
                from_contract="CLF4", to_contract="CLG4",
                from_close=72.0, to_close=73.5,
                adjustment=1.5, cumulative_adjustment=3.0,
            ),
            RollEvent(
                date=datetime.date(2024, 4, 15),
                from_contract="CLG4", to_contract="CLH4",
                from_close=74.0, to_close=75.0,
                adjustment=1.0, cumulative_adjustment=1.0,
            ),
        ]
        rl = _build_roll_log(rolls, n_contracts=4)
        expected = {"date", "from_contract", "to_contract", "from_close",
                    "to_close", "adjustment", "cumulative_adjustment",
                    "active_contract_count"}
        assert set(rl.columns) == expected
        assert len(rl) == 2

    def test_active_contract_count_decreasing(self):
        rolls = [
            RollEvent(date=datetime.date(2024, 3, 15),
                      from_contract="A", to_contract="B",
                      from_close=100, to_close=102, adjustment=2),
            RollEvent(date=datetime.date(2024, 6, 15),
                      from_contract="B", to_contract="C",
                      from_close=104, to_close=105, adjustment=1),
        ]
        rl = _build_roll_log(rolls, n_contracts=5)
        assert rl.iloc[0]["active_contract_count"] == 4  # 5 - 1
        assert rl.iloc[1]["active_contract_count"] == 3  # 5 - 2

    def test_empty_rolls_returns_empty_with_schema(self):
        rl = _build_roll_log([], n_contracts=3)
        assert rl.empty
        assert "active_contract_count" in rl.columns


# ---------------------------------------------------------------------------
# Tests: Sanity guards in load_contract_data
# ---------------------------------------------------------------------------

class TestSanityGuards:
    def _write_contract_csv(self, tmp_dir, symbol, df):
        csv_path = tmp_dir / f"test.{symbol}.csv.zst"
        out = pd.DataFrame({
            "ts_event": pd.to_datetime(df["date"]).astype(str),
            "rtype": 0, "publisher_id": 0, "instrument_id": 0,
            "open": df["open"], "high": df["high"], "low": df["low"],
            "close": df["close"], "volume": df["volume"], "symbol": symbol,
        })
        out.to_csv(csv_path, index=False)

    def test_single_row_contract_skipped(self, tmp_path):
        """Contracts with < 2 bars should be skipped."""
        df = _make_contract_df("ESH5", "2024-01-02", 1)
        self._write_contract_csv(tmp_path, "ESH5", df)
        contracts = load_contract_data(tmp_path, "ES")
        assert "ESH5" not in contracts

    def test_zero_volume_contract_skipped(self, tmp_path):
        """Contracts with all-zero volume should be skipped."""
        df = _make_contract_df("ESH5", "2024-01-02", 10, base_volume=0)
        self._write_contract_csv(tmp_path, "ESH5", df)
        contracts = load_contract_data(tmp_path, "ES")
        assert "ESH5" not in contracts

    def test_valid_contract_still_loaded(self, tmp_path):
        """Contracts with >= 2 bars and non-zero volume should load."""
        df = _make_contract_df("ESH5", "2024-01-02", 50, base_volume=10000)
        self._write_contract_csv(tmp_path, "ESH5", df)
        contracts = load_contract_data(tmp_path, "ES")
        assert "ESH5" in contracts
        assert len(contracts["ESH5"]) == 50


# ---------------------------------------------------------------------------
# Tests: Deterministic output (hotfix)
# ---------------------------------------------------------------------------

class TestDeterministicHotfix:
    def test_deterministic_subtract(self):
        contracts, order = _two_contract_setup(
            front_base=100.0, back_base=110.0,
            front_vol_start=50000, back_vol_start=10000,
            front_vol_trend=-3000, back_vol_trend=5000,
        )
        rolls1, active1 = detect_rolls(contracts, order)
        rolls2, active2 = detect_rolls(contracts, order)
        r1 = apply_panama_adjustment(contracts, active1, rolls1, convention="subtract")
        r2 = apply_panama_adjustment(contracts, active2, rolls2, convention="subtract")
        pd.testing.assert_frame_equal(r1, r2)

    def test_deterministic_add(self):
        contracts, order = _two_contract_setup(
            front_base=100.0, back_base=110.0,
            front_vol_start=50000, back_vol_start=10000,
            front_vol_trend=-3000, back_vol_trend=5000,
        )
        rolls1, active1 = detect_rolls(contracts, order)
        rolls2, active2 = detect_rolls(contracts, order)
        r1 = apply_panama_adjustment(contracts, active1, rolls1, convention="add")
        r2 = apply_panama_adjustment(contracts, active2, rolls2, convention="add")
        pd.testing.assert_frame_equal(r1, r2)

    def test_roll_log_schema_stable(self):
        """Roll log columns should be the same across runs."""
        contracts, order = _two_contract_setup(
            front_base=100.0, back_base=110.0,
            front_vol_start=50000, back_vol_start=10000,
            front_vol_trend=-3000, back_vol_trend=5000,
        )
        rolls1, _ = detect_rolls(contracts, order)
        rolls2, _ = detect_rolls(contracts, order)
        rl1 = _build_roll_log(rolls1, len(contracts))
        rl2 = _build_roll_log(rolls2, len(contracts))
        assert list(rl1.columns) == list(rl2.columns)
        pd.testing.assert_frame_equal(rl1, rl2)


# ---------------------------------------------------------------------------
# Tests: Calibration helper
# ---------------------------------------------------------------------------

class TestCalibrationHelper:
    def test_no_overlap_returns_inf(self):
        """If reference dates don't overlap with continuous, scores are inf."""
        contracts, order = _two_contract_setup(
            front_base=100.0, back_base=105.0,
            front_vol_start=50000, back_vol_start=10000,
            front_vol_trend=-3000, back_vol_trend=5000,
        )
        # Reference from a completely different date range.
        ref = pd.DataFrame({
            "Date": pd.bdate_range("2010-01-02", periods=100),
            "Close": np.linspace(50, 60, 100),
        })
        result = calibrate_convention(contracts, order, ref)
        for s in result.scores.values():
            assert s.overlap_bars == 0
            assert s.mean_close_diff == np.inf

    def test_perfect_match_scores_zero(self):
        """When reference is built with same convention, diff should be ~0."""
        contracts, order = _two_contract_setup(
            front_base=100.0, back_base=105.0,
            front_vol_start=50000, back_vol_start=10000,
            front_vol_trend=-3000, back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        rolls_ref = [RollEvent(date=r.date, from_contract=r.from_contract,
                               to_contract=r.to_contract, from_close=r.from_close,
                               to_close=r.to_close, adjustment=r.adjustment)
                     for r in rolls]
        ref = apply_panama_adjustment(contracts, active, rolls_ref, convention="subtract")
        result = calibrate_convention(contracts, order, ref)
        assert result.recommended == "subtract"
        assert result.scores["subtract"].mean_close_diff < 1e-6

    def test_result_has_symbol(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        rolls_ref = [RollEvent(date=r.date, from_contract=r.from_contract,
                               to_contract=r.to_contract, from_close=r.from_close,
                               to_close=r.to_close, adjustment=r.adjustment)
                     for r in rolls]
        ref = apply_panama_adjustment(contracts, active, rolls_ref, convention="subtract")
        result = calibrate_convention(contracts, order, ref)
        assert result.symbol == "ES"
