"""Unit tests for dataset assembler and health checks."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.b1_detector import B1Trigger
from ctl.dataset_assembler import (
    SCHEMA_COLUMNS,
    assemble_dataset,
    compute_manifest,
    save_dataset,
)
from ctl.health_check import (
    HealthReport,
    run_health_checks,
)
from ctl.simulator import TradeResult
from ctl.universe import Universe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trigger(
    symbol: str = "/ES",
    trigger_bar_idx: int = 100,
    trigger_date: str = "2024-03-15",
    slope: float = 10.0,
    air: int = 8,
    stop: float = 4450.0,
    swing: float = 4550.0,
    wr_div: bool = True,
    clean: bool = False,
    vol_dec: bool = True,
    gap: bool = False,
    multi_yr: bool = True,
    single_bar: bool = False,
    fib: bool = None,
    weekly_aligned: bool = True,
    monthly_aligned: bool = None,
    cot_delta: float = None,
    cot_zscore: float = None,
    vix: bool = True,
) -> B1Trigger:
    rng = swing - stop
    t = B1Trigger(
        trigger_bar_idx=trigger_bar_idx,
        trigger_date=pd.Timestamp(trigger_date),
        symbol=symbol,
        timeframe="daily",
        slope_20=slope,
        bars_of_air=air,
        ema10_at_trigger=4500.0,
        atr14_at_trigger=50.0,
        stop_price=stop,
        swing_high=swing,
        tp1=stop + rng * 0.618,
        tp2=stop + rng * 0.786,
        tp3=stop + rng * 1.000,
        tp4=stop + rng * 1.618,
        tp5=stop + rng * 2.618,
        confirmed=True,
        entry_bar_idx=102,
        entry_date=pd.Timestamp(trigger_date) + pd.Timedelta(days=2),
        entry_price=4510.0,
    )
    t.wr_divergence = wr_div
    t.clean_pullback = clean
    t.volume_declining = vol_dec
    t.gap_fill_below = gap
    t.multi_year_highs = multi_yr
    t.single_bar_pullback = single_bar
    t.fib_confluence = fib
    t.weekly_trend_aligned = weekly_aligned
    t.monthly_trend_aligned = monthly_aligned
    t.cot_20d_delta = cot_delta
    t.cot_zscore_1y = cot_zscore
    t.vix_regime = vix
    return t


def _make_result(
    symbol: str = "/ES",
    trigger_bar_idx: int = 100,
    trigger_date: str = "2024-03-15",
) -> TradeResult:
    return TradeResult(
        symbol=symbol,
        timeframe="daily",
        trigger_date=pd.Timestamp(trigger_date),
        trigger_bar_idx=trigger_bar_idx,
        slope_20=10.0,
        bars_of_air=8,
        ema10_at_trigger=4500.0,
        atr14_at_trigger=50.0,
        entry_date=pd.Timestamp(trigger_date) + pd.Timedelta(days=2),
        entry_bar_idx=102,
        entry_price=4510.0,
        stop_price=4450.0,
        swing_high=4550.0,
        tp1=4511.8,
        tp2=4528.6,
        tp3=4550.0,
        tp4=4611.8,
        tp5=4711.8,
        exit_date=pd.Timestamp(trigger_date) + pd.Timedelta(days=10),
        exit_bar_idx=110,
        exit_price=4530.0,
        exit_reason="TP2",
        risk_per_unit=60.0,
        r_multiple_actual=(4530.0 - 4510.0) / 60.0,
        theoretical_r=0.5,
        mfe_r=1.5,
        mae_r=0.3,
        day1_fail=False,
        same_bar_collision=False,
        exit_on_last_bar=False,
        trade_outcome="Win",
        hold_bars=8,
        tp1_hit=True,
        tp2_hit=True,
        tp3_hit=False,
        asset_cluster="IDX_FUT",
        tradable_status="tradable",
    )


@pytest.fixture
def universe():
    return Universe.from_yaml()


# ---------------------------------------------------------------------------
# Tests: assemble_dataset
# ---------------------------------------------------------------------------

class TestAssembleDataset:
    def test_single_row(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        assert len(df) == 1
        assert list(df.columns) == SCHEMA_COLUMNS

    def test_confluence_flags_carried(self, universe):
        trig = _make_trigger(wr_div=True, clean=False, vol_dec=True, multi_yr=True)
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        row = df.iloc[0]
        assert row["WR_Divergence"] == True   # noqa: E712
        assert row["CleanPullback"] == False   # noqa: E712
        assert row["VolumeDeclining"] == True  # noqa: E712
        assert row["MultiYearHighs"] == True   # noqa: E712

    def test_mtfa_flags_carried(self, universe):
        trig = _make_trigger(weekly_aligned=True, monthly_aligned=False)
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        row = df.iloc[0]
        assert row["WeeklyTrendAligned"] == True    # noqa: E712
        assert row["MonthlyTrendAligned"] == False   # noqa: E712

    def test_external_features_carried(self, universe):
        trig = _make_trigger(cot_delta=500.0, cot_zscore=1.2, vix=True)
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        row = df.iloc[0]
        assert row["COT_20D_Delta"] == 500.0
        assert row["COT_ZScore_1Y"] == 1.2
        assert row["VIX_Regime"] == True  # noqa: E712

    def test_universe_metadata_injected(self, universe):
        trig = _make_trigger(symbol="/ES")
        result = _make_result(symbol="/ES")
        df = assemble_dataset([trig], [result], universe)
        assert df.iloc[0]["AssetCluster"] == "IDX_FUT"
        assert df.iloc[0]["TradableStatus"] == "tradable"

    def test_deterministic_sort(self, universe):
        """Rows sorted by (Date, Ticker) ascending."""
        trigs = [
            _make_trigger(symbol="/CL", trigger_date="2024-03-20", trigger_bar_idx=200),
            _make_trigger(symbol="/ES", trigger_date="2024-03-15", trigger_bar_idx=100),
            _make_trigger(symbol="/ES", trigger_date="2024-03-20", trigger_bar_idx=201),
        ]
        results = [
            _make_result(symbol="/CL", trigger_date="2024-03-20", trigger_bar_idx=200),
            _make_result(symbol="/ES", trigger_date="2024-03-15", trigger_bar_idx=100),
            _make_result(symbol="/ES", trigger_date="2024-03-20", trigger_bar_idx=201),
        ]
        df = assemble_dataset(trigs, results, universe)
        dates = df["Date"].tolist()
        tickers = df["Ticker"].tolist()
        assert dates[0] <= dates[1] <= dates[2]
        # Same date: /CL before /ES (alphabetical).
        assert tickers[2] == "/ES"

    def test_empty_results(self, universe):
        df = assemble_dataset([], [], universe)
        assert len(df) == 0
        assert list(df.columns) == SCHEMA_COLUMNS

    def test_unmatched_result_still_included(self, universe):
        """TradeResult without a matching trigger still produces a row (flags=None)."""
        result = _make_result()
        df = assemble_dataset([], [result], universe)
        assert len(df) == 1
        assert df.iloc[0]["WR_Divergence"] is None

    def test_multiple_symbols(self, universe):
        trigs = [
            _make_trigger(symbol="/ES", trigger_bar_idx=100),
            _make_trigger(symbol="XLE", trigger_bar_idx=100, cot_delta=None),
        ]
        results = [
            _make_result(symbol="/ES", trigger_bar_idx=100),
            _make_result(symbol="XLE", trigger_bar_idx=100),
        ]
        df = assemble_dataset(trigs, results, universe)
        assert len(df) == 2
        # ETF should have ETF_SECTOR cluster.
        xle_row = df[df["Ticker"] == "XLE"].iloc[0]
        assert xle_row["AssetCluster"] == "ETF_SECTOR"


# ---------------------------------------------------------------------------
# Tests: compute_manifest + hash reproducibility
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_fields(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        manifest = compute_manifest(df)
        assert "sha256" in manifest
        assert manifest["n_rows"] == 1
        assert manifest["n_columns"] == len(SCHEMA_COLUMNS)
        assert manifest["columns"] == SCHEMA_COLUMNS

    def test_hash_reproducibility(self, universe):
        """Same data produces same hash across two calls."""
        trig = _make_trigger()
        result = _make_result()
        df1 = assemble_dataset([trig], [result], universe)
        df2 = assemble_dataset([trig], [result], universe)
        m1 = compute_manifest(df1)
        m2 = compute_manifest(df2)
        assert m1["sha256"] == m2["sha256"]

    def test_different_data_different_hash(self, universe):
        t1 = _make_trigger(wr_div=True)
        t2 = _make_trigger(wr_div=False)
        r1 = _make_result()
        df1 = assemble_dataset([t1], [r1], universe)
        df2 = assemble_dataset([t2], [r1], universe)
        assert compute_manifest(df1)["sha256"] != compute_manifest(df2)["sha256"]

    def test_hash_is_valid_sha256(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        h = compute_manifest(df)["sha256"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_dataset_hash(self, universe):
        df = assemble_dataset([], [], universe)
        manifest = compute_manifest(df)
        assert manifest["n_rows"] == 0
        assert len(manifest["sha256"]) == 64


# ---------------------------------------------------------------------------
# Tests: save_dataset
# ---------------------------------------------------------------------------

class TestSaveDataset:
    def test_save_creates_files(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, manifest = save_dataset(df, Path(tmpdir))
            assert csv_path.exists()
            manifest_path = csv_path.with_name(csv_path.stem + "_manifest.json")
            assert manifest_path.exists()
            # Verify manifest JSON is valid.
            with open(manifest_path) as f:
                loaded = json.load(f)
            assert loaded["sha256"] == manifest["sha256"]

    def test_saved_csv_matches_hash(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, manifest = save_dataset(df, Path(tmpdir))
            with open(csv_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            assert file_hash == manifest["sha256"]


# ---------------------------------------------------------------------------
# Tests: health checks
# ---------------------------------------------------------------------------

class TestHealthChecks:
    def test_healthy_dataset_passes(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        # Should pass all critical checks (row count, nulls, duplicates, R consistency).
        for c in report.checks:
            if c.name.startswith("no_nulls") or c.name in ("row_count", "no_duplicates", "r_consistency"):
                assert c.passed, f"{c.name} failed: {c.detail}"

    def test_empty_dataset_fails_row_count(self, universe):
        df = assemble_dataset([], [], universe)
        report = run_health_checks(df, universe)
        row_check = next(c for c in report.checks if c.name == "row_count")
        assert not row_check.passed

    def test_duplicate_detection(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig, trig], [result, result], universe)
        report = run_health_checks(df, universe)
        dup_check = next(c for c in report.checks if c.name == "no_duplicates")
        assert not dup_check.passed

    def test_critical_null_detection(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        # Inject a null into a critical field.
        df.loc[0, "EntryPrice"] = np.nan
        report = run_health_checks(df, universe)
        entry_check = next(c for c in report.checks if c.name == "no_nulls_EntryPrice")
        assert not entry_check.passed

    def test_cot_null_for_non_futures(self, universe):
        """Non-futures should have NULL COT — health check passes if so."""
        trig = _make_trigger(symbol="XLE", cot_delta=None, cot_zscore=None)
        result = _make_result(symbol="XLE")
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        cot_checks = [c for c in report.checks if "cot_null_non_futures" in c.name]
        for c in cot_checks:
            assert c.passed, f"{c.name}: {c.detail}"

    def test_cot_non_null_in_non_futures_fails(self, universe):
        """Non-futures with non-null COT should fail health check."""
        trig = _make_trigger(symbol="XLE", cot_delta=500.0, cot_zscore=1.0)
        result = _make_result(symbol="XLE")
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        cot_checks = [c for c in report.checks if "cot_null_non_futures" in c.name]
        assert any(not c.passed for c in cot_checks)

    def test_r_consistency_passes(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        r_check = next(c for c in report.checks if c.name == "r_consistency")
        assert r_check.passed

    def test_r_consistency_fails_on_mismatch(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        # Corrupt the R-multiple.
        df.loc[0, "RMultiple_Actual"] = 999.0
        report = run_health_checks(df, universe)
        r_check = next(c for c in report.checks if c.name == "r_consistency")
        assert not r_check.passed

    def test_missingness_summary(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        assert len(report.missingness) == len(SCHEMA_COLUMNS)

    def test_report_summary_string(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        summary = report.summary()
        assert "Health Report" in summary
        assert "passed" in summary

    def test_all_passed_property(self, universe):
        trig = _make_trigger()
        result = _make_result()
        df = assemble_dataset([trig], [result], universe)
        report = run_health_checks(df, universe)
        # We know symbol_coverage is 1/29 but that's still "passed" (>0).
        # all_passed depends on all checks.
        assert isinstance(report.all_passed, bool)
