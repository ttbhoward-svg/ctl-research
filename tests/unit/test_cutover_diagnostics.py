"""Unit tests for cutover diagnostics orchestrator (Data Cutover Task H)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.cutover_diagnostics import (
    DiagnosticResult,
    L2Result,
    L3Result,
    L4Result,
    run_diagnostics,
    run_l2,
    run_l3,
    run_l4,
    save_diagnostic_artifacts,
)
from ctl.roll_reconciliation import (
    RollComparisonResult,
    RollManifestEntry,
    RollMatch,
    TSRollEvent,
    compare_roll_schedules,
    derive_ts_roll_events_from_spread,
)

# Also test the continuous_builder extensions.
from ctl.continuous_builder import (
    BUILD_MODES,
    MODE_CONVENTION,
    BuildMode,
    ContinuousResult,
    RollEvent,
    _build_roll_log,
    build_continuous_with_mode,
    export_roll_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest_entry(
    roll_date: str,
    from_contract: str = "ESH5",
    to_contract: str = "ESM5",
    gap: float = 10.0,
    cumulative_adj: float = 10.0,
    convention: str = "subtract",
) -> RollManifestEntry:
    return RollManifestEntry(
        roll_date=roll_date,
        from_contract=from_contract,
        to_contract=to_contract,
        from_close=5000.0,
        to_close=5000.0 + gap,
        gap=gap,
        cumulative_adj=cumulative_adj,
        convention=convention,
    )


def _make_ts_roll(date: str, gap: float = 10.0) -> TSRollEvent:
    return TSRollEvent(
        date=pd.Timestamp(date).date(),
        gap=gap,
        close_before=5000.0,
        close_after=5000.0 + gap,
    )


def _make_adjusted_df(
    start: str,
    n_bars: int,
    base_price: float = 100.0,
    trend: float = 0.1,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_bars)
    prices = [base_price + i * trend for i in range(n_bars)]
    return pd.DataFrame({"Date": dates, "Close": prices})


def _make_unadjusted_ts_df(dates: list, closes: list) -> pd.DataFrame:
    return pd.DataFrame({"Date": dates, "Close": closes})


# ===========================================================================
# L2 — Roll schedule comparison diagnostic
# ===========================================================================

class TestL2:
    def test_exact_match(self):
        """Identical schedules should produce PASS."""
        entries = [_make_manifest_entry("2024-03-15")]
        ts_rolls = [_make_ts_roll("2024-03-15")]
        result = run_l2(entries, ts_rolls, "ES")
        assert result.status == "PASS"
        assert result.symbol == "ES"
        assert len(result.detail_df) == 1

    def test_one_day_shift(self):
        """1-day shift should produce WATCH."""
        entries = [_make_manifest_entry("2024-03-15")]
        ts_rolls = [_make_ts_roll("2024-03-14")]
        result = run_l2(entries, ts_rolls, "ES")
        assert result.status == "WATCH"

    def test_missing_ts_roll(self):
        """Missing TS roll should produce FAIL."""
        entries = [
            _make_manifest_entry("2024-03-15"),
            _make_manifest_entry("2024-06-20"),
        ]
        ts_rolls = [_make_ts_roll("2024-03-15")]
        result = run_l2(entries, ts_rolls, "ES")
        assert result.status == "FAIL"
        assert result.comparison.n_fail == 1

    def test_to_dict(self):
        entries = [_make_manifest_entry("2024-03-15")]
        ts_rolls = [_make_ts_roll("2024-03-15")]
        result = run_l2(entries, ts_rolls, "ES")
        d = result.to_dict()
        assert d["symbol"] == "ES"
        assert d["status"] == "PASS"
        assert d["n_matched"] == 1


# ===========================================================================
# L3 — Roll gap comparison
# ===========================================================================

class TestL3:
    def test_gap_comparison(self):
        """Matched rolls should have gap diff computed."""
        entries = [_make_manifest_entry("2024-03-15", gap=10.0)]
        ts_rolls = [_make_ts_roll("2024-03-15", gap=12.0)]
        l2 = run_l2(entries, ts_rolls, "ES")
        result = run_l3(l2, "ES")
        assert result.n_compared == 1
        assert result.mean_gap_diff == pytest.approx(2.0, abs=0.01)
        assert result.max_gap_diff == pytest.approx(2.0, abs=0.01)

    def test_no_matched_rolls(self):
        """No matched rolls means nothing to compare."""
        entries = [_make_manifest_entry("2024-03-15")]
        l2 = run_l2(entries, [], "ES")
        result = run_l3(l2, "ES")
        assert result.n_compared == 0

    def test_to_dict(self):
        entries = [_make_manifest_entry("2024-03-15", gap=10.0)]
        ts_rolls = [_make_ts_roll("2024-03-15", gap=10.0)]
        l2 = run_l2(entries, ts_rolls, "ES")
        result = run_l3(l2, "ES")
        d = result.to_dict()
        assert d["symbol"] == "ES"
        assert d["n_compared"] == 1

    def test_multiple_gaps(self):
        """Multiple matched rolls should average gaps."""
        entries = [
            _make_manifest_entry("2024-03-15", gap=10.0),
            _make_manifest_entry("2024-06-20", gap=20.0),
        ]
        ts_rolls = [
            _make_ts_roll("2024-03-15", gap=12.0),
            _make_ts_roll("2024-06-20", gap=18.0),
        ]
        l2 = run_l2(entries, ts_rolls, "ES")
        result = run_l3(l2, "ES")
        assert result.n_compared == 2
        assert result.mean_gap_diff == pytest.approx(2.0, abs=0.01)
        assert result.max_gap_diff == pytest.approx(2.0, abs=0.01)


# ===========================================================================
# L4 — Adjusted series drift
# ===========================================================================

class TestL4:
    def test_constant_drift(self):
        """Parallel series with offset should have constant drift."""
        can = _make_adjusted_df("2024-01-02", 30, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 30, base_price=105.0)
        roll_df = pd.DataFrame()
        result = run_l4(can, ts, roll_df, "ES")
        assert result.n_overlap == 30
        assert result.mean_drift == pytest.approx(5.0, abs=0.01)
        assert result.max_drift == pytest.approx(5.0, abs=0.01)

    def test_no_overlap(self):
        """Non-overlapping series should produce empty result."""
        can = _make_adjusted_df("2020-01-02", 10)
        ts = _make_adjusted_df("2025-01-02", 10)
        roll_df = pd.DataFrame()
        result = run_l4(can, ts, roll_df, "ES")
        assert result.n_overlap == 0

    def test_drift_explanation_populated(self):
        """L4 should include drift explanation."""
        can = _make_adjusted_df("2024-01-02", 50, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 50, base_price=110.0)
        roll_df = pd.DataFrame()
        result = run_l4(can, ts, roll_df, "ES")
        assert result.explanation is not None
        assert result.explanation.symbol == "ES"
        assert result.explanation.n_intervals >= 1

    def test_to_dict(self):
        can = _make_adjusted_df("2024-01-02", 10, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 10, base_price=100.0)
        result = run_l4(can, ts, pd.DataFrame(), "ES")
        d = result.to_dict()
        assert d["symbol"] == "ES"
        assert "n_overlap" in d


# ===========================================================================
# Full orchestrator
# ===========================================================================

class TestRunDiagnostics:
    def test_exact_match_scenario(self):
        """All layers should run without error for exact match."""
        can = _make_adjusted_df("2024-01-02", 50, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 50, base_price=100.0)
        entries = [_make_manifest_entry("2024-02-01")]
        # Build spread-compatible unadjusted/adjusted pair.
        # Spread shifts by 50.0 on 2024-02-01.
        dates = list(ts["Date"])
        adj_closes = list(ts["Close"])
        unadj_closes = []
        roll_ts = pd.Timestamp("2024-02-01")
        for i, d in enumerate(dates):
            if d >= roll_ts:
                unadj_closes.append(adj_closes[i] + 50.0)
            else:
                unadj_closes.append(adj_closes[i])
        ts_unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})
        result = run_diagnostics(can, ts, entries, ts_unadj, "ES", tick_size=0.25)
        assert isinstance(result, DiagnosticResult)
        assert result.symbol == "ES"
        assert result.l2 is not None
        assert result.l3 is not None
        assert result.l4 is not None

    def test_no_ts_unadjusted(self):
        """Should handle None ts_unadj_df gracefully."""
        can = _make_adjusted_df("2024-01-02", 20)
        ts = _make_adjusted_df("2024-01-02", 20)
        entries = [_make_manifest_entry("2024-01-15")]
        result = run_diagnostics(can, ts, entries, None, "ES")
        assert result.l2.comparison.n_ts == 0
        assert result.l2.status == "FAIL"  # No TS rolls to match against.

    def test_missing_roll_produces_fail(self):
        """Unmatched canonical roll should produce FAIL in L2."""
        can = _make_adjusted_df("2024-01-02", 30)
        ts = _make_adjusted_df("2024-01-02", 30)
        entries = [
            _make_manifest_entry("2024-01-15"),
            _make_manifest_entry("2024-02-15"),
        ]
        # Spread-compatible pair: one roll on 2024-01-15 only.
        dates = list(ts["Date"])
        adj_closes = list(ts["Close"])
        unadj_closes = []
        roll_ts = pd.Timestamp("2024-01-15")
        for i, d in enumerate(dates):
            if d >= roll_ts:
                unadj_closes.append(adj_closes[i] + 100.0)
            else:
                unadj_closes.append(adj_closes[i])
        ts_unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})
        result = run_diagnostics(can, ts, entries, ts_unadj, "ES", tick_size=0.25)
        assert result.l2.comparison.n_fail >= 1

    def test_unexplained_drift(self):
        """L4 should capture drift even without roll explanation."""
        can = _make_adjusted_df("2024-01-02", 40, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 40, base_price=120.0)
        result = run_diagnostics(can, ts, [], None, "CL")
        assert result.l4.mean_drift == pytest.approx(20.0, abs=0.01)
        assert result.l4.explanation is not None
        assert result.l4.explanation.overall_mean_drift == pytest.approx(20.0, abs=0.01)

    def test_to_dict(self):
        can = _make_adjusted_df("2024-01-02", 10)
        ts = _make_adjusted_df("2024-01-02", 10)
        result = run_diagnostics(can, ts, [], None, "ES")
        d = result.to_dict()
        assert "l2" in d
        assert "l3" in d
        assert "l4" in d
        assert d["symbol"] == "ES"

    def test_spread_method_plausible_count(self):
        """Integration: run_diagnostics with spread method yields plausible n_ts."""
        # Build 200-bar series with 2 true rolls via spread.
        dates = pd.bdate_range("2024-01-02", periods=200)
        adj_closes = [5000.0 + i * 0.5 for i in range(200)]
        unadj_closes = list(adj_closes)
        # Roll 1 at bar 50, Roll 2 at bar 120.
        for i in range(50, 200):
            unadj_closes[i] += 25.0
        for i in range(120, 200):
            unadj_closes[i] += 15.0
        can = pd.DataFrame({"Date": dates, "Close": adj_closes})
        ts = pd.DataFrame({"Date": dates, "Close": adj_closes})
        ts_unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})

        entries = [
            _make_manifest_entry(str(dates[50].date()), gap=25.0),
            _make_manifest_entry(str(dates[120].date()), gap=15.0),
        ]
        result = run_diagnostics(can, ts, entries, ts_unadj, "ES", tick_size=0.25)
        assert result.l2.comparison.n_ts == 2
        assert result.l2.comparison.n_matched == 2


# ===========================================================================
# Artifact persistence
# ===========================================================================

class TestSaveDiagnosticArtifacts:
    def test_saves_all_files(self, tmp_path):
        """Should save L2, L3, L4 CSV and L4 JSON."""
        can = _make_adjusted_df("2024-01-02", 30, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 30, base_price=105.0)
        entries = [_make_manifest_entry("2024-01-15")]
        # Spread-compatible unadj: shift spread on 2024-01-15.
        dates = list(ts["Date"])
        adj_closes = list(ts["Close"])
        unadj_closes = []
        roll_ts = pd.Timestamp("2024-01-15")
        for i, d in enumerate(dates):
            if d >= roll_ts:
                unadj_closes.append(adj_closes[i] + 100.0)
            else:
                unadj_closes.append(adj_closes[i])
        ts_unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})
        result = run_diagnostics(can, ts, entries, ts_unadj, "ES", tick_size=0.25)
        paths = save_diagnostic_artifacts(result, tmp_path, prefix="ES_")

        assert "l2_csv" in paths
        assert "l3_csv" in paths
        assert "l4_csv" in paths
        assert "l4_json" in paths

        for p in paths.values():
            assert p.exists()

    def test_file_names(self, tmp_path):
        """Artifact file names should follow naming convention."""
        can = _make_adjusted_df("2024-01-02", 10)
        ts = _make_adjusted_df("2024-01-02", 10)
        result = run_diagnostics(can, ts, [], None, "CL")
        paths = save_diagnostic_artifacts(result, tmp_path, prefix="CL_")

        assert paths["l2_csv"].name == "CL_L2_roll_schedule_comparison.csv"
        assert paths["l3_csv"].name == "CL_L3_roll_gap_comparison.csv"
        assert paths["l4_csv"].name == "CL_L4_adjusted_series_drift.csv"
        assert paths["l4_json"].name == "CL_L4_drift_explanation.json"

    def test_l4_json_valid(self, tmp_path):
        """L4 drift explanation JSON should be valid JSON."""
        can = _make_adjusted_df("2024-01-02", 20, base_price=100.0)
        ts = _make_adjusted_df("2024-01-02", 20, base_price=110.0)
        result = run_diagnostics(can, ts, [], None, "ES")
        paths = save_diagnostic_artifacts(result, tmp_path, prefix="ES_")

        with open(paths["l4_json"]) as f:
            data = json.load(f)
        assert "overall_mean_drift" in data
        assert data["symbol"] == "ES"

    def test_creates_output_dir(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        out = tmp_path / "sub" / "diagnostics"
        can = _make_adjusted_df("2024-01-02", 10)
        ts = _make_adjusted_df("2024-01-02", 10)
        result = run_diagnostics(can, ts, [], None, "ES")
        paths = save_diagnostic_artifacts(result, out, prefix="ES_")
        assert out.exists()
        for p in paths.values():
            assert p.exists()


# ===========================================================================
# continuous_builder extensions
# ===========================================================================

class TestBuildMode:
    def test_build_modes_tuple(self):
        assert "parity_mode" in BUILD_MODES
        assert "canonical_mode" in BUILD_MODES

    def test_mode_convention_mapping(self):
        assert MODE_CONVENTION["parity_mode"] == "add"
        assert MODE_CONVENTION["canonical_mode"] == "subtract"

    def test_continuous_result_has_build_mode(self):
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=pd.DataFrame(),
        )
        assert result.build_mode == "canonical_mode"

    def test_invalid_build_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown build mode"):
            build_continuous_with_mode("ES", tmp_path, mode="invalid_mode")


class TestExportRollManifest:
    def test_export_basic(self, tmp_path):
        """Export should write valid JSON."""
        rolls = [
            RollEvent(
                date="2024-03-15",
                from_contract="ESH5",
                to_contract="ESM5",
                from_close=5000.0,
                to_close=5010.0,
                adjustment=10.0,
                cumulative_adjustment=10.0,
            ),
        ]
        roll_log = _build_roll_log(rolls, 3)
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=roll_log,
            n_contracts=3,
            n_rolls=1,
            convention="add",
            build_mode="parity_mode",
        )
        path = tmp_path / "ES_roll_manifest.json"
        export_roll_manifest(result, path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["symbol"] == "ES"
        assert data["build_mode"] == "parity_mode"
        assert data["convention"] == "add"
        assert len(data["rolls"]) == 1
        assert data["rolls"][0]["from_contract"] == "ESH5"

    def test_manifest_schema_complete(self, tmp_path):
        """Every roll entry should have all required fields."""
        rolls = [
            RollEvent(
                date="2024-03-15",
                from_contract="ESH5",
                to_contract="ESM5",
                from_close=5000.0,
                to_close=5010.0,
                adjustment=10.0,
                cumulative_adjustment=10.0,
            ),
        ]
        roll_log = _build_roll_log(rolls, 3)
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=roll_log,
            n_contracts=3,
            n_rolls=1,
        )
        path = tmp_path / "manifest.json"
        export_roll_manifest(result, path)

        with open(path) as f:
            data = json.load(f)

        required = {
            "roll_date", "from_contract", "to_contract",
            "from_close", "to_close", "gap", "cumulative_adj",
            "trigger_reason", "confirmation_days", "convention",
            "session_template", "close_type",
        }
        for entry in data["rolls"]:
            assert required.issubset(set(entry.keys()))

    def test_export_empty_roll_log(self, tmp_path):
        """Export with no rolls should still produce valid JSON."""
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=pd.DataFrame(
                columns=["date", "from_contract", "to_contract",
                          "from_close", "to_close", "adjustment",
                          "cumulative_adjustment", "active_contract_count"]
            ),
        )
        path = tmp_path / "empty_manifest.json"
        export_roll_manifest(result, path)

        with open(path) as f:
            data = json.load(f)
        assert data["n_rolls"] == 0
        assert data["rolls"] == []

    def test_export_creates_parent_dirs(self, tmp_path):
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=pd.DataFrame(
                columns=["date", "from_contract", "to_contract",
                          "from_close", "to_close", "adjustment",
                          "cumulative_adjustment", "active_contract_count"]
            ),
        )
        path = tmp_path / "sub" / "deep" / "manifest.json"
        export_roll_manifest(result, path)
        assert path.exists()

    def test_gap_computed_from_closes(self, tmp_path):
        """Gap should be to_close - from_close."""
        rolls = [
            RollEvent(
                date="2024-03-15",
                from_contract="ESH5",
                to_contract="ESM5",
                from_close=5000.0,
                to_close=5025.0,
                adjustment=25.0,
                cumulative_adjustment=25.0,
            ),
        ]
        roll_log = _build_roll_log(rolls, 3)
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=roll_log,
            n_contracts=3,
            n_rolls=1,
        )
        path = tmp_path / "manifest.json"
        export_roll_manifest(result, path)

        with open(path) as f:
            data = json.load(f)
        assert data["rolls"][0]["gap"] == pytest.approx(25.0)
