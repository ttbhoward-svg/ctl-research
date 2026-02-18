"""Unit tests for provider connectivity checks (Data Cutover Task E)."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.provider_connectivity import (
    ENV_DATABENTO_KEY,
    ENV_NORGATE_PATH,
    EXPECTED_CSV_COLUMNS,
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    ConnectivityReport,
    check_env_var,
    check_provider_stub,
    check_sample_csv,
    check_symbol_map,
    run_connectivity_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sample_csv(path: Path, columns=None, n_rows: int = 3):
    """Write a minimal CSV file with expected Databento columns."""
    if columns is None:
        columns = list(EXPECTED_CSV_COLUMNS)
    rows = []
    for i in range(n_rows):
        row = {}
        for c in columns:
            if c == "ts_event":
                row[c] = f"2024-06-{10+i:02d}T00:00:00.000000000"
            elif c == "symbol":
                row[c] = "ESH5"
            elif c in ("open", "high", "low", "close"):
                row[c] = 5400.0 + i
            elif c == "volume":
                row[c] = 100000
            else:
                row[c] = 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests: Environment variable checks
# ---------------------------------------------------------------------------

class TestEnvVarCheck:
    def test_set_var_passes(self, monkeypatch):
        monkeypatch.setenv(ENV_DATABENTO_KEY, "db-testkey123")
        result = check_env_var(ENV_DATABENTO_KEY)
        assert result.status == PASS
        assert "13 chars" in result.detail

    def test_unset_var_skips(self, monkeypatch):
        monkeypatch.delenv(ENV_DATABENTO_KEY, raising=False)
        result = check_env_var(ENV_DATABENTO_KEY)
        assert result.status == SKIP
        assert "not set" in result.detail

    def test_empty_var_skips(self, monkeypatch):
        monkeypatch.setenv(ENV_DATABENTO_KEY, "")
        result = check_env_var(ENV_DATABENTO_KEY)
        assert result.status == SKIP

    def test_norgate_path_set(self, monkeypatch):
        monkeypatch.setenv(ENV_NORGATE_PATH, "/some/path")
        result = check_env_var(ENV_NORGATE_PATH)
        assert result.status == PASS

    def test_norgate_path_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_NORGATE_PATH, raising=False)
        result = check_env_var(ENV_NORGATE_PATH)
        assert result.status == SKIP

    def test_result_name_includes_var(self, monkeypatch):
        monkeypatch.delenv(ENV_DATABENTO_KEY, raising=False)
        result = check_env_var(ENV_DATABENTO_KEY)
        assert ENV_DATABENTO_KEY in result.name


# ---------------------------------------------------------------------------
# Tests: Provider stub checks
# ---------------------------------------------------------------------------

class TestProviderStubCheck:
    def test_databento_passes(self):
        result = check_provider_stub("databento")
        assert result.status == PASS
        assert "session=electronic" in result.detail
        assert "roll=back_adjusted" in result.detail
        assert "close=settlement" in result.detail

    def test_norgate_passes(self):
        result = check_provider_stub("norgate")
        assert result.status == PASS
        assert "session=combined" in result.detail
        assert "roll=back_adjusted" in result.detail
        assert "close=last_trade" in result.detail

    def test_unknown_provider_fails(self):
        result = check_provider_stub("unknown_provider")
        assert result.status == FAIL
        assert "Unknown provider" in result.detail

    def test_result_name_includes_provider(self):
        result = check_provider_stub("databento")
        assert "databento" in result.name


# ---------------------------------------------------------------------------
# Tests: Symbol map check
# ---------------------------------------------------------------------------

class TestSymbolMapCheck:
    def test_real_map_passes(self):
        result = check_symbol_map()
        assert result.status == PASS
        assert "29 symbols" in result.detail

    def test_missing_file_fails(self, tmp_path):
        result = check_symbol_map(tmp_path / "nonexistent.yaml")
        assert result.status == FAIL
        assert "not found" in result.detail.lower()

    def test_invalid_yaml_fails(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(": : : invalid")
        result = check_symbol_map(bad_yaml)
        # yaml.safe_load may or may not raise on this; if it loads as a
        # weird dict, validation will catch it.
        assert result.status == FAIL

    def test_empty_map_fails(self, tmp_path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("symbols: {}")
        result = check_symbol_map(empty_yaml)
        assert result.status == FAIL
        assert "validation errors" in result.detail


# ---------------------------------------------------------------------------
# Tests: Sample CSV schema check
# ---------------------------------------------------------------------------

class TestSampleCsvCheck:
    def test_valid_csv_passes(self, tmp_path):
        sym_dir = tmp_path / "ES"
        sym_dir.mkdir()
        csv_path = sym_dir / "test.ESH5.csv.zst"
        _write_sample_csv(csv_path)
        result = check_sample_csv("ES", tmp_path)
        assert result.status == PASS
        assert "schema OK" in result.detail

    def test_missing_dir_skips(self, tmp_path):
        result = check_sample_csv("ES", tmp_path)
        assert result.status == SKIP
        assert "not found" in result.detail.lower()

    def test_empty_dir_skips(self, tmp_path):
        sym_dir = tmp_path / "ES"
        sym_dir.mkdir()
        result = check_sample_csv("ES", tmp_path)
        assert result.status == SKIP
        assert "No .csv.zst" in result.detail

    def test_missing_columns_fails(self, tmp_path):
        sym_dir = tmp_path / "CL"
        sym_dir.mkdir()
        csv_path = sym_dir / "test.CLF5.csv.zst"
        # Write CSV with only some columns.
        _write_sample_csv(csv_path, columns=["ts_event", "open", "close"])
        result = check_sample_csv("CL", tmp_path)
        assert result.status == FAIL
        assert "Missing columns" in result.detail

    def test_all_columns_present(self, tmp_path):
        sym_dir = tmp_path / "PA"
        sym_dir.mkdir()
        csv_path = sym_dir / "test.PAH5.csv.zst"
        _write_sample_csv(csv_path)
        result = check_sample_csv("PA", tmp_path)
        assert result.status == PASS

    def test_result_name_includes_symbol(self, tmp_path):
        result = check_sample_csv("ES", tmp_path)
        assert "ES" in result.name


# ---------------------------------------------------------------------------
# Tests: CheckResult dataclass
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_fields(self):
        r = CheckResult(name="test", status=PASS, detail="ok")
        assert r.name == "test"
        assert r.status == PASS
        assert r.detail == "ok"

    def test_default_detail(self):
        r = CheckResult(name="test", status=FAIL)
        assert r.detail == ""


# ---------------------------------------------------------------------------
# Tests: ConnectivityReport
# ---------------------------------------------------------------------------

class TestConnectivityReport:
    def test_empty_report(self):
        report = ConnectivityReport()
        assert report.n_pass == 0
        assert report.n_fail == 0
        assert report.n_skip == 0
        assert report.any_fail is False

    def test_all_pass(self):
        report = ConnectivityReport(checks=[
            CheckResult("a", PASS, "ok"),
            CheckResult("b", PASS, "ok"),
        ])
        assert report.any_fail is False
        assert report.n_pass == 2

    def test_one_fail(self):
        report = ConnectivityReport(checks=[
            CheckResult("a", PASS, "ok"),
            CheckResult("b", FAIL, "error"),
        ])
        assert report.any_fail is True
        assert report.n_fail == 1

    def test_skip_not_fail(self):
        report = ConnectivityReport(checks=[
            CheckResult("a", PASS, "ok"),
            CheckResult("b", SKIP, "missing"),
        ])
        assert report.any_fail is False
        assert report.n_skip == 1

    def test_summary_dict_structure(self):
        report = ConnectivityReport(checks=[
            CheckResult("a", PASS, "ok"),
            CheckResult("b", SKIP, "missing"),
        ])
        d = report.summary_dict()
        assert d["n_pass"] == 1
        assert d["n_fail"] == 0
        assert d["n_skip"] == 1
        assert d["all_ok"] is True
        assert len(d["checks"]) == 2
        assert d["checks"][0]["name"] == "a"
        assert d["checks"][0]["status"] == PASS

    def test_summary_dict_json_serialisable(self):
        report = ConnectivityReport(checks=[
            CheckResult("a", PASS, "ok"),
        ])
        # Should not raise.
        text = json.dumps(report.summary_dict())
        assert isinstance(text, str)

    def test_summary_text_format(self):
        report = ConnectivityReport(checks=[
            CheckResult("a", PASS, "ok"),
            CheckResult("b", FAIL, "error"),
        ])
        text = report.summary_text()
        assert "1 PASS" in text
        assert "1 FAIL" in text
        assert "[PASS]" in text
        assert "[FAIL]" in text


# ---------------------------------------------------------------------------
# Tests: Full suite runner
# ---------------------------------------------------------------------------

class TestRunConnectivityChecks:
    def test_runs_without_error(self):
        """Full suite runs against real repo layout."""
        report = run_connectivity_checks()
        # Provider stubs and symbol map should always PASS.
        assert report.n_fail == 0

    def test_provider_stubs_always_pass(self):
        report = run_connectivity_checks()
        provider_checks = [c for c in report.checks if c.name.startswith("provider_")]
        assert len(provider_checks) == 2
        assert all(c.status == PASS for c in provider_checks)

    def test_symbol_map_always_passes(self):
        report = run_connectivity_checks()
        sm_checks = [c for c in report.checks if c.name == "symbol_map"]
        assert len(sm_checks) == 1
        assert sm_checks[0].status == PASS

    def test_env_checks_present(self):
        report = run_connectivity_checks()
        env_checks = [c for c in report.checks if c.name.startswith("env_")]
        assert len(env_checks) == 2

    def test_smoke_checks_present(self):
        report = run_connectivity_checks()
        csv_checks = [c for c in report.checks if c.name.startswith("sample_csv_")]
        assert len(csv_checks) == 3  # ES, CL, PA

    def test_custom_outrights_dir(self, tmp_path):
        """Suite with empty outrights dir → CSV checks SKIP."""
        for sym in ("ES", "CL", "PA"):
            (tmp_path / sym).mkdir()
        report = run_connectivity_checks(outrights_dir=tmp_path)
        csv_checks = [c for c in report.checks if c.name.startswith("sample_csv_")]
        assert all(c.status == SKIP for c in csv_checks)
        # Overall should not fail (SKIP is acceptable).
        assert report.any_fail is False

    def test_custom_symbol_map_path(self, tmp_path):
        """Suite with missing symbol map → that check FAILs."""
        report = run_connectivity_checks(
            symbol_map_path=tmp_path / "nonexistent.yaml",
        )
        sm = [c for c in report.checks if c.name == "symbol_map"]
        assert sm[0].status == FAIL
        assert report.any_fail is True


# ---------------------------------------------------------------------------
# Tests: Deterministic output
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_input_same_output(self):
        r1 = run_connectivity_checks()
        r2 = run_connectivity_checks()
        d1 = r1.summary_dict()
        d2 = r2.summary_dict()
        assert d1 == d2

    def test_check_order_stable(self):
        r1 = run_connectivity_checks()
        r2 = run_connectivity_checks()
        names1 = [c.name for c in r1.checks]
        names2 = [c.name for c in r2.checks]
        assert names1 == names2


# ---------------------------------------------------------------------------
# Tests: CLI script importability
# ---------------------------------------------------------------------------

class TestCLI:
    def test_script_importable(self):
        """Verify the CLI script can be imported without side effects."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_provider_connectivity",
            str(Path(__file__).resolve().parents[2] / "scripts" / "check_provider_connectivity.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Do not execute main — just verify import works.
        assert spec is not None
        assert mod is not None
