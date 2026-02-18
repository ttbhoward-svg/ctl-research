"""Unit tests for parity prep and overlap validator (Data Cutover Task G)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.parity_prep import (
    FAIL,
    INCOMPLETE,
    MIN_OVERLAP_BARS,
    PASS,
    REQUIRED_COLUMNS,
    ParityPrepReport,
    SymbolPrepResult,
    compute_overlap,
    discover_db_file,
    discover_ts_file,
    load_and_validate,
    prep_symbol,
    run_parity_prep,
    save_prep_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_csv(
    path: Path,
    start: str = "2019-01-02",
    n_bars: int = 600,
    base: float = 100.0,
    extra_cols: dict = None,
    lowercase: bool = False,
):
    """Write a synthetic OHLCV CSV file."""
    dates = pd.bdate_range(start, periods=n_bars)
    close = base + np.arange(n_bars, dtype=float) * 0.1
    df = pd.DataFrame({
        "Date": dates,
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.full(n_bars, 50000.0),
    })
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    if lowercase:
        df.columns = [c.lower() for c in df.columns]
    df.to_csv(path, index=False)


def _setup_both_files(
    tmp_path: Path,
    symbol: str = "ES",
    db_bars: int = 700,
    ts_bars: int = 700,
    db_start: str = "2019-01-02",
    ts_start: str = "2019-01-02",
):
    """Create matching DB and TS files in tmp_path and return (db_dir, ts_dir)."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()

    _make_ohlcv_csv(
        db_dir / f"{symbol}_continuous.csv",
        start=db_start,
        n_bars=db_bars,
        extra_cols={"contract": "ESH5", "adjustment": 0.0},
    )
    _make_ohlcv_csv(
        ts_dir / f"TS_{symbol}_1D_20190102_20220101.csv",
        start=ts_start,
        n_bars=ts_bars,
    )
    return db_dir, ts_dir


# ---------------------------------------------------------------------------
# Tests: File discovery — Databento
# ---------------------------------------------------------------------------

class TestDiscoverDbFile:
    def test_found(self, tmp_path):
        f = tmp_path / "ES_continuous.csv"
        f.write_text("Date,Open,High,Low,Close,Volume\n")
        assert discover_db_file("ES", tmp_path) == f

    def test_not_found(self, tmp_path):
        assert discover_db_file("ES", tmp_path) is None

    def test_wrong_symbol_not_found(self, tmp_path):
        (tmp_path / "CL_continuous.csv").write_text("x\n")
        assert discover_db_file("ES", tmp_path) is None


# ---------------------------------------------------------------------------
# Tests: File discovery — TradeStation
# ---------------------------------------------------------------------------

class TestDiscoverTsFile:
    def test_glob_match(self, tmp_path):
        f = tmp_path / "TS_ES_1D_20190101_20220101.csv"
        f.write_text("Date,Open,High,Low,Close,Volume\n")
        assert discover_ts_file("ES", tmp_path) == f

    def test_exact_match(self, tmp_path):
        f = tmp_path / "TS_ES_1D.csv"
        f.write_text("x\n")
        assert discover_ts_file("ES", tmp_path) == f

    def test_not_found(self, tmp_path):
        assert discover_ts_file("ES", tmp_path) is None

    def test_multiple_matches_uses_first_sorted(self, tmp_path):
        f1 = tmp_path / "TS_ES_1D_a.csv"
        f2 = tmp_path / "TS_ES_1D_b.csv"
        f1.write_text("x\n")
        f2.write_text("x\n")
        assert discover_ts_file("ES", tmp_path) == f1

    def test_wrong_symbol_not_found(self, tmp_path):
        (tmp_path / "TS_CL_1D_foo.csv").write_text("x\n")
        assert discover_ts_file("ES", tmp_path) is None


# ---------------------------------------------------------------------------
# Tests: Schema validation & loading
# ---------------------------------------------------------------------------

class TestLoadAndValidate:
    def test_valid_csv(self, tmp_path):
        f = tmp_path / "test.csv"
        _make_ohlcv_csv(f, n_bars=10)
        df, errors = load_and_validate(f, "test")
        assert errors == []
        assert len(df) == 10
        assert "Date" in df.columns

    def test_lowercase_columns_normalised(self, tmp_path):
        f = tmp_path / "test.csv"
        _make_ohlcv_csv(f, n_bars=10, lowercase=True)
        df, errors = load_and_validate(f, "test")
        assert errors == []
        assert "Date" in df.columns

    def test_missing_column_fails(self, tmp_path):
        f = tmp_path / "test.csv"
        pd.DataFrame({"Date": ["2024-01-02"], "Open": [100]}).to_csv(f, index=False)
        df, errors = load_and_validate(f, "test")
        assert df is None
        assert any("missing columns" in e.lower() for e in errors)

    def test_bad_date_fails(self, tmp_path):
        f = tmp_path / "test.csv"
        pd.DataFrame({
            "Date": ["not-a-date"],
            "Open": [1], "High": [2], "Low": [0], "Close": [1], "Volume": [100],
        }).to_csv(f, index=False)
        df, errors = load_and_validate(f, "test")
        assert df is None
        assert any("date parse" in e.lower() for e in errors)

    def test_read_error(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("")  # empty file
        df, errors = load_and_validate(f, "test")
        assert df is None
        assert len(errors) >= 1

    def test_deduplicates_dates(self, tmp_path):
        f = tmp_path / "test.csv"
        df = pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [103, 104, 105],
            "Volume": [1000, 2000, 3000],
        })
        df.to_csv(f, index=False)
        result, errors = load_and_validate(f, "test")
        assert errors == []
        assert len(result) == 2  # deduplicated

    def test_sorted_by_date(self, tmp_path):
        f = tmp_path / "test.csv"
        df = pd.DataFrame({
            "Date": ["2024-01-05", "2024-01-02", "2024-01-03"],
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [103, 104, 105],
            "Volume": [1000, 2000, 3000],
        })
        df.to_csv(f, index=False)
        result, errors = load_and_validate(f, "test")
        assert errors == []
        dates = result["Date"].tolist()
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Tests: Overlap computation
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def test_full_overlap(self):
        dates = pd.bdate_range("2024-01-02", periods=10)
        df1 = pd.DataFrame({"Date": dates})
        df2 = pd.DataFrame({"Date": dates})
        result = compute_overlap(df1, df2)
        assert result["overlap_bar_count"] == 10
        assert result["first_common_date"] == str(dates[0].date())
        assert result["last_common_date"] == str(dates[-1].date())

    def test_no_overlap(self):
        df1 = pd.DataFrame({"Date": pd.bdate_range("2020-01-02", periods=10)})
        df2 = pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=10)})
        result = compute_overlap(df1, df2)
        assert result["overlap_bar_count"] == 0
        assert result["first_common_date"] is None
        assert result["last_common_date"] is None

    def test_partial_overlap(self):
        df1 = pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=20)})
        df2 = pd.DataFrame({"Date": pd.bdate_range("2024-01-15", periods=20)})
        result = compute_overlap(df1, df2)
        assert result["overlap_bar_count"] > 0
        assert result["overlap_bar_count"] < 20

    def test_overlap_count_excludes_unique_dates(self):
        common_dates = pd.bdate_range("2024-03-01", periods=5)
        df1 = pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=50)})
        df2 = pd.DataFrame({"Date": common_dates})
        result = compute_overlap(df1, df2)
        assert result["overlap_bar_count"] == 5


# ---------------------------------------------------------------------------
# Tests: Per-symbol prep
# ---------------------------------------------------------------------------

class TestPrepSymbol:
    def test_both_files_sufficient_overlap(self, tmp_path):
        db_dir, ts_dir = _setup_both_files(tmp_path, db_bars=700, ts_bars=700)
        result = prep_symbol("ES", db_dir, ts_dir, min_overlap=500)
        assert result.status == PASS
        assert result.overlap_bar_count >= 500
        assert result.db_file == "ES_continuous.csv"
        assert result.ts_file is not None
        assert result.errors == []

    def test_insufficient_overlap_fails(self, tmp_path):
        db_dir, ts_dir = _setup_both_files(
            tmp_path, db_bars=100, ts_bars=100,
        )
        result = prep_symbol("ES", db_dir, ts_dir, min_overlap=500)
        assert result.status == FAIL
        assert result.overlap_bar_count < 500
        assert any("< 500" in e for e in result.errors)

    def test_missing_db_file_incomplete(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_ohlcv_csv(ts_dir / "TS_ES_1D_foo.csv", n_bars=10)
        result = prep_symbol("ES", db_dir, ts_dir)
        assert result.status == INCOMPLETE
        assert any("Databento file not found" in e for e in result.errors)

    def test_missing_ts_file_incomplete(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_ohlcv_csv(
            db_dir / "ES_continuous.csv",
            n_bars=100,
            extra_cols={"contract": "ESH5", "adjustment": 0.0},
        )
        result = prep_symbol("ES", db_dir, ts_dir)
        assert result.status == INCOMPLETE
        assert any("TradeStation file not found" in e for e in result.errors)

    def test_both_files_missing_incomplete(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        result = prep_symbol("ES", db_dir, ts_dir)
        assert result.status == INCOMPLETE

    def test_db_bad_schema_fails(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        # DB file with missing columns.
        pd.DataFrame({"Date": ["2024-01-02"], "Foo": [1]}).to_csv(
            db_dir / "ES_continuous.csv", index=False
        )
        _make_ohlcv_csv(ts_dir / "TS_ES_1D_foo.csv", n_bars=10)
        result = prep_symbol("ES", db_dir, ts_dir)
        assert result.status == FAIL
        assert any("missing columns" in e.lower() for e in result.errors)

    def test_ts_bad_schema_fails(self, tmp_path):
        db_dir, ts_dir = _setup_both_files(tmp_path, db_bars=10, ts_bars=10)
        # Overwrite TS file with bad schema.
        pd.DataFrame({"Date": ["2024-01-02"], "Foo": [1]}).to_csv(
            ts_dir / "TS_ES_1D_20190102_20220101.csv", index=False
        )
        result = prep_symbol("ES", db_dir, ts_dir)
        assert result.status == FAIL

    def test_date_range_reported(self, tmp_path):
        db_dir, ts_dir = _setup_both_files(
            tmp_path, db_bars=50, ts_bars=50,
        )
        result = prep_symbol("ES", db_dir, ts_dir, min_overlap=10)
        assert result.db_first_date is not None
        assert result.db_last_date is not None
        assert result.ts_first_date is not None
        assert result.ts_last_date is not None

    def test_non_overlapping_dates_fail(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_ohlcv_csv(
            db_dir / "ES_continuous.csv",
            start="2023-01-02", n_bars=100,
            extra_cols={"contract": "ESH5", "adjustment": 0.0},
        )
        _make_ohlcv_csv(
            ts_dir / "TS_ES_1D_foo.csv",
            start="2019-01-02", n_bars=100,
        )
        result = prep_symbol("ES", db_dir, ts_dir, min_overlap=10)
        assert result.status == FAIL
        assert result.overlap_bar_count == 0


# ---------------------------------------------------------------------------
# Tests: SymbolPrepResult
# ---------------------------------------------------------------------------

class TestSymbolPrepResult:
    def test_to_dict(self):
        r = SymbolPrepResult(symbol="ES", status=PASS)
        d = r.to_dict()
        assert d["symbol"] == "ES"
        assert d["status"] == PASS
        assert "overlap_bar_count" in d
        assert "errors" in d

    def test_default_values(self):
        r = SymbolPrepResult(symbol="CL", status=INCOMPLETE)
        assert r.db_bars == 0
        assert r.ts_bars == 0
        assert r.overlap_bar_count == 0
        assert r.errors == []


# ---------------------------------------------------------------------------
# Tests: ParityPrepReport
# ---------------------------------------------------------------------------

class TestParityPrepReport:
    def test_all_pass(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
            SymbolPrepResult("CL", PASS, overlap_bar_count=700),
        ])
        assert report.gate_status == PASS

    def test_one_incomplete(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
            SymbolPrepResult("CL", INCOMPLETE),
        ])
        assert report.gate_status == INCOMPLETE

    def test_one_fail(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
            SymbolPrepResult("CL", FAIL, overlap_bar_count=100),
        ])
        assert report.gate_status == FAIL

    def test_fail_overrides_incomplete(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", FAIL, overlap_bar_count=10),
            SymbolPrepResult("CL", INCOMPLETE),
        ])
        assert report.gate_status == FAIL

    def test_empty_report_passes(self):
        report = ParityPrepReport()
        assert report.gate_status == PASS

    def test_summary_dict_structure(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
        ])
        d = report.summary_dict()
        assert "gate_status" in d
        assert "min_overlap_bars" in d
        assert "symbols" in d
        assert len(d["symbols"]) == 1

    def test_summary_dict_json_serialisable(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
        ])
        text = json.dumps(report.summary_dict())
        assert isinstance(text, str)

    def test_summary_text_format(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
            SymbolPrepResult("CL", INCOMPLETE, errors=["TS file missing"]),
        ])
        text = report.summary_text()
        assert "PASS" in text
        assert "INCOMPLETE" in text
        assert "TS file missing" in text

    def test_to_csv_df(self):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
            SymbolPrepResult("CL", FAIL, overlap_bar_count=10, errors=["low overlap"]),
        ])
        df = report.to_csv_df()
        assert len(df) == 2
        assert "symbol" in df.columns
        assert "status" in df.columns
        assert "errors" in df.columns
        # errors should be string in CSV.
        assert df.loc[df["symbol"] == "CL", "errors"].iloc[0] == "low overlap"

    def test_to_csv_df_empty(self):
        report = ParityPrepReport()
        df = report.to_csv_df()
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: Full suite runner
# ---------------------------------------------------------------------------

class TestRunParityPrep:
    def test_all_incomplete_when_no_files(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        report = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir, report_dir=tmp_path / "out",
            symbols=("ES", "CL", "PA"), save=False,
        )
        assert report.gate_status == INCOMPLETE
        assert all(s.status == INCOMPLETE for s in report.symbols)

    def test_all_pass_with_synthetic_data(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        for sym in ("ES", "CL", "PA"):
            _make_ohlcv_csv(
                db_dir / f"{sym}_continuous.csv",
                start="2019-01-02", n_bars=700,
                extra_cols={"contract": f"{sym}H5", "adjustment": 0.0},
            )
            _make_ohlcv_csv(
                ts_dir / f"TS_{sym}_1D_test.csv",
                start="2019-01-02", n_bars=700,
            )
        report = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir,
            report_dir=tmp_path / "out", save=False,
        )
        assert report.gate_status == PASS
        assert all(s.status == PASS for s in report.symbols)

    def test_partial_incomplete(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        # Only ES has both files.
        _make_ohlcv_csv(
            db_dir / "ES_continuous.csv",
            start="2019-01-02", n_bars=700,
            extra_cols={"contract": "ESH5", "adjustment": 0.0},
        )
        _make_ohlcv_csv(
            ts_dir / "TS_ES_1D_test.csv",
            start="2019-01-02", n_bars=700,
        )
        report = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir,
            report_dir=tmp_path / "out", symbols=("ES", "CL"), save=False,
        )
        es = next(s for s in report.symbols if s.symbol == "ES")
        cl = next(s for s in report.symbols if s.symbol == "CL")
        assert es.status == PASS
        assert cl.status == INCOMPLETE
        assert report.gate_status == INCOMPLETE

    def test_custom_min_overlap(self, tmp_path):
        db_dir, ts_dir = _setup_both_files(tmp_path, db_bars=50, ts_bars=50)
        report = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir,
            report_dir=tmp_path / "out",
            symbols=("ES",), min_overlap=10, save=False,
        )
        assert report.gate_status == PASS

    def test_three_symbols_default(self, tmp_path):
        report = run_parity_prep(
            db_dir=tmp_path, ts_dir=tmp_path,
            report_dir=tmp_path / "out",
            save=False,
        )
        assert len(report.symbols) == 3
        syms = {s.symbol for s in report.symbols}
        assert syms == {"ES", "CL", "PA"}


# ---------------------------------------------------------------------------
# Tests: Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_files(self, tmp_path):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
        ])
        paths = save_prep_report(report, tmp_path)
        assert paths["json_path"].exists()
        assert paths["csv_path"].exists()

    def test_json_content(self, tmp_path):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
        ])
        paths = save_prep_report(report, tmp_path)
        with open(paths["json_path"]) as f:
            data = json.load(f)
        assert data["gate_status"] == PASS
        assert len(data["symbols"]) == 1

    def test_csv_readable(self, tmp_path):
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", PASS, overlap_bar_count=600),
            SymbolPrepResult("CL", FAIL, overlap_bar_count=10),
        ])
        paths = save_prep_report(report, tmp_path)
        df = pd.read_csv(paths["csv_path"])
        assert len(df) == 2
        assert "symbol" in df.columns

    def test_save_creates_directory(self, tmp_path):
        out_dir = tmp_path / "nested" / "output"
        report = ParityPrepReport(symbols=[
            SymbolPrepResult("ES", INCOMPLETE),
        ])
        paths = save_prep_report(report, out_dir)
        assert out_dir.exists()
        assert paths["json_path"].exists()

    def test_run_with_save(self, tmp_path):
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        out_dir = tmp_path / "out"
        report = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir,
            report_dir=out_dir, symbols=("ES",), save=True,
        )
        assert (out_dir / "parity_prep_report.json").exists()
        assert (out_dir / "parity_prep_report.csv").exists()


# ---------------------------------------------------------------------------
# Tests: Deterministic output
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_input_same_output(self, tmp_path):
        db_dir, ts_dir = _setup_both_files(tmp_path, db_bars=700, ts_bars=700)
        r1 = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir,
            report_dir=tmp_path / "out1", symbols=("ES",), save=False,
        )
        r2 = run_parity_prep(
            db_dir=db_dir, ts_dir=ts_dir,
            report_dir=tmp_path / "out2", symbols=("ES",), save=False,
        )
        d1 = r1.summary_dict()
        d2 = r2.summary_dict()
        assert d1 == d2
