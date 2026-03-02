"""Unit tests for provider manifest loader and availability checks."""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.provider_manifest import (  # noqa: E402
    evaluate_manifest_availability,
    load_provider_manifest,
    validate_provider_manifest,
)


def _write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


class TestProviderManifest:
    def test_load_and_validate_real_manifest(self):
        manifest = load_provider_manifest()
        errors = validate_provider_manifest(manifest)
        assert errors == []
        assert len(manifest.symbols) == 29

    def test_validate_missing_symbol_error(self, tmp_path):
        p = tmp_path / "manifest.yaml"
        _write_manifest(
            p,
            {
                "cycle_id": "cutover_v1",
                "version": "x",
                "symbols": {
                    "/ES": {
                        "primary": "databento",
                        "fallback": "tradestation",
                        "reference": "norgate",
                    }
                },
            },
        )
        manifest = load_provider_manifest(p)
        errors = validate_provider_manifest(manifest, expected_symbols=["/ES", "/CL"])
        assert any("missing symbols" in e for e in errors)

    def test_validate_invalid_provider(self, tmp_path):
        p = tmp_path / "manifest.yaml"
        _write_manifest(
            p,
            {
                "cycle_id": "cutover_v1",
                "version": "x",
                "symbols": {
                    "/ES": {
                        "primary": "bad_provider",
                        "fallback": "tradestation",
                        "reference": "norgate",
                    }
                },
            },
        )
        manifest = load_provider_manifest(p)
        errors = validate_provider_manifest(manifest, expected_symbols=["/ES"])
        assert any("invalid primary provider" in e for e in errors)

    def test_availability_eval(self, tmp_path):
        p = tmp_path / "manifest.yaml"
        _write_manifest(
            p,
            {
                "cycle_id": "cutover_v1",
                "version": "x",
                "symbols": {
                    "/ES": {
                        "primary": "databento",
                        "fallback": "tradestation",
                        "reference": "norgate",
                    },
                    "AAPL": {
                        "primary": "yfinance",
                        "fallback": "none",
                        "reference": "none",
                    },
                },
            },
        )
        db_dir = tmp_path / "db"
        ts_dir = tmp_path / "ts"
        ng_dir = tmp_path / "ng"
        db_dir.mkdir()
        ts_dir.mkdir()
        ng_dir.mkdir()

        # /ES: available across all providers.
        (db_dir / "ES_continuous.csv").write_text("Date,Open,High,Low,Close,Volume\n")
        (ts_dir / "TS_ES_1D_20260101.csv").write_text("Date,Open,High,Low,Close,Volume\n")
        (ng_dir / "NG_ES_1D_20260101.csv").write_text("Date,Open,High,Low,Close,Volume\n")
        # AAPL (yfinance path reuses continuous location convention).
        (db_dir / "AAPL_continuous.csv").write_text("Date,Open,High,Low,Close,Volume\n")

        manifest = load_provider_manifest(p)
        rows = evaluate_manifest_availability(
            manifest,
            db_continuous_dir=db_dir,
            ts_dir=ts_dir,
            norgate_dir=ng_dir,
        )
        row_map = {r.symbol: r for r in rows}
        assert row_map["/ES"].primary_available is True
        assert row_map["/ES"].fallback_available is True
        assert row_map["/ES"].reference_available is True
        assert row_map["AAPL"].primary_available is True
