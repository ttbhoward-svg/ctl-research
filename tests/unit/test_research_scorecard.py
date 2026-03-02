"""Tests for research confidence scorecard."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.research_registry import ResearchSymbol, ResearchTickerRegistry  # noqa: E402
from ctl.research_scorecard import build_research_scorecard  # noqa: E402


def _registry() -> ResearchTickerRegistry:
    return ResearchTickerRegistry(
        cycle_id="cutover_v1",
        registry_id="r1",
        profile_path="configs/cutover/operating_profile_v1.yaml",
        symbols=[
            ResearchSymbol("PA", enabled=True, tick_size=0.1, max_day_delta=3),
            ResearchSymbol("AAPL", enabled=True),
        ],
    )


def test_build_research_scorecard_basic():
    batch = {
        "summary": {
            "symbol_results": [
                {"symbol": "PA", "status": "EXECUTED", "trade_count": 2, "total_r": 1.2, "win_rate": 1.0},
                {"symbol": "AAPL", "status": "ERROR"},
            ]
        }
    }
    rows = build_research_scorecard(_registry(), batch, db_dir=Path("/nope"), ts_dir=Path("/nope"))
    by = {r.symbol: r for r in rows}
    assert set(by.keys()) == {"PA", "AAPL"}
    assert by["AAPL"].diagnostics_available is False
    assert by["AAPL"].confidence_band == "LOW"
    assert by["PA"].run_status == "EXECUTED"


def test_missing_symbol_row_defaults_to_missing():
    batch = {"summary": {"symbol_results": []}}
    rows = build_research_scorecard(_registry(), batch, db_dir=Path("/nope"), ts_dir=Path("/nope"))
    by = {r.symbol: r for r in rows}
    assert by["PA"].run_status == "MISSING"
    assert by["AAPL"].run_status == "MISSING"
