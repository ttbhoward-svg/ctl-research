"""Tests for research batch runner helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.research_batch import run_research_batch  # noqa: E402
from ctl.research_registry import ResearchSymbol, ResearchTickerRegistry  # noqa: E402


def test_run_research_batch_dry_run():
    reg = ResearchTickerRegistry(
        cycle_id="cutover_v1",
        registry_id="r1",
        profile_path="configs/cutover/operating_profile_v1.yaml",
        symbols=[
            ResearchSymbol(symbol="PA", enabled=True),
            ResearchSymbol(symbol="XLE", enabled=True),
            ResearchSymbol(symbol="AAPL", enabled=False),
        ],
    )
    out = run_research_batch(reg, dry_run=True)
    assert out.dry_run is True
    assert out.symbols == ["PA", "XLE"]
    assert [r["status"] for r in out.symbol_results] == ["DRY_RUN", "DRY_RUN"]
