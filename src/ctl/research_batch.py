"""Research-tier batch backtest runner helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ctl.research_registry import ResearchTickerRegistry
from ctl.run_orchestrator import (
    DEFAULT_DB_CONTINUOUS_DIR,
    SymbolRunResult,
    execute_b1_symbol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_OUT_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "research_runs"


@dataclass(frozen=True)
class ResearchBatchSummary:
    timestamp: str
    registry_id: str
    cycle_id: str
    dry_run: bool
    symbols: List[str]
    symbol_results: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def run_research_batch(
    registry: ResearchTickerRegistry,
    data_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    dry_run: bool = False,
    symbols_override: Optional[List[str]] = None,
) -> ResearchBatchSummary:
    """Run B1 batch for enabled research symbols."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    symbols = list(symbols_override) if symbols_override is not None else registry.enabled_symbols()
    results: List[SymbolRunResult] = []

    for sym in symbols:
        if dry_run:
            results.append(SymbolRunResult(symbol=sym, status="DRY_RUN", detail="execution skipped"))
            continue
        slip = registry.slippage_map().get(sym, 0.0)
        results.append(
            execute_b1_symbol(
                symbol=sym,
                data_dir=data_dir,
                timeframe="daily",
                slippage_per_side=float(slip),
            )
        )

    return ResearchBatchSummary(
        timestamp=ts,
        registry_id=registry.registry_id,
        cycle_id=registry.cycle_id,
        dry_run=dry_run,
        symbols=symbols,
        symbol_results=[r.to_dict() for r in results],
    )


def save_research_batch_summary(
    summary: ResearchBatchSummary,
    out_dir: Path = DEFAULT_RESEARCH_OUT_DIR,
) -> Path:
    """Persist research batch summary JSON."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary.timestamp}_research_batch.json"
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    return path
