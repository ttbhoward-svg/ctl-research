"""Run orchestrator — gate-first portfolio execution wiring (H.8).

Provides typed dataclasses and pure functions for building, executing,
and summarising a gated portfolio run.  The gate check re-derives
acceptance from live diagnostics (reusing the same functions as
``scripts/check_operating_profile.py``) and must pass before any
strategy execution proceeds.

Usage
-----
>>> from ctl.run_orchestrator import build_run_plan, run_profile_gate
>>> profile = load_operating_profile("configs/cutover/operating_profile_v1.yaml")
>>> gate = run_profile_gate(profile_path)
>>> plan = build_run_plan(profile, include_non_gating=False)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ctl.canonical_acceptance import acceptance_from_diagnostics
from ctl.cutover_diagnostics import run_diagnostics
from ctl.operating_profile import (
    OperatingProfile,
    PortfolioCheckResult,
    SymbolCheckResult,
    check_portfolio,
    check_symbol_status,
    discover_ts_custom_file,
    load_operating_profile,
)
from ctl.parity_prep import load_and_validate
from ctl.roll_reconciliation import load_roll_manifest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_CONTINUOUS_DIR = (
    REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
)
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"
DEFAULT_SUMMARY_DIR = (
    REPO_ROOT / "data" / "processed" / "cutover_v1" / "run_summaries"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPlan:
    """Describes which symbols will be executed and under what profile."""

    cycle_id: str
    profile_path: str
    symbols: List[str]
    gating_symbols: List[str]
    non_gating_symbols: List[str]
    include_non_gating: bool
    portfolio_recommendation: str

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "profile_path": self.profile_path,
            "symbols": list(self.symbols),
            "gating_symbols": list(self.gating_symbols),
            "non_gating_symbols": list(self.non_gating_symbols),
            "include_non_gating": self.include_non_gating,
            "portfolio_recommendation": self.portfolio_recommendation,
        }


@dataclass
class SymbolRunResult:
    """Outcome of executing a strategy for one symbol."""

    symbol: str
    status: str  # "EXECUTED", "SKIPPED", "ERROR", "DRY_RUN"
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class RunSummary:
    """Aggregate run output — gate result, plan, per-symbol outcomes."""

    timestamp: str
    cycle_id: str
    gate_passed: bool
    portfolio_recommendation: str
    dry_run: bool
    plan: dict
    gate_result: dict
    symbol_run_results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "gate_passed": self.gate_passed,
            "portfolio_recommendation": self.portfolio_recommendation,
            "dry_run": self.dry_run,
            "plan": dict(self.plan),
            "gate_result": dict(self.gate_result),
            "symbol_run_results": list(self.symbol_run_results),
        }


# ---------------------------------------------------------------------------
# Gate check (reuses same functions as check_operating_profile.py)
# ---------------------------------------------------------------------------

def run_profile_gate(
    profile_path: Path,
    db_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    ts_dir: Path = DEFAULT_TS_DIR,
) -> PortfolioCheckResult:
    """Execute the operating-profile gate check.

    Re-derives canonical acceptance for each gating symbol and compares
    against the locked expected statuses.  This is the same logic as
    ``scripts/check_operating_profile._run_gate`` but parameterised for
    testability.

    Parameters
    ----------
    profile_path : Path
        Path to the operating profile YAML.
    db_dir : Path
        Databento continuous CSV directory.
    ts_dir : Path
        TradeStation reference CSV directory.

    Returns
    -------
    PortfolioCheckResult
    """
    profile = load_operating_profile(profile_path)
    symbol_results: List[SymbolCheckResult] = []

    for sym in profile.gating_universe:
        settings = profile.symbol_settings.get(sym)
        if settings is None:
            symbol_results.append(SymbolCheckResult(
                symbol=sym, expected="(missing)", actual="ERROR",
                passed=False, detail=f"No symbol_settings entry for {sym}",
            ))
            continue

        # Load canonical continuous.
        canonical_path = db_dir / f"{sym}_continuous.csv"
        canonical_df, can_errors = load_and_validate(canonical_path, f"DB {sym}")
        if can_errors:
            symbol_results.append(SymbolCheckResult(
                symbol=sym, expected=settings.expected_status, actual="ERROR",
                passed=False, detail=f"Load error: {'; '.join(can_errors)}",
            ))
            continue

        # Load manifest.
        manifest_path = db_dir / f"{sym}_roll_manifest.json"
        manifest_entries = load_roll_manifest(manifest_path)

        # Load TS adj/unadj.
        ts_adj_path = discover_ts_custom_file(sym, ts_dir, "ADJ")
        ts_unadj_path = discover_ts_custom_file(sym, ts_dir, "UNADJ")

        ts_adj_df = None
        ts_unadj_df = None

        if ts_adj_path:
            ts_adj_df, adj_errs = load_and_validate(ts_adj_path, f"TS {sym} ADJ")
            if adj_errs:
                logger.warning("TS ADJ load issues for %s: %s", sym, adj_errs)
                ts_adj_df = None

        if ts_unadj_path:
            ts_unadj_df, unadj_errs = load_and_validate(
                ts_unadj_path, f"TS {sym} UNADJ",
            )
            if unadj_errs:
                logger.warning("TS UNADJ load issues for %s: %s", sym, unadj_errs)
                ts_unadj_df = None

        if ts_adj_df is None:
            symbol_results.append(SymbolCheckResult(
                symbol=sym, expected=settings.expected_status, actual="ERROR",
                passed=False, detail=f"TS ADJ file not found for {sym}",
            ))
            continue

        # Run diagnostics + acceptance.
        diag = run_diagnostics(
            canonical_adj_df=canonical_df,
            ts_adj_df=ts_adj_df,
            manifest_entries=manifest_entries,
            ts_unadj_df=ts_unadj_df,
            symbol=sym,
            tick_size=settings.tick_size,
            max_day_delta=settings.max_day_delta,
        )
        acceptance = acceptance_from_diagnostics(diag)
        result = check_symbol_status(sym, settings.expected_status, acceptance.decision)
        symbol_results.append(result)

    return check_portfolio(profile, symbol_results)


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

def build_run_plan(
    profile: OperatingProfile,
    include_non_gating: bool = False,
    profile_path: str = "",
) -> RunPlan:
    """Build a run plan from the operating profile.

    Parameters
    ----------
    profile : OperatingProfile
        Loaded operating profile.
    include_non_gating : bool
        Whether to include non-gating symbols (e.g. PA) in the run.
    profile_path : str
        Path string recorded in the plan for provenance.

    Returns
    -------
    RunPlan
    """
    gating = list(profile.gating_universe)
    non_gating = list(profile.non_gating_symbols) if include_non_gating else []
    symbols = gating + non_gating

    return RunPlan(
        cycle_id=profile.cycle_id,
        profile_path=profile_path,
        symbols=symbols,
        gating_symbols=gating,
        non_gating_symbols=non_gating,
        include_non_gating=include_non_gating,
        portfolio_recommendation=profile.portfolio_recommendation,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

#: Type alias for a symbol executor callback.
SymbolExecutor = Callable[[str], SymbolRunResult]


def _default_executor(symbol: str) -> SymbolRunResult:
    """Default no-op executor (placeholder for future strategy wiring)."""
    return SymbolRunResult(
        symbol=symbol,
        status="EXECUTED",
        detail="strategy execution placeholder",
    )


def execute_run_plan(
    plan: RunPlan,
    executor: Optional[SymbolExecutor] = None,
    dry_run: bool = False,
) -> List[SymbolRunResult]:
    """Execute the run plan for each symbol.

    Parameters
    ----------
    plan : RunPlan
        The run plan built from ``build_run_plan``.
    executor : callable, optional
        ``(symbol: str) -> SymbolRunResult``.  Defaults to a no-op
        placeholder executor.
    dry_run : bool
        If True, skip actual execution and return DRY_RUN status for
        each symbol.

    Returns
    -------
    list of SymbolRunResult
    """
    if executor is None:
        executor = _default_executor

    results: List[SymbolRunResult] = []
    for sym in plan.symbols:
        if dry_run:
            results.append(SymbolRunResult(
                symbol=sym, status="DRY_RUN", detail="gate passed, execution skipped",
            ))
        else:
            try:
                results.append(executor(sym))
            except Exception as exc:
                logger.error("Execution error for %s: %s", sym, exc)
                results.append(SymbolRunResult(
                    symbol=sym, status="ERROR", detail=str(exc),
                ))
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_run(
    plan: RunPlan,
    gate_result: PortfolioCheckResult,
    symbol_run_results: List[SymbolRunResult],
    dry_run: bool = False,
    timestamp: Optional[str] = None,
) -> RunSummary:
    """Package run outputs into a serialisable summary.

    Parameters
    ----------
    plan : RunPlan
        The run plan.
    gate_result : PortfolioCheckResult
        Gate check outcome.
    symbol_run_results : list of SymbolRunResult
        Per-symbol execution outcomes.
    dry_run : bool
        Whether this was a dry run.
    timestamp : str, optional
        ISO-format timestamp.  Defaults to current UTC time.

    Returns
    -------
    RunSummary
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return RunSummary(
        timestamp=timestamp,
        cycle_id=plan.cycle_id,
        gate_passed=gate_result.passed,
        portfolio_recommendation=plan.portfolio_recommendation,
        dry_run=dry_run,
        plan=plan.to_dict(),
        gate_result=gate_result.to_dict(),
        symbol_run_results=[r.to_dict() for r in symbol_run_results],
    )


def save_run_summary(
    summary: RunSummary,
    out_dir: Path = DEFAULT_SUMMARY_DIR,
) -> Path:
    """Write run summary JSON to disk.

    Parameters
    ----------
    summary : RunSummary
        The run summary to persist.
    out_dir : Path
        Output directory.

    Returns
    -------
    Path to the written JSON file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{summary.timestamp}_portfolio_run.json"
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    logger.info("Run summary saved to %s", path)
    return path
