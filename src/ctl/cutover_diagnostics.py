"""Cutover diagnostics orchestrator — L2/L3/L4 (Data Cutover Task H).

Runs layered diagnostics comparing Databento-built continuous series
against TradeStation reference data:

- **L2**: Roll schedule comparison (date alignment).
- **L3**: Roll gap comparison (magnitude alignment).
- **L4**: Adjusted series drift + drift explanation by roll interval.

Artifacts saved per symbol:
- ``{symbol}_L2_roll_schedule_comparison.csv``
- ``{symbol}_L3_roll_gap_comparison.csv``
- ``{symbol}_L4_adjusted_series_drift.csv``
- ``{symbol}_L4_drift_explanation.json``

See docs/notes/TaskH_assumptions.md for design rationale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ctl.roll_reconciliation import (
    DriftExplanationResult,
    RollComparisonResult,
    RollManifestEntry,
    TSRollEvent,
    _worst_status,
    compare_roll_schedules,
    derive_ts_roll_events_from_spread,
    derive_ts_roll_events_from_unadjusted,
    explain_step_changes,
    load_roll_manifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class L2Result:
    """L2: Roll schedule comparison output."""

    symbol: str
    comparison: RollComparisonResult
    detail_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def status(self) -> str:
        return self.comparison.status

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "n_canonical": self.comparison.n_canonical,
            "n_ts": self.comparison.n_ts,
            "n_matched": self.comparison.n_matched,
            "n_watch": self.comparison.n_watch,
            "n_fail": self.comparison.n_fail,
        }


@dataclass
class L3Result:
    """L3: Roll gap comparison output."""

    symbol: str
    n_compared: int = 0
    mean_gap_diff: float = 0.0
    max_gap_diff: float = 0.0
    detail_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_compared": self.n_compared,
            "mean_gap_diff": round(self.mean_gap_diff, 6),
            "max_gap_diff": round(self.max_gap_diff, 6),
        }


@dataclass
class L4Result:
    """L4: Adjusted series drift output."""

    symbol: str
    n_overlap: int = 0
    mean_drift: float = 0.0
    max_drift: float = 0.0
    drift_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    explanation: Optional[DriftExplanationResult] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_overlap": self.n_overlap,
            "mean_drift": round(self.mean_drift, 6),
            "max_drift": round(self.max_drift, 6),
            "n_explanation_intervals": (
                self.explanation.n_intervals if self.explanation else 0
            ),
        }


@dataclass
class DiagnosticResult:
    """Combined L2 + L3 + L4 diagnostic result for one symbol."""

    symbol: str
    l2: L2Result
    l3: L3Result
    l4: L4Result

    @property
    def status(self) -> str:
        return _worst_status([self.l2.status])

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "overall_status": self.status,
            "l2": self.l2.to_dict(),
            "l3": self.l3.to_dict(),
            "l4": self.l4.to_dict(),
        }


# ---------------------------------------------------------------------------
# L2: Roll schedule comparison
# ---------------------------------------------------------------------------

def run_l2(
    manifest_entries: List[RollManifestEntry],
    ts_rolls: List[TSRollEvent],
    symbol: str,
    max_day_delta: int = 2,
) -> L2Result:
    """Run L2 roll schedule comparison diagnostic.

    Parameters
    ----------
    manifest_entries : list of RollManifestEntry
        Canonical (Databento) roll events.
    ts_rolls : list of TSRollEvent
        TradeStation-inferred roll events.
    symbol : str
        Symbol label for reporting.
    max_day_delta : int
        Maximum calendar-day tolerance for matching.

    Returns
    -------
    L2Result
    """
    comparison = compare_roll_schedules(manifest_entries, ts_rolls, max_day_delta)
    detail_df = comparison.to_dataframe()
    return L2Result(symbol=symbol, comparison=comparison, detail_df=detail_df)


# ---------------------------------------------------------------------------
# L3: Roll gap comparison
# ---------------------------------------------------------------------------

def run_l3(
    l2_result: L2Result,
    symbol: str,
) -> L3Result:
    """Run L3 roll gap comparison diagnostic.

    Computes gap magnitude differences for matched/watch roll pairs
    from the L2 comparison.

    Parameters
    ----------
    l2_result : L2Result
        Output from ``run_l2``.
    symbol : str
        Symbol label for reporting.

    Returns
    -------
    L3Result
    """
    detail_df = l2_result.detail_df.copy()

    if detail_df.empty:
        return L3Result(symbol=symbol, detail_df=detail_df)

    # Only include matched/watch pairs where both gaps exist.
    mask = (
        detail_df["status"].isin(["PASS", "WATCH"])
        & detail_df["canonical_gap"].notna()
        & detail_df["ts_gap"].notna()
    )
    matched = detail_df.loc[mask].copy()

    if matched.empty:
        return L3Result(symbol=symbol, detail_df=detail_df)

    matched["gap_diff_abs"] = (matched["canonical_gap"] - matched["ts_gap"]).abs()
    mean_diff = float(matched["gap_diff_abs"].mean())
    max_diff = float(matched["gap_diff_abs"].max())

    return L3Result(
        symbol=symbol,
        n_compared=len(matched),
        mean_gap_diff=mean_diff,
        max_gap_diff=max_diff,
        detail_df=detail_df,
    )


# ---------------------------------------------------------------------------
# L4: Adjusted series drift
# ---------------------------------------------------------------------------

def run_l4(
    canonical_adj_df: pd.DataFrame,
    ts_adj_df: pd.DataFrame,
    roll_compare_df: pd.DataFrame,
    symbol: str,
) -> L4Result:
    """Run L4 adjusted series drift diagnostic.

    Parameters
    ----------
    canonical_adj_df : pd.DataFrame
        Databento adjusted continuous series with ``Date`` and ``Close``.
    ts_adj_df : pd.DataFrame
        TradeStation adjusted series with ``Date`` and ``Close``.
    roll_compare_df : pd.DataFrame
        Output of L2 comparison (``L2Result.detail_df``).
    symbol : str
        Symbol label for reporting.

    Returns
    -------
    L4Result
    """
    # Align on overlapping dates.
    can = canonical_adj_df.copy()
    can["Date"] = pd.to_datetime(can["Date"])
    can = can.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    ts = ts_adj_df.copy()
    ts["Date"] = pd.to_datetime(ts["Date"])
    ts = ts.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    merged = pd.merge(
        can[["Date", "Close"]].rename(columns={"Close": "close_can"}),
        ts[["Date", "Close"]].rename(columns={"Close": "close_ts"}),
        on="Date",
    ).sort_values("Date").reset_index(drop=True)

    if merged.empty:
        return L4Result(symbol=symbol)

    merged["drift"] = (merged["close_can"] - merged["close_ts"]).abs()

    # Run drift explanation.
    explanation = explain_step_changes(canonical_adj_df, ts_adj_df, roll_compare_df)
    explanation.symbol = symbol

    return L4Result(
        symbol=symbol,
        n_overlap=len(merged),
        mean_drift=float(merged["drift"].mean()),
        max_drift=float(merged["drift"].max()),
        drift_df=merged,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_diagnostics(
    canonical_adj_df: pd.DataFrame,
    ts_adj_df: pd.DataFrame,
    manifest_entries: List[RollManifestEntry],
    ts_unadj_df: Optional[pd.DataFrame],
    symbol: str,
    tick_size: float = 0.01,
    min_gap_ticks: int = 2,
    max_day_delta: int = 2,
    dedup_window: int = 3,
) -> DiagnosticResult:
    """Run full L2 + L3 + L4 diagnostics for one symbol.

    Parameters
    ----------
    canonical_adj_df : pd.DataFrame
        Databento adjusted continuous series with ``Date`` and ``Close``.
    ts_adj_df : pd.DataFrame
        TradeStation adjusted series with ``Date`` and ``Close``.
    manifest_entries : list of RollManifestEntry
        Canonical roll manifest entries.
    ts_unadj_df : pd.DataFrame or None
        TradeStation unadjusted series for TS roll inference.
        If None, L2/L3 will run with an empty TS roll list.
    symbol : str
        Symbol label.
    tick_size : float
        Instrument tick size for TS roll detection.
    min_gap_ticks : int
        Minimum spread-step in ticks for TS roll detection.
    max_day_delta : int
        Maximum calendar-day tolerance for roll matching.
    dedup_window : int
        Trading-day window for deduplicating clustered TS roll detections.

    Returns
    -------
    DiagnosticResult
    """
    # Derive TS rolls using spread-step method (unadj - adj).
    ts_rolls: List[TSRollEvent] = []
    if ts_unadj_df is not None and not ts_unadj_df.empty:
        if ts_adj_df is not None and not ts_adj_df.empty:
            ts_rolls = derive_ts_roll_events_from_spread(
                ts_unadj_df, ts_adj_df, tick_size, min_gap_ticks,
                dedup_window,
            )
        else:
            # Fallback to deprecated unadjusted-only method.
            ts_rolls = derive_ts_roll_events_from_unadjusted(
                ts_unadj_df, tick_size, min_gap_ticks,
            )

    # L2.
    l2 = run_l2(manifest_entries, ts_rolls, symbol, max_day_delta)

    # L3.
    l3 = run_l3(l2, symbol)

    # L4.
    l4 = run_l4(canonical_adj_df, ts_adj_df, l2.detail_df, symbol)

    return DiagnosticResult(symbol=symbol, l2=l2, l3=l3, l4=l4)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_diagnostic_artifacts(
    result: DiagnosticResult,
    out_dir: Path,
    prefix: str = "",
) -> Dict[str, Path]:
    """Save diagnostic artifacts for one symbol.

    Files written:
    - ``{prefix}L2_roll_schedule_comparison.csv``
    - ``{prefix}L3_roll_gap_comparison.csv``
    - ``{prefix}L4_adjusted_series_drift.csv``
    - ``{prefix}L4_drift_explanation.json``

    Parameters
    ----------
    result : DiagnosticResult
    out_dir : Path
    prefix : str
        Optional prefix (e.g. ``"ES_"``).

    Returns
    -------
    dict mapping artifact name to saved file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = prefix

    paths: Dict[str, Path] = {}

    # L2.
    l2_path = out_dir / f"{pfx}L2_roll_schedule_comparison.csv"
    result.l2.detail_df.to_csv(l2_path, index=False)
    paths["l2_csv"] = l2_path

    # L3.
    l3_path = out_dir / f"{pfx}L3_roll_gap_comparison.csv"
    result.l3.detail_df.to_csv(l3_path, index=False)
    paths["l3_csv"] = l3_path

    # L4 drift.
    l4_path = out_dir / f"{pfx}L4_adjusted_series_drift.csv"
    result.l4.drift_df.to_csv(l4_path, index=False)
    paths["l4_csv"] = l4_path

    # L4 explanation.
    l4_json_path = out_dir / f"{pfx}L4_drift_explanation.json"
    explanation_data = (
        result.l4.explanation.to_dict() if result.l4.explanation else {}
    )
    with open(l4_json_path, "w") as f:
        json.dump(explanation_data, f, indent=2)
    paths["l4_json"] = l4_json_path

    return paths
