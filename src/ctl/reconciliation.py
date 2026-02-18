"""Reconciliation engine + health gating (Data Cutover Task C).

Compares primary (Databento) vs secondary (Norgate) canonical-schema
DataFrames to detect divergences in bar count, prices, volume, roll
alignment, and bar coverage.

An ALERT-level divergence blocks downstream scoring and signal generation
via the ``gate_allows_downstream`` flag.

See docs/notes/TaskC_assumptions.md for design rationale.

Usage
-----
>>> from ctl.reconciliation import reconcile_symbol, ReconciliationReport
>>> result = reconcile_symbol(primary_df, secondary_df, "/ES")
>>> report = ReconciliationReport.from_symbols([result])
>>> report.gate_allows_downstream  # False if any ALERT
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ctl.universe import EQUITY_TICK, TICK_VALUES

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Status = str  # "OK", "WATCH", "ALERT"

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

BAR_COUNT_WATCH = 2
BAR_COUNT_ALERT = 5

CLOSE_TICK_WATCH = 1.0
CLOSE_TICK_ALERT = 3.0

VOLUME_PCT_WATCH = 5.0
VOLUME_PCT_ALERT = 15.0

MISSING_BAR_WATCH = 2
MISSING_BAR_ALERT = 5

ROLL_MISALIGN_WATCH = 1
ROLL_MISALIGN_ALERT = 2

#: Close change > this many ticks flags a potential roll date.
ROLL_THRESHOLD_TICKS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(value: float, watch: float, alert: float) -> Status:
    """Classify a metric value into OK / WATCH / ALERT."""
    if value > alert:
        return "ALERT"
    if value > watch:
        return "WATCH"
    return "OK"


def _tick_value(symbol: str) -> float:
    """Look up tick value for a symbol."""
    return TICK_VALUES.get(symbol, EQUITY_TICK)


def _worst_status(statuses: List[Status]) -> Status:
    """Return the worst (most severe) status in the list."""
    if "ALERT" in statuses:
        return "ALERT"
    if "WATCH" in statuses:
        return "WATCH"
    return "OK"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckDetail:
    """One reconciliation check result."""

    name: str
    status: Status
    value: float
    threshold_watch: float
    threshold_alert: float
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "threshold_watch": self.threshold_watch,
            "threshold_alert": self.threshold_alert,
            "detail": self.detail,
        }


@dataclass
class SymbolReconciliation:
    """Reconciliation result for a single symbol."""

    symbol: str
    primary_provider: str
    secondary_provider: str
    primary_bars: int = 0
    secondary_bars: int = 0
    matched_bars: int = 0
    checks: List[CheckDetail] = field(default_factory=list)
    missing_in_primary: List[str] = field(default_factory=list)
    missing_in_secondary: List[str] = field(default_factory=list)
    duplicate_primary: int = 0
    duplicate_secondary: int = 0

    @property
    def status(self) -> Status:
        return _worst_status([c.status for c in self.checks]) if self.checks else "OK"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "primary_provider": self.primary_provider,
            "secondary_provider": self.secondary_provider,
            "primary_bars": self.primary_bars,
            "secondary_bars": self.secondary_bars,
            "matched_bars": self.matched_bars,
            "checks": [c.to_dict() for c in self.checks],
            "missing_in_primary": self.missing_in_primary,
            "missing_in_secondary": self.missing_in_secondary,
            "duplicate_primary": self.duplicate_primary,
            "duplicate_secondary": self.duplicate_secondary,
        }


@dataclass
class ReconciliationReport:
    """Full reconciliation report across all symbols."""

    timestamp: str = ""
    primary_provider: str = ""
    secondary_provider: str = ""
    symbols: List[SymbolReconciliation] = field(default_factory=list)

    @property
    def aggregate_status(self) -> Status:
        return _worst_status([s.status for s in self.symbols]) if self.symbols else "OK"

    @property
    def gate_allows_downstream(self) -> bool:
        """Returns ``False`` when any symbol has ALERT status."""
        return self.aggregate_status != "ALERT"

    @classmethod
    def from_symbols(
        cls,
        results: List[SymbolReconciliation],
        primary_provider: str = "databento",
        secondary_provider: str = "norgate",
    ) -> "ReconciliationReport":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            primary_provider=primary_provider,
            secondary_provider=secondary_provider,
            symbols=results,
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "aggregate_status": self.aggregate_status,
            "gate_allows_downstream": self.gate_allows_downstream,
            "primary_provider": self.primary_provider,
            "secondary_provider": self.secondary_provider,
            "n_symbols": len(self.symbols),
            "n_ok": sum(1 for s in self.symbols if s.status == "OK"),
            "n_watch": sum(1 for s in self.symbols if s.status == "WATCH"),
            "n_alert": sum(1 for s in self.symbols if s.status == "ALERT"),
            "symbols": [s.to_dict() for s in self.symbols],
        }

    def to_csv_rows(self) -> pd.DataFrame:
        """One-row-per-symbol summary suitable for CSV export."""
        rows = []
        for s in self.symbols:
            row: Dict = {
                "symbol": s.symbol,
                "status": s.status,
                "primary_bars": s.primary_bars,
                "secondary_bars": s.secondary_bars,
                "matched_bars": s.matched_bars,
                "missing_in_primary": len(s.missing_in_primary),
                "missing_in_secondary": len(s.missing_in_secondary),
                "duplicate_primary": s.duplicate_primary,
                "duplicate_secondary": s.duplicate_secondary,
            }
            for c in s.checks:
                row[f"{c.name}_value"] = c.value
                row[f"{c.name}_status"] = c.status
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Reconciliation Report",
            "=" * 60,
            f"Timestamp  : {self.timestamp}",
            f"Primary    : {self.primary_provider}",
            f"Secondary  : {self.secondary_provider}",
            f"Aggregate  : {self.aggregate_status}",
            f"Gate open  : {self.gate_allows_downstream}",
            f"Symbols    : {len(self.symbols)} "
            f"(OK={sum(1 for s in self.symbols if s.status == 'OK')}, "
            f"WATCH={sum(1 for s in self.symbols if s.status == 'WATCH')}, "
            f"ALERT={sum(1 for s in self.symbols if s.status == 'ALERT')})",
            "",
        ]
        for s in self.symbols:
            lines.append(f"  [{s.status:5s}] {s.symbol}  "
                         f"bars={s.primary_bars}/{s.secondary_bars}  "
                         f"matched={s.matched_bars}")
            for c in s.checks:
                lines.append(f"          {c.name}: {c.detail}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core reconciliation logic
# ---------------------------------------------------------------------------

def _check_bar_count(n_primary: int, n_secondary: int) -> CheckDetail:
    diff = abs(n_primary - n_secondary)
    status = _classify(diff, BAR_COUNT_WATCH, BAR_COUNT_ALERT)
    return CheckDetail(
        name="bar_count",
        status=status,
        value=float(diff),
        threshold_watch=float(BAR_COUNT_WATCH),
        threshold_alert=float(BAR_COUNT_ALERT),
        detail=f"primary={n_primary}, secondary={n_secondary}, diff={diff}",
    )


def _check_close_divergence(
    matched: pd.DataFrame, tick: float,
) -> CheckDetail:
    """Mean close divergence in ticks across matched bars."""
    if matched.empty or tick == 0:
        return CheckDetail(
            name="close_divergence",
            status="OK",
            value=0.0,
            threshold_watch=CLOSE_TICK_WATCH,
            threshold_alert=CLOSE_TICK_ALERT,
            detail="no matched bars or zero tick",
        )
    diff_ticks = (matched["Close_primary"] - matched["Close_secondary"]).abs() / tick
    mean_ticks = float(diff_ticks.mean())
    max_ticks = float(diff_ticks.max())
    status = _classify(mean_ticks, CLOSE_TICK_WATCH, CLOSE_TICK_ALERT)
    return CheckDetail(
        name="close_divergence",
        status=status,
        value=round(mean_ticks, 4),
        threshold_watch=CLOSE_TICK_WATCH,
        threshold_alert=CLOSE_TICK_ALERT,
        detail=f"mean={mean_ticks:.4f} ticks, max={max_ticks:.4f} ticks",
    )


def _check_volume_divergence(matched: pd.DataFrame) -> CheckDetail:
    """Mean volume divergence % across matched bars."""
    if matched.empty:
        return CheckDetail(
            name="volume_divergence",
            status="OK",
            value=0.0,
            threshold_watch=VOLUME_PCT_WATCH,
            threshold_alert=VOLUME_PCT_ALERT,
            detail="no matched bars",
        )
    # Exclude bars where primary volume is zero (can't compute %).
    mask = matched["Volume_primary"] > 0
    sub = matched.loc[mask]
    if sub.empty:
        return CheckDetail(
            name="volume_divergence",
            status="OK",
            value=0.0,
            threshold_watch=VOLUME_PCT_WATCH,
            threshold_alert=VOLUME_PCT_ALERT,
            detail="all primary volume = 0",
        )
    pct = (
        (sub["Volume_primary"] - sub["Volume_secondary"]).abs()
        / sub["Volume_primary"]
        * 100
    )
    mean_pct = float(pct.mean())
    status = _classify(mean_pct, VOLUME_PCT_WATCH, VOLUME_PCT_ALERT)
    return CheckDetail(
        name="volume_divergence",
        status=status,
        value=round(mean_pct, 4),
        threshold_watch=VOLUME_PCT_WATCH,
        threshold_alert=VOLUME_PCT_ALERT,
        detail=f"mean={mean_pct:.2f}%",
    )


def _check_duplicates(df: pd.DataFrame, label: str) -> CheckDetail:
    """Count duplicate timestamps within one provider's data."""
    n_dup = int(df["timestamp"].duplicated().sum())
    status = "ALERT" if n_dup > 0 else "OK"
    return CheckDetail(
        name=f"duplicates_{label}",
        status=status,
        value=float(n_dup),
        threshold_watch=0.0,
        threshold_alert=0.0,
        detail=f"{n_dup} duplicate timestamps in {label}",
    )


def _check_missing_bars(
    n_missing: int, label: str,
) -> CheckDetail:
    """Classify the number of missing bars."""
    status = _classify(n_missing, MISSING_BAR_WATCH, MISSING_BAR_ALERT)
    return CheckDetail(
        name=f"missing_{label}",
        status=status,
        value=float(n_missing),
        threshold_watch=float(MISSING_BAR_WATCH),
        threshold_alert=float(MISSING_BAR_ALERT),
        detail=f"{n_missing} bars missing in {label}",
    )


def _detect_roll_dates(
    df: pd.DataFrame, tick: float,
    threshold_ticks: float = ROLL_THRESHOLD_TICKS,
) -> List[pd.Timestamp]:
    """Detect potential roll dates based on large close-to-close changes."""
    if len(df) < 2 or tick == 0:
        return []
    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    changes = sorted_df["Close"].diff().abs()
    threshold = threshold_ticks * tick
    roll_mask = changes > threshold
    return list(sorted_df.loc[roll_mask, "timestamp"])


def _check_roll_alignment(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    symbol: str,
    tick: float,
) -> CheckDetail:
    """Check whether roll dates align between providers (futures only)."""
    if not symbol.startswith("/"):
        return CheckDetail(
            name="roll_alignment",
            status="OK",
            value=0.0,
            threshold_watch=float(ROLL_MISALIGN_WATCH),
            threshold_alert=float(ROLL_MISALIGN_ALERT),
            detail="N/A (non-futures)",
        )

    rolls_p = set(_detect_roll_dates(primary, tick))
    rolls_s = set(_detect_roll_dates(secondary, tick))

    # Misaligned = present in one but not the other.
    misaligned = rolls_p.symmetric_difference(rolls_s)
    n_mis = len(misaligned)
    status = _classify(n_mis, ROLL_MISALIGN_WATCH, ROLL_MISALIGN_ALERT)

    detail_parts = [f"primary_rolls={len(rolls_p)}, secondary_rolls={len(rolls_s)}, misaligned={n_mis}"]
    if misaligned:
        dates_str = ", ".join(str(d.date()) for d in sorted(misaligned)[:5])
        if n_mis > 5:
            dates_str += f" ... (+{n_mis - 5} more)"
        detail_parts.append(f"dates: {dates_str}")
    return CheckDetail(
        name="roll_alignment",
        status=status,
        value=float(n_mis),
        threshold_watch=float(ROLL_MISALIGN_WATCH),
        threshold_alert=float(ROLL_MISALIGN_ALERT),
        detail="; ".join(detail_parts),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconcile_symbol(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    symbol: str,
    primary_provider: str = "databento",
    secondary_provider: str = "norgate",
) -> SymbolReconciliation:
    """Reconcile two canonical DataFrames for a single symbol.

    Parameters
    ----------
    primary : pd.DataFrame
        Canonical-schema DataFrame from the primary provider.
    secondary : pd.DataFrame
        Canonical-schema DataFrame from the secondary provider.
    symbol : str
        CTL canonical symbol (e.g. "/ES", "XLE").
    primary_provider, secondary_provider : str
        Provider labels for reporting.

    Returns
    -------
    SymbolReconciliation
    """
    tick = _tick_value(symbol)
    result = SymbolReconciliation(
        symbol=symbol,
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
        primary_bars=len(primary),
        secondary_bars=len(secondary),
    )

    # --- Duplicates ---
    dup_p = _check_duplicates(primary, "primary")
    dup_s = _check_duplicates(secondary, "secondary")
    result.duplicate_primary = int(dup_p.value)
    result.duplicate_secondary = int(dup_s.value)
    result.checks.append(dup_p)
    result.checks.append(dup_s)

    # --- Bar count parity ---
    result.checks.append(_check_bar_count(len(primary), len(secondary)))

    # --- Missing bars ---
    ts_p = set(primary["timestamp"])
    ts_s = set(secondary["timestamp"])
    missing_primary = sorted(ts_s - ts_p)
    missing_secondary = sorted(ts_p - ts_s)
    result.missing_in_primary = [str(t) for t in missing_primary]
    result.missing_in_secondary = [str(t) for t in missing_secondary]
    result.checks.append(_check_missing_bars(len(missing_primary), "primary"))
    result.checks.append(_check_missing_bars(len(missing_secondary), "secondary"))

    # --- Merge matched bars ---
    merged = pd.merge(
        primary[["timestamp", "Open", "High", "Low", "Close", "Volume"]],
        secondary[["timestamp", "Open", "High", "Low", "Close", "Volume"]],
        on="timestamp",
        suffixes=("_primary", "_secondary"),
        how="inner",
    )
    result.matched_bars = len(merged)

    # --- Close divergence ---
    result.checks.append(_check_close_divergence(merged, tick))

    # --- Volume divergence ---
    result.checks.append(_check_volume_divergence(merged))

    # --- Roll date alignment ---
    result.checks.append(_check_roll_alignment(primary, secondary, symbol, tick))

    return result


def reconcile_multi(
    primary_frames: Dict[str, pd.DataFrame],
    secondary_frames: Dict[str, pd.DataFrame],
    primary_provider: str = "databento",
    secondary_provider: str = "norgate",
) -> ReconciliationReport:
    """Reconcile multiple symbols and produce an aggregate report.

    Parameters
    ----------
    primary_frames : dict
        ``{symbol: canonical_df}`` from the primary provider.
    secondary_frames : dict
        ``{symbol: canonical_df}`` from the secondary provider.

    Returns
    -------
    ReconciliationReport
    """
    all_symbols = sorted(set(primary_frames) | set(secondary_frames))
    results: List[SymbolReconciliation] = []

    empty = pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])

    for sym in all_symbols:
        p = primary_frames.get(sym, empty)
        s = secondary_frames.get(sym, empty)
        results.append(
            reconcile_symbol(p, s, sym, primary_provider, secondary_provider)
        )

    return ReconciliationReport.from_symbols(
        results,
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(
    report: ReconciliationReport,
    out_dir: Path,
    prefix: str = "reconciliation",
) -> Dict[str, Path]:
    """Write reconciliation report as JSON + CSV.

    Returns
    -------
    dict with keys ``"json"`` and ``"csv"`` pointing to saved file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"

    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    csv_df = report.to_csv_rows()
    csv_df.to_csv(csv_path, index=False)

    return {"json": json_path, "csv": csv_path}
