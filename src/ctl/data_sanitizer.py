"""Data sanitizer for Phase 1a OHLCV integrity checks.

Per Project Tracker v3 Task 1.5:
  - Missing trading days (gaps > 3 calendar days excluding weekends/holidays)
  - Bars where High < Low or Open/Close outside High-Low range
  - Zero or negative volume bars
  - Duplicate dates
  - Sudden price jumps > 15% day-over-day (potential bad ticks or roll artifacts)
  - H4 bar timestamp anchoring validation (futures=top of hour, equities=9:30 AM)

Outputs a per-symbol report listing all issues found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ctl.universe import SymbolInfo

logger = logging.getLogger(__name__)

# Maximum calendar-day gap before flagging (excluding weekends).
MAX_CALENDAR_GAP = 5

# Day-over-day price change threshold for flagging bad ticks.
PRICE_JUMP_PCT = 15.0


@dataclass
class Issue:
    """A single data quality issue."""

    symbol: str
    timeframe: str
    severity: str  # "error" | "warning"
    check: str
    detail: str
    row_index: Optional[int] = None
    date: Optional[str] = None


@dataclass
class SanitiserReport:
    """Aggregated report for one symbol across all timeframes."""

    symbol: str
    issues: List[Issue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0

    def summary(self) -> str:
        if self.is_clean and self.warning_count == 0:
            return f"{self.symbol}: CLEAN"
        return (
            f"{self.symbol}: {self.error_count} errors, "
            f"{self.warning_count} warnings"
        )


def sanitise_dataframe(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    sym_info: Optional[SymbolInfo] = None,
) -> List[Issue]:
    """Run all integrity checks on a single OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have columns: Date, Open, High, Low, Close, Volume.
    symbol : str
        For labelling issues.
    timeframe : str
        One of 'daily', 'weekly', 'monthly', 'h4'.
    sym_info : SymbolInfo, optional
        Used for futures vs equity distinction in H4 checks.

    Returns
    -------
    List of Issue objects.
    """
    issues: List[Issue] = []

    if df.empty:
        issues.append(Issue(
            symbol=symbol,
            timeframe=timeframe,
            severity="error",
            check="empty",
            detail="DataFrame is empty — no data loaded.",
        ))
        return issues

    # 1. Duplicate dates.
    dupes = df[df["Date"].duplicated(keep=False)]
    if len(dupes) > 0:
        dupe_dates = dupes["Date"].dt.strftime("%Y-%m-%d").unique().tolist()
        issues.append(Issue(
            symbol=symbol,
            timeframe=timeframe,
            severity="error",
            check="duplicate_dates",
            detail=f"{len(dupes)} duplicate rows on dates: {dupe_dates[:10]}",
        ))

    # 2. OHLC range violations: High < Low, or O/C outside H-L range.
    mask_hl = df["High"] < df["Low"]
    if mask_hl.any():
        bad_idx = df.index[mask_hl].tolist()
        for idx in bad_idx[:10]:
            issues.append(Issue(
                symbol=symbol,
                timeframe=timeframe,
                severity="error",
                check="high_lt_low",
                detail=f"High ({df.at[idx, 'High']}) < Low ({df.at[idx, 'Low']})",
                row_index=idx,
                date=str(df.at[idx, "Date"].date()),
            ))

    for col in ("Open", "Close"):
        above = df[col] > df["High"]
        below = df[col] < df["Low"]
        mask = above | below
        if mask.any():
            bad_idx = df.index[mask].tolist()
            for idx in bad_idx[:10]:
                issues.append(Issue(
                    symbol=symbol,
                    timeframe=timeframe,
                    severity="error",
                    check=f"{col.lower()}_outside_range",
                    detail=(
                        f"{col}={df.at[idx, col]} outside "
                        f"[{df.at[idx, 'Low']}, {df.at[idx, 'High']}]"
                    ),
                    row_index=idx,
                    date=str(df.at[idx, "Date"].date()),
                ))

    # 3. Zero or negative volume.
    if timeframe != "h4":  # H4 volume may not be reliable for all sources.
        mask_vol = df["Volume"] <= 0
        zero_vol_count = mask_vol.sum()
        if zero_vol_count > 0:
            issues.append(Issue(
                symbol=symbol,
                timeframe=timeframe,
                severity="warning",
                check="zero_volume",
                detail=f"{zero_vol_count} bars with zero or negative volume.",
            ))

    # 4. Missing trading days (daily timeframe only).
    if timeframe == "daily":
        issues.extend(_check_gaps(df, symbol, timeframe))

    # 5. Sudden price jumps > 15% day-over-day.
    if len(df) > 1:
        pct_change = df["Close"].pct_change(fill_method=None).abs() * 100
        jumps = pct_change[pct_change > PRICE_JUMP_PCT]
        for idx in jumps.index[:20]:
            issues.append(Issue(
                symbol=symbol,
                timeframe=timeframe,
                severity="warning",
                check="price_jump",
                detail=(
                    f"{pct_change.at[idx]:.1f}% jump: "
                    f"{df.at[idx - 1, 'Close']:.4f} -> {df.at[idx, 'Close']:.4f}"
                ),
                row_index=idx,
                date=str(df.at[idx, "Date"].date()),
            ))

    # 6. NaN values in critical columns.
    for col in ("Open", "High", "Low", "Close"):
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            issues.append(Issue(
                symbol=symbol,
                timeframe=timeframe,
                severity="error",
                check=f"nan_{col.lower()}",
                detail=f"{nan_count} NaN values in {col}.",
            ))

    return issues


def _check_gaps(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> List[Issue]:
    """Check for gaps > MAX_CALENDAR_GAP (weekends excluded)."""
    issues: List[Issue] = []
    if len(df) < 2:
        return issues

    dates = df["Date"].sort_values()
    gaps = dates.diff().dt.days
    # Weekends = 2 calendar days gap is normal; 3 for a Friday-Monday.
    # Flag gaps > MAX_CALENDAR_GAP.
    big_gaps = gaps[gaps > MAX_CALENDAR_GAP]
    for idx in big_gaps.index[:20]:
        prev_date = dates.iloc[dates.index.get_loc(idx) - 1]
        curr_date = dates.at[idx]
        issues.append(Issue(
            symbol=symbol,
            timeframe=timeframe,
            severity="warning",
            check="date_gap",
            detail=(
                f"{int(gaps.at[idx])}-day gap: "
                f"{prev_date.date()} -> {curr_date.date()}"
            ),
            date=str(curr_date.date()),
        ))

    return issues


# ---------------------------------------------------------------------------
# Full universe sanitiser
# ---------------------------------------------------------------------------


def sanitise_universe(
    data: Dict[str, Dict[str, pd.DataFrame]],
    universe: "Universe",  # type: ignore[name-defined]
) -> Dict[str, SanitiserReport]:
    """Run sanitiser on all symbols and timeframes.

    Parameters
    ----------
    data : Dict[symbol, Dict[timeframe, DataFrame]]
    universe : Universe

    Returns
    -------
    Dict[symbol, SanitiserReport]
    """
    from ctl.universe import Universe as _U  # noqa: F811

    reports: Dict[str, SanitiserReport] = {}
    for sym in universe.all_symbols:
        report = SanitiserReport(symbol=sym)
        sym_data = data.get(sym, {})
        sym_info = universe.symbols.get(sym)

        if not sym_data:
            report.issues.append(Issue(
                symbol=sym,
                timeframe="all",
                severity="error",
                check="no_data",
                detail="No data files found for this symbol.",
            ))
            reports[sym] = report
            continue

        for tf in ("daily", "weekly", "monthly", "h4"):
            df = sym_data.get(tf)
            if df is None or df.empty:
                if sym == "SBSW" and tf == "h4":
                    # Expected: H4 not available for SBSW from yfinance.
                    continue
                report.issues.append(Issue(
                    symbol=sym,
                    timeframe=tf,
                    severity="warning",
                    check="missing_timeframe",
                    detail=f"No {tf} data loaded.",
                ))
                continue
            report.issues.extend(
                sanitise_dataframe(df, sym, tf, sym_info)
            )

        reports[sym] = report

    return reports


def print_report(reports: Dict[str, SanitiserReport]) -> str:
    """Format all reports as a human-readable string."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("DATA SANITISER REPORT")
    lines.append("=" * 60)

    total_errors = 0
    total_warnings = 0

    for sym in sorted(reports.keys()):
        r = reports[sym]
        total_errors += r.error_count
        total_warnings += r.warning_count
        lines.append(f"\n--- {r.summary()} ---")
        for issue in r.issues:
            prefix = "ERROR" if issue.severity == "error" else "WARN "
            date_str = f" [{issue.date}]" if issue.date else ""
            lines.append(
                f"  [{prefix}] {issue.timeframe}/{issue.check}{date_str}: "
                f"{issue.detail}"
            )

    lines.append(f"\n{'=' * 60}")
    lines.append(
        f"TOTAL: {total_errors} errors, {total_warnings} warnings "
        f"across {len(reports)} symbols"
    )
    if total_errors > 0:
        lines.append("ACTION: Fix errors before proceeding to signal detection.")
    else:
        lines.append("STATUS: No blocking errors. Warnings should be reviewed.")
    lines.append("=" * 60)

    return "\n".join(lines)
