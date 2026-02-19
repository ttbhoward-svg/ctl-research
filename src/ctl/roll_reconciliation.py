"""Roll manifest alignment and schedule comparison (Data Cutover Task H).

Loads Databento roll manifests, infers TradeStation roll events from
unadjusted price step-changes, and compares roll schedules to identify
alignment issues that explain adjusted-series drift.

Status vocabulary: PASS / WATCH / FAIL
- PASS  — roll aligned within tolerance
- WATCH — roll shifted by 1–``max_day_delta`` trading days
- FAIL  — roll unmatched or outside tolerance

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

Status = str  # "PASS", "WATCH", "FAIL"

_SEVERITY = {"PASS": 0, "WATCH": 1, "FAIL": 2}


def _worst_status(statuses: List[Status]) -> Status:
    """Return the worst (most severe) status in the list."""
    if "FAIL" in statuses:
        return "FAIL"
    if "WATCH" in statuses:
        return "WATCH"
    return "PASS"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RollManifestEntry:
    """One roll event from the Databento roll manifest."""

    roll_date: str
    from_contract: str
    to_contract: str
    from_close: float
    to_close: float
    gap: float
    cumulative_adj: float
    trigger_reason: str = "volume_crossover"
    confirmation_days: int = 2
    convention: str = "subtract"
    session_template: str = "electronic"
    close_type: str = "settlement"

    def to_dict(self) -> dict:
        return {
            "roll_date": self.roll_date,
            "from_contract": self.from_contract,
            "to_contract": self.to_contract,
            "from_close": round(self.from_close, 6),
            "to_close": round(self.to_close, 6),
            "gap": round(self.gap, 6),
            "cumulative_adj": round(self.cumulative_adj, 6),
            "trigger_reason": self.trigger_reason,
            "confirmation_days": self.confirmation_days,
            "convention": self.convention,
            "session_template": self.session_template,
            "close_type": self.close_type,
        }


@dataclass
class TSRollEvent:
    """Roll event inferred from TradeStation unadjusted close step-change."""

    date: object  # datetime.date or str
    gap: float
    close_before: float
    close_after: float


@dataclass
class RollMatch:
    """One matched (or unmatched) pair from schedule comparison."""

    canonical_date: Optional[str] = None
    ts_date: Optional[str] = None
    day_delta: Optional[int] = None
    status: Status = "FAIL"
    canonical_gap: Optional[float] = None
    ts_gap: Optional[float] = None
    gap_diff: Optional[float] = None
    from_contract: Optional[str] = None
    to_contract: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "canonical_date": self.canonical_date,
            "ts_date": self.ts_date,
            "day_delta": self.day_delta,
            "status": self.status,
            "canonical_gap": (
                round(self.canonical_gap, 6) if self.canonical_gap is not None else None
            ),
            "ts_gap": (
                round(self.ts_gap, 6) if self.ts_gap is not None else None
            ),
            "gap_diff": (
                round(self.gap_diff, 6) if self.gap_diff is not None else None
            ),
            "from_contract": self.from_contract,
            "to_contract": self.to_contract,
        }


@dataclass
class RollComparisonResult:
    """Aggregate roll schedule comparison."""

    matches: List[RollMatch] = field(default_factory=list)
    n_canonical: int = 0
    n_ts: int = 0
    n_matched: int = 0
    n_watch: int = 0
    n_fail: int = 0

    @property
    def status(self) -> Status:
        if not self.matches:
            return "PASS"
        return _worst_status([m.status for m in self.matches])

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "n_canonical": self.n_canonical,
            "n_ts": self.n_ts,
            "n_matched": self.n_matched,
            "n_watch": self.n_watch,
            "n_fail": self.n_fail,
            "matches": [m.to_dict() for m in self.matches],
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.matches:
            return pd.DataFrame(
                columns=["canonical_date", "ts_date", "day_delta", "status",
                          "canonical_gap", "ts_gap", "gap_diff",
                          "from_contract", "to_contract"]
            )
        return pd.DataFrame([m.to_dict() for m in self.matches])


@dataclass
class StepExplanation:
    """Explanation of drift contribution from one roll interval."""

    interval_start: str
    interval_end: str
    roll_status: Status
    mean_drift: float
    max_drift: float
    drift_contribution_pct: float

    def to_dict(self) -> dict:
        return {
            "interval_start": self.interval_start,
            "interval_end": self.interval_end,
            "roll_status": self.roll_status,
            "mean_drift": round(self.mean_drift, 6),
            "max_drift": round(self.max_drift, 6),
            "drift_contribution_pct": round(self.drift_contribution_pct, 4),
        }


@dataclass
class DriftExplanationResult:
    """Full drift explanation across all roll intervals."""

    symbol: str
    overall_mean_drift: float
    overall_max_drift: float
    n_intervals: int
    intervals: List[StepExplanation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "overall_mean_drift": round(self.overall_mean_drift, 6),
            "overall_max_drift": round(self.overall_max_drift, 6),
            "n_intervals": self.n_intervals,
            "intervals": [i.to_dict() for i in self.intervals],
        }


# ---------------------------------------------------------------------------
# Roll manifest I/O
# ---------------------------------------------------------------------------

def load_roll_manifest(path: Path) -> List[RollManifestEntry]:
    """Load a roll manifest JSON file.

    Parameters
    ----------
    path : Path
        Path to a JSON file containing a list of roll manifest entries.

    Returns
    -------
    List of RollManifestEntry.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    entries: List[RollManifestEntry] = []
    items = data if isinstance(data, list) else data.get("rolls", [])
    for item in items:
        entries.append(RollManifestEntry(
            roll_date=str(item["roll_date"]),
            from_contract=str(item["from_contract"]),
            to_contract=str(item["to_contract"]),
            from_close=float(item["from_close"]),
            to_close=float(item["to_close"]),
            gap=float(item["gap"]),
            cumulative_adj=float(item["cumulative_adj"]),
            trigger_reason=str(item.get("trigger_reason", "volume_crossover")),
            confirmation_days=int(item.get("confirmation_days", 2)),
            convention=str(item.get("convention", "subtract")),
            session_template=str(item.get("session_template", "electronic")),
            close_type=str(item.get("close_type", "settlement")),
        ))
    return entries


def save_roll_manifest(
    entries: List[RollManifestEntry],
    path: Path,
) -> Path:
    """Save roll manifest entries to a JSON file.

    Returns the path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in entries], f, indent=2)
    return path


# ---------------------------------------------------------------------------
# TS roll inference
# ---------------------------------------------------------------------------

#: Default dedup window (trading days) for clustered spread-step detections.
DEFAULT_DEDUP_WINDOW = 3


def derive_ts_roll_events_from_unadjusted(
    ts_unadj_df: pd.DataFrame,
    tick_size: float,
    min_gap_ticks: int = 2,
) -> List[TSRollEvent]:
    """Infer roll events from a TradeStation unadjusted close series.

    .. deprecated::
        This function detects roll events from raw close-to-close changes,
        which fires on normal daily volatility and produces thousands of
        false positives.  Use :func:`derive_ts_roll_events_from_spread`
        instead, which compares unadjusted vs adjusted closes to isolate
        true roll-day step changes.

    Parameters
    ----------
    ts_unadj_df : pd.DataFrame
        Must have columns ``Date`` and ``Close`` (unadjusted prices).
    tick_size : float
        Minimum tick size for the instrument (e.g. 0.01 for CL, 0.25 for ES).
    min_gap_ticks : int
        Minimum step-change in ticks to flag as a roll event.

    Returns
    -------
    List of TSRollEvent, sorted by date ascending.
    """
    df = ts_unadj_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    threshold = tick_size * min_gap_ticks
    events: List[TSRollEvent] = []

    for i in range(1, len(df)):
        close_prev = float(df.loc[i - 1, "Close"])
        close_cur = float(df.loc[i, "Close"])
        gap = close_cur - close_prev
        if abs(gap) >= threshold:
            events.append(TSRollEvent(
                date=df.loc[i, "Date"].date(),
                gap=round(gap, 6),
                close_before=round(close_prev, 6),
                close_after=round(close_cur, 6),
            ))

    return events


def derive_ts_roll_events_from_spread(
    ts_unadj_df: pd.DataFrame,
    ts_adj_df: pd.DataFrame,
    tick_size: float,
    min_gap_ticks: int = 2,
    dedup_window: int = DEFAULT_DEDUP_WINDOW,
) -> List[TSRollEvent]:
    """Infer roll events from the spread between TS unadjusted and adjusted closes.

    The spread ``Close_unadj - Close_adj`` is constant between rolls and
    shifts by the roll gap on roll dates.  Detecting step changes in this
    spread isolates true roll events from normal daily price volatility.

    Clustered detections within ``dedup_window`` trading days are
    deduplicated by keeping only the first event per cluster.

    Parameters
    ----------
    ts_unadj_df : pd.DataFrame
        Must have columns ``Date`` and ``Close`` (unadjusted prices).
    ts_adj_df : pd.DataFrame
        Must have columns ``Date`` and ``Close`` (adjusted prices).
    tick_size : float
        Minimum tick size for the instrument (e.g. 0.01 for CL, 0.25 for ES).
    min_gap_ticks : int
        Minimum spread-step in ticks to flag as a roll event.
    dedup_window : int
        Trading-day window for deduplicating clustered detections.
        Within each window, only the first detection is kept.

    Returns
    -------
    List of TSRollEvent, sorted by date ascending.
    """
    # Prepare unadjusted.
    unadj = ts_unadj_df.copy()
    unadj["Date"] = pd.to_datetime(unadj["Date"])
    unadj = unadj.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Prepare adjusted.
    adj = ts_adj_df.copy()
    adj["Date"] = pd.to_datetime(adj["Date"])
    adj = adj.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Merge on date.
    merged = pd.merge(
        unadj[["Date", "Close"]].rename(columns={"Close": "close_unadj"}),
        adj[["Date", "Close"]].rename(columns={"Close": "close_adj"}),
        on="Date",
    ).sort_values("Date").reset_index(drop=True)

    if len(merged) < 2:
        return []

    # Compute spread and its first difference.
    merged["spread"] = merged["close_unadj"] - merged["close_adj"]
    merged["spread_diff"] = merged["spread"].diff()

    threshold = tick_size * min_gap_ticks

    # Collect raw detections.
    raw_events: List[TSRollEvent] = []
    for i in range(1, len(merged)):
        diff = float(merged.loc[i, "spread_diff"])
        if abs(diff) >= threshold:
            raw_events.append(TSRollEvent(
                date=merged.loc[i, "Date"].date(),
                gap=round(diff, 6),
                close_before=round(float(merged.loc[i - 1, "close_unadj"]), 6),
                close_after=round(float(merged.loc[i, "close_unadj"]), 6),
            ))

    if not raw_events:
        return []

    # Dedup clustered detections: keep first per window.
    deduped: List[TSRollEvent] = [raw_events[0]]
    for evt in raw_events[1:]:
        prev_date = deduped[-1].date
        curr_date = evt.date
        delta_days = (pd.Timestamp(curr_date) - pd.Timestamp(prev_date)).days
        if delta_days > dedup_window:
            deduped.append(evt)

    return deduped


# ---------------------------------------------------------------------------
# Schedule comparison
# ---------------------------------------------------------------------------

def compare_roll_schedules(
    canonical_rolls: List[RollManifestEntry],
    ts_rolls: List[TSRollEvent],
    max_day_delta: int = 2,
) -> RollComparisonResult:
    """Compare canonical (Databento) roll schedule against TS-inferred rolls.

    Each canonical roll is matched to the closest TS roll within
    ``max_day_delta`` trading days. Unmatched rolls in either set are
    marked FAIL.

    Parameters
    ----------
    canonical_rolls : list of RollManifestEntry
    ts_rolls : list of TSRollEvent
    max_day_delta : int
        Maximum calendar-day difference to consider a match.

    Returns
    -------
    RollComparisonResult
    """
    # Parse dates.
    can_dates = []
    for r in canonical_rolls:
        d = pd.Timestamp(r.roll_date).date() if isinstance(r.roll_date, str) else r.roll_date
        can_dates.append(d)

    ts_dates = []
    for r in ts_rolls:
        d = pd.Timestamp(r.date).date() if isinstance(r.date, str) else r.date
        ts_dates.append(d)

    # Greedy nearest-match: for each canonical roll, find closest TS roll.
    ts_used = set()
    matches: List[RollMatch] = []

    for i, can_r in enumerate(canonical_rolls):
        can_d = can_dates[i]
        best_j: Optional[int] = None
        best_delta: Optional[int] = None

        for j, ts_d in enumerate(ts_dates):
            if j in ts_used:
                continue
            delta = abs((can_d - ts_d).days)
            if delta <= max_day_delta:
                if best_delta is None or delta < best_delta:
                    best_j = j
                    best_delta = delta

        if best_j is not None:
            ts_used.add(best_j)
            ts_r = ts_rolls[best_j]
            gap_diff = abs(can_r.gap - ts_r.gap) if ts_r.gap is not None else None
            status: Status = "PASS" if best_delta == 0 else "WATCH"
            matches.append(RollMatch(
                canonical_date=str(can_d),
                ts_date=str(ts_dates[best_j]),
                day_delta=best_delta,
                status=status,
                canonical_gap=can_r.gap,
                ts_gap=ts_r.gap,
                gap_diff=gap_diff,
                from_contract=can_r.from_contract,
                to_contract=can_r.to_contract,
            ))
        else:
            matches.append(RollMatch(
                canonical_date=str(can_d),
                ts_date=None,
                day_delta=None,
                status="FAIL",
                canonical_gap=can_r.gap,
                ts_gap=None,
                gap_diff=None,
                from_contract=can_r.from_contract,
                to_contract=can_r.to_contract,
            ))

    # Unmatched TS rolls.
    for j, ts_r in enumerate(ts_rolls):
        if j not in ts_used:
            matches.append(RollMatch(
                canonical_date=None,
                ts_date=str(ts_dates[j]),
                day_delta=None,
                status="FAIL",
                canonical_gap=None,
                ts_gap=ts_r.gap,
                gap_diff=None,
            ))

    n_matched = sum(1 for m in matches if m.status == "PASS")
    n_watch = sum(1 for m in matches if m.status == "WATCH")
    n_fail = sum(1 for m in matches if m.status == "FAIL")

    return RollComparisonResult(
        matches=matches,
        n_canonical=len(canonical_rolls),
        n_ts=len(ts_rolls),
        n_matched=n_matched,
        n_watch=n_watch,
        n_fail=n_fail,
    )


# ---------------------------------------------------------------------------
# Drift explanation
# ---------------------------------------------------------------------------

def explain_step_changes(
    canonical_adj_df: pd.DataFrame,
    ts_adj_df: pd.DataFrame,
    roll_compare_df: pd.DataFrame,
) -> DriftExplanationResult:
    """Explain adjusted-series drift in terms of roll-interval contributions.

    For each interval between consecutive matched/watch roll dates, compute
    the mean and max absolute drift (canonical close - TS close). Report
    each interval's contribution to overall drift.

    Parameters
    ----------
    canonical_adj_df : pd.DataFrame
        Databento adjusted continuous series with ``Date`` and ``Close``.
    ts_adj_df : pd.DataFrame
        TradeStation adjusted series with ``Date`` and ``Close``.
    roll_compare_df : pd.DataFrame
        Output of ``RollComparisonResult.to_dataframe()``.

    Returns
    -------
    DriftExplanationResult
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
        return DriftExplanationResult(
            symbol="",
            overall_mean_drift=0.0,
            overall_max_drift=0.0,
            n_intervals=0,
        )

    merged["drift"] = (merged["close_can"] - merged["close_ts"]).abs()
    overall_mean = float(merged["drift"].mean())
    overall_max = float(merged["drift"].max())

    # Build interval boundaries from matched/watch rolls.
    boundaries = []
    statuses_map: Dict[str, str] = {}
    if not roll_compare_df.empty and "canonical_date" in roll_compare_df.columns:
        for _, row in roll_compare_df.iterrows():
            if row.get("status") in ("PASS", "WATCH") and pd.notna(row.get("canonical_date")):
                d = str(row["canonical_date"])
                boundaries.append(d)
                statuses_map[d] = str(row["status"])

    boundaries.sort()

    # Define intervals: [start_of_data, first_roll), [first_roll, second_roll), ...
    all_min = str(merged["Date"].min().date())
    all_max = str(merged["Date"].max().date())

    interval_edges = [all_min] + boundaries + [all_max]
    intervals: List[StepExplanation] = []
    total_drift_sum = float(merged["drift"].sum()) if len(merged) > 0 else 1.0

    for k in range(len(interval_edges) - 1):
        start = pd.Timestamp(interval_edges[k])
        end = pd.Timestamp(interval_edges[k + 1])
        # Use half-open intervals [start, end) except for the last interval.
        if k < len(interval_edges) - 2:
            mask = (merged["Date"] >= start) & (merged["Date"] < end)
        else:
            mask = (merged["Date"] >= start) & (merged["Date"] <= end)
        segment = merged.loc[mask]
        if segment.empty:
            continue
        seg_mean = float(segment["drift"].mean())
        seg_max = float(segment["drift"].max())
        seg_sum = float(segment["drift"].sum())
        contribution = (seg_sum / total_drift_sum * 100.0) if total_drift_sum > 0 else 0.0

        roll_status = statuses_map.get(interval_edges[k], "PASS")

        intervals.append(StepExplanation(
            interval_start=str(start.date()),
            interval_end=str(end.date()),
            roll_status=roll_status,
            mean_drift=seg_mean,
            max_drift=seg_max,
            drift_contribution_pct=contribution,
        ))

    return DriftExplanationResult(
        symbol="",
        overall_mean_drift=overall_mean,
        overall_max_drift=overall_max,
        n_intervals=len(intervals),
        intervals=intervals,
    )
