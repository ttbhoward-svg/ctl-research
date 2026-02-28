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
    day_delta_histogram: Dict[int, int] = field(default_factory=dict)
    unmatched_canonical: int = 0
    unmatched_ts: int = 0
    cumulative_signed_day_shift: int = 0

    @property
    def n_paired(self) -> int:
        """Number of paired events (PASS + WATCH)."""
        return self.n_matched + self.n_watch

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
            "n_paired": self.n_paired,
            "day_delta_histogram": self.day_delta_histogram,
            "unmatched_canonical": self.unmatched_canonical,
            "unmatched_ts": self.unmatched_ts,
            "cumulative_signed_day_shift": self.cumulative_signed_day_shift,
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
# Schedule comparison — alignment config
# ---------------------------------------------------------------------------

#: Cost weight for day-delta in monotonic alignment.
ALIGN_W_DAY: float = 1.0

#: Cost weight for gap-delta (in ticks) in monotonic alignment.
#: Kept small so gap differences act as a tiebreaker, not a veto.
ALIGN_W_GAP: float = 0.01

#: Penalty for leaving a roll event unmatched.
ALIGN_UNMATCHED_PENALTY: float = 50.0

#: Default tick size for gap normalisation (overridden per-symbol at call site).
ALIGN_DEFAULT_TICK_SIZE: float = 0.01


# ---------------------------------------------------------------------------
# Schedule comparison — monotonic DP alignment
# ---------------------------------------------------------------------------

def _monotonic_align(
    can_dates: List,
    can_gaps: List[float],
    ts_dates: List,
    ts_gaps: List[float],
    max_day_delta: int,
    w_day: float,
    w_gap: float,
    unmatched_penalty: float,
    tick_size: float,
) -> List:
    """DP-based monotonic sequence alignment.

    Returns a list of ``(action, can_idx_or_None, ts_idx_or_None)`` tuples
    where *action* is ``"match"``, ``"skip_can"``, or ``"skip_ts"``.
    """
    n = len(can_dates)
    m = len(ts_dates)
    INF = float("inf")

    # dp[i][j] = min cost to align can[0..i-1] with ts[0..j-1].
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + unmatched_penalty
        bt[i][0] = "skip_can"

    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + unmatched_penalty
        bt[0][j] = "skip_ts"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Option 1: skip canonical[i-1].
            cost_skip_can = dp[i - 1][j] + unmatched_penalty

            # Option 2: skip ts[j-1].
            cost_skip_ts = dp[i][j - 1] + unmatched_penalty

            # Option 3: match canonical[i-1] with ts[j-1].
            day_delta = abs((can_dates[i - 1] - ts_dates[j - 1]).days)
            if day_delta <= max_day_delta:
                gap_delta_ticks = (
                    abs(can_gaps[i - 1] - ts_gaps[j - 1]) / tick_size
                    if tick_size > 0
                    else 0.0
                )
                cost_match = dp[i - 1][j - 1] + w_day * day_delta + w_gap * gap_delta_ticks
            else:
                cost_match = INF

            best = min(cost_skip_can, cost_skip_ts, cost_match)
            dp[i][j] = best
            if best == cost_match and cost_match < INF:
                bt[i][j] = "match"
            elif best == cost_skip_can:
                bt[i][j] = "skip_can"
            else:
                bt[i][j] = "skip_ts"

    # Backtrack.
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        action = bt[i][j]
        if action == "match" and i > 0 and j > 0:
            alignment.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "skip_can" and i > 0:
            alignment.append(("skip_can", i - 1, None))
            i -= 1
        elif j > 0:
            alignment.append(("skip_ts", None, j - 1))
            j -= 1
        else:
            break  # safety

    alignment.reverse()
    return alignment


def compare_roll_schedules(
    canonical_rolls: List[RollManifestEntry],
    ts_rolls: List[TSRollEvent],
    max_day_delta: int = 2,
    tick_size: float = ALIGN_DEFAULT_TICK_SIZE,
    w_day: float = ALIGN_W_DAY,
    w_gap: float = ALIGN_W_GAP,
    unmatched_penalty: float = ALIGN_UNMATCHED_PENALTY,
) -> RollComparisonResult:
    """Compare canonical (Databento) roll schedule against TS-inferred rolls.

    Uses monotonic DP alignment to find the minimum-cost order-preserving
    pairing.  This prevents systematic day-shifts from cascading into
    mass FAILs as the old greedy matcher did.

    Parameters
    ----------
    canonical_rolls : list of RollManifestEntry
    ts_rolls : list of TSRollEvent
    max_day_delta : int
        Maximum calendar-day difference to consider a match.
    tick_size : float
        Tick size for normalising gap differences.
    w_day : float
        Cost weight for ``abs(day_delta)``.
    w_gap : float
        Cost weight for ``abs(gap_delta_ticks)``.
    unmatched_penalty : float
        Penalty for leaving a roll event unmatched.

    Returns
    -------
    RollComparisonResult
    """
    if not canonical_rolls and not ts_rolls:
        return RollComparisonResult()

    # Parse dates and gaps.
    can_dates = []
    can_gaps: List[float] = []
    for r in canonical_rolls:
        d = pd.Timestamp(r.roll_date).date() if isinstance(r.roll_date, str) else r.roll_date
        can_dates.append(d)
        can_gaps.append(r.gap)

    ts_dates = []
    ts_gaps: List[float] = []
    for r in ts_rolls:
        d = pd.Timestamp(r.date).date() if isinstance(r.date, str) else r.date
        ts_dates.append(d)
        ts_gaps.append(r.gap)

    # Run DP alignment.
    alignment = _monotonic_align(
        can_dates, can_gaps, ts_dates, ts_gaps,
        max_day_delta, w_day, w_gap, unmatched_penalty, tick_size,
    )

    # Convert alignment to RollMatch list.
    matches: List[RollMatch] = []
    signed_deltas: List[int] = []

    for action, ci, ti in alignment:
        if action == "match":
            can_r = canonical_rolls[ci]
            ts_r = ts_rolls[ti]
            day_delta = abs((can_dates[ci] - ts_dates[ti]).days)
            signed_delta = (ts_dates[ti] - can_dates[ci]).days
            signed_deltas.append(signed_delta)
            gap_diff = abs(can_r.gap - ts_r.gap)
            status: Status = "PASS" if day_delta == 0 else "WATCH"
            matches.append(RollMatch(
                canonical_date=str(can_dates[ci]),
                ts_date=str(ts_dates[ti]),
                day_delta=day_delta,
                status=status,
                canonical_gap=can_r.gap,
                ts_gap=ts_r.gap,
                gap_diff=gap_diff,
                from_contract=can_r.from_contract,
                to_contract=can_r.to_contract,
            ))
        elif action == "skip_can":
            can_r = canonical_rolls[ci]
            matches.append(RollMatch(
                canonical_date=str(can_dates[ci]),
                ts_date=None,
                day_delta=None,
                status="FAIL",
                canonical_gap=can_r.gap,
                ts_gap=None,
                gap_diff=None,
                from_contract=can_r.from_contract,
                to_contract=can_r.to_contract,
            ))
        else:  # skip_ts
            ts_r = ts_rolls[ti]
            matches.append(RollMatch(
                canonical_date=None,
                ts_date=str(ts_dates[ti]),
                day_delta=None,
                status="FAIL",
                canonical_gap=None,
                ts_gap=ts_r.gap,
                gap_diff=None,
            ))

    n_matched = sum(1 for m in matches if m.status == "PASS")
    n_watch = sum(1 for m in matches if m.status == "WATCH")
    n_fail = sum(1 for m in matches if m.status == "FAIL")

    # Richer diagnostics.
    day_delta_hist: Dict[int, int] = {}
    for m in matches:
        if m.day_delta is not None:
            day_delta_hist[m.day_delta] = day_delta_hist.get(m.day_delta, 0) + 1

    unmatched_can = sum(
        1 for m in matches if m.status == "FAIL" and m.canonical_date is not None and m.ts_date is None
    )
    unmatched_ts = sum(
        1 for m in matches if m.status == "FAIL" and m.ts_date is not None and m.canonical_date is None
    )
    cumulative_signed = sum(signed_deltas)

    return RollComparisonResult(
        matches=matches,
        n_canonical=len(canonical_rolls),
        n_ts=len(ts_rolls),
        n_matched=n_matched,
        n_watch=n_watch,
        n_fail=n_fail,
        day_delta_histogram=day_delta_hist,
        unmatched_canonical=unmatched_can,
        unmatched_ts=unmatched_ts,
        cumulative_signed_day_shift=cumulative_signed,
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
