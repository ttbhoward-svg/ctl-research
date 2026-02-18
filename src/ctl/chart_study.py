"""Chart Study Session Infrastructure (Task 12).

Provides selection helpers and output generators for structured
visual review of top/bottom/stratified trade setups.

Usage
-----
>>> import pandas as pd
>>> from ctl.chart_study import (
...     filter_dataset, select_top_n, select_bottom_n,
...     select_stratified, generate_study_queue, generate_notes_template,
...     save_study_queue, save_notes_template,
... )
>>> df = pd.read_csv("outputs/phase1a_triggers_v1_20240101.csv")
>>> # Filter to a single symbol
>>> sub = filter_dataset(df, symbol="/ES")
>>> # Select top 10 trades by TheoreticalR
>>> top10 = select_top_n(sub, n=10)
>>> # Generate study queue (JSON) and notes template (Markdown)
>>> queue = generate_study_queue(top10, selection_reason="top_10_by_TheoreticalR")
>>> md = generate_notes_template(top10)
>>> save_study_queue(queue, Path("outputs/study_queue.json"))
>>> save_notes_template(md, Path("outputs/study_notes.md"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PULLBACK_CHOICES = ("textbook", "deep", "shallow", "choppy", "other")
VOLATILITY_CHOICES = ("low", "normal", "high", "extreme")

#: Columns included in study queue records (trade context for review).
QUEUE_COLUMNS = [
    "Date", "Ticker", "Timeframe", "SetupType",
    "EntryPrice", "StopPrice", "TP1", "TP2", "TP3",
    "TheoreticalR", "RMultiple_Actual", "TradeOutcome",
    "ExitReason", "HoldBars", "ScoreBucket", "AssetCluster",
    "BarsOfAir", "Slope_20",
]


# ---------------------------------------------------------------------------
# Trade ID generation
# ---------------------------------------------------------------------------

def make_trade_id(row: pd.Series) -> str:
    """Deterministic trade ID from row identity columns.

    Format: ``{YYYY-MM-DD}_{Ticker}_{Timeframe}_{SetupType}``
    with ``/`` replaced by ``_`` for filesystem safety.
    """
    dt = row["Date"]
    if isinstance(dt, pd.Timestamp):
        dt = dt.strftime("%Y-%m-%d")
    else:
        dt = str(dt)[:10]
    ticker = str(row["Ticker"]).replace("/", "_")
    tf = str(row["Timeframe"])
    setup = str(row.get("SetupType", "B1"))
    return f"{dt}_{ticker}_{tf}_{setup}"


# ---------------------------------------------------------------------------
# Observation schema
# ---------------------------------------------------------------------------

@dataclass
class TradeObservation:
    """Structured observation for a single trade review.

    Fixed fields ensure parseable, consistent records.
    """

    trade_id: str
    setup_quality: Optional[int] = None  # 1-5
    pullback_character: Optional[str] = None  # PULLBACK_CHOICES
    volatility_context: Optional[str] = None  # VOLATILITY_CHOICES
    regime_alignment_note: str = ""
    execution_quality_note: str = ""
    post_trade_reflection: str = ""

    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        errors: List[str] = []
        if self.setup_quality is not None:
            if not (1 <= self.setup_quality <= 5):
                errors.append(
                    f"setup_quality must be 1-5, got {self.setup_quality}"
                )
        if self.pullback_character is not None:
            if self.pullback_character not in PULLBACK_CHOICES:
                errors.append(
                    f"pullback_character must be one of {PULLBACK_CHOICES}, "
                    f"got '{self.pullback_character}'"
                )
        if self.volatility_context is not None:
            if self.volatility_context not in VOLATILITY_CHOICES:
                errors.append(
                    f"volatility_context must be one of {VOLATILITY_CHOICES}, "
                    f"got '{self.volatility_context}'"
                )
        return errors

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeObservation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_dataset(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    score_bucket: Optional[str] = None,
) -> pd.DataFrame:
    """Apply optional AND-combined filters to the assembled dataset.

    Parameters
    ----------
    symbol : str, optional
        Filter to this ticker (e.g. ``"/ES"``).
    timeframe : str, optional
        Filter to this timeframe (e.g. ``"daily"``).
    date_range : (start, end), optional
        Inclusive ISO date strings.
    score_bucket : str, optional
        Filter to this ScoreBucket value (``"top"``, ``"mid"``, ``"bottom"``).

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    mask = pd.Series(True, index=df.index)
    if symbol is not None:
        mask &= df["Ticker"] == symbol
    if timeframe is not None:
        mask &= df["Timeframe"] == timeframe
    if date_range is not None:
        start, end = date_range
        dates = pd.to_datetime(df["Date"])
        mask &= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if score_bucket is not None:
        mask &= df["ScoreBucket"] == score_bucket
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def select_top_n(
    df: pd.DataFrame,
    n: int = 10,
    by: str = "TheoreticalR",
) -> pd.DataFrame:
    """Select the top *n* trades by descending *by* column.

    Ties broken by Date ascending for determinism.
    """
    if by not in df.columns:
        raise ValueError(f"Column '{by}' not in dataset")
    sorted_df = df.sort_values(
        [by, "Date"], ascending=[False, True]
    )
    return sorted_df.head(n).copy()


def select_bottom_n(
    df: pd.DataFrame,
    n: int = 10,
    by: str = "TheoreticalR",
) -> pd.DataFrame:
    """Select the bottom *n* trades by ascending *by* column.

    Ties broken by Date ascending for determinism.
    """
    if by not in df.columns:
        raise ValueError(f"Column '{by}' not in dataset")
    sorted_df = df.sort_values(
        [by, "Date"], ascending=[True, True]
    )
    return sorted_df.head(n).copy()


def select_stratified(
    df: pd.DataFrame,
    n_per_tercile: int = 5,
    by: str = "TheoreticalR",
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample: *n_per_tercile* trades from each tercile.

    Tercile assignment follows the same logic as ``regression.assign_terciles``.
    If a tercile has fewer than *n_per_tercile* trades, all are included.

    Returns
    -------
    pd.DataFrame
        Selected rows with an added ``_tercile`` column.
    """
    if by not in df.columns:
        raise ValueError(f"Column '{by}' not in dataset")
    values = df[by].values.astype(float)
    n = len(values)
    if n == 0:
        out = df.copy()
        out["_tercile"] = pd.Series(dtype=str)
        return out

    third = n // 3
    sorted_vals = np.sort(values)
    low_cutoff = float(sorted_vals[third]) if third < n else 0.0
    high_cutoff = float(sorted_vals[2 * third]) if 2 * third < n else 0.0

    terciles = np.where(
        values >= high_cutoff, "top",
        np.where(values >= low_cutoff, "mid", "bottom"),
    )

    df_work = df.copy()
    df_work["_tercile"] = terciles

    parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for label in ("top", "mid", "bottom"):
        group = df_work[df_work["_tercile"] == label]
        if len(group) <= n_per_tercile:
            parts.append(group)
        else:
            idx = rng.choice(len(group), size=n_per_tercile, replace=False)
            parts.append(group.iloc[sorted(idx)])

    return pd.concat(parts, ignore_index=False)


# ---------------------------------------------------------------------------
# Study queue generation
# ---------------------------------------------------------------------------

@dataclass
class StudyQueueEntry:
    """A single entry in the study queue."""

    trade_id: str
    selection_reason: str
    context: Dict  # subset of trade columns
    observation: Dict  # blank TradeObservation dict


def generate_study_queue(
    df: pd.DataFrame,
    selection_reason: str = "manual",
) -> List[Dict]:
    """Build a study queue from selected trades.

    Returns a list of JSON-serializable dicts, one per trade.
    """
    entries: List[Dict] = []
    for _, row in df.iterrows():
        tid = make_trade_id(row)
        ctx = {}
        for col in QUEUE_COLUMNS:
            if col in row.index:
                val = row[col]
                # Convert numpy/pandas types for JSON serialization.
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, (np.bool_,)):
                    val = bool(val)
                elif isinstance(val, pd.Timestamp):
                    val = val.isoformat()
                elif pd.isna(val):
                    val = None
                ctx[col] = val
        blank_obs = TradeObservation(trade_id=tid).to_dict()
        entries.append({
            "trade_id": tid,
            "selection_reason": selection_reason,
            "context": ctx,
            "observation": blank_obs,
        })
    return entries


def save_study_queue(entries: List[Dict], out_path: Path) -> Path:
    """Write study queue to JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)
    return out_path


def load_study_queue(path: Path) -> List[Dict]:
    """Load a previously saved study queue."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Notes template generation
# ---------------------------------------------------------------------------

def generate_notes_template(df: pd.DataFrame) -> str:
    """Generate a Markdown notes template with one section per trade.

    Each section is pre-filled with trade context and has blank
    observation fields for the reviewer to complete.
    """
    lines: List[str] = []
    lines.append("# Chart Study Notes")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append(f"Trades: {len(df)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        tid = make_trade_id(row)
        lines.append(f"## Trade {i}: {tid}")
        lines.append("")

        # Trade context table.
        lines.append("### Context")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for col in QUEUE_COLUMNS:
            if col in row.index:
                val = row[col]
                if isinstance(val, pd.Timestamp):
                    val = val.strftime("%Y-%m-%d")
                elif isinstance(val, float):
                    val = f"{val:.4f}"
                elif pd.isna(val):
                    val = "—"
                lines.append(f"| {col} | {val} |")
        lines.append("")

        # Observation fields.
        lines.append("### Observations")
        lines.append("")
        lines.append(f"- **setup_quality** (1-5): ")
        lines.append(
            f"- **pullback_character** ({'/'.join(PULLBACK_CHOICES)}): "
        )
        lines.append(
            f"- **volatility_context** ({'/'.join(VOLATILITY_CHOICES)}): "
        )
        lines.append("- **regime_alignment_note**: ")
        lines.append("- **execution_quality_note**: ")
        lines.append("- **post_trade_reflection**: ")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def save_notes_template(content: str, out_path: Path) -> Path:
    """Write notes template to a Markdown file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    return out_path


# ---------------------------------------------------------------------------
# Observation I/O
# ---------------------------------------------------------------------------

def save_observations(
    observations: Sequence[TradeObservation],
    out_path: Path,
) -> Path:
    """Save completed observations as a JSON array."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [obs.to_dict() for obs in observations]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def load_observations(path: Path) -> List[TradeObservation]:
    """Reload observations from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [TradeObservation.from_dict(d) for d in data]
