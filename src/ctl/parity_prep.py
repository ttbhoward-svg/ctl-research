"""Parity prep and overlap validator (Data Cutover Task G).

Discovers Databento continuous files and TradeStation reference files,
validates schemas, computes per-symbol date overlap windows, and enforces
a preflight gate before the full parity run.

See docs/notes/TaskG_assumptions.md for design rationale.

Usage
-----
>>> from ctl.parity_prep import run_parity_prep
>>> report = run_parity_prep()
>>> report.summary_text()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default path to Databento continuous CSVs.
DEFAULT_DB_CONTINUOUS_DIR = (
    REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
)

#: Default path to TradeStation reference CSVs.
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"

#: Default output directory for prep reports.
DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1"

#: Symbols to validate.
PARITY_SYMBOLS = ("ES", "CL", "PA")

#: Minimum overlapping daily bars required to PASS the preflight gate.
MIN_OVERLAP_BARS = 500

#: Required OHLCV columns (after normalisation).
REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")

# Status values.
PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SymbolPrepResult:
    """Prep result for one symbol."""

    symbol: str
    status: str  # PASS, FAIL, or INCOMPLETE

    db_file: Optional[str] = None
    db_bars: int = 0
    db_first_date: Optional[str] = None
    db_last_date: Optional[str] = None

    ts_file: Optional[str] = None
    ts_bars: int = 0
    ts_first_date: Optional[str] = None
    ts_last_date: Optional[str] = None

    first_common_date: Optional[str] = None
    last_common_date: Optional[str] = None
    overlap_bar_count: int = 0

    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "db_file": self.db_file,
            "db_bars": self.db_bars,
            "db_first_date": self.db_first_date,
            "db_last_date": self.db_last_date,
            "ts_file": self.ts_file,
            "ts_bars": self.ts_bars,
            "ts_first_date": self.ts_first_date,
            "ts_last_date": self.ts_last_date,
            "first_common_date": self.first_common_date,
            "last_common_date": self.last_common_date,
            "overlap_bar_count": self.overlap_bar_count,
            "errors": self.errors,
        }


@dataclass
class ParityPrepReport:
    """Aggregate parity prep report."""

    symbols: List[SymbolPrepResult] = field(default_factory=list)
    min_overlap_bars: int = MIN_OVERLAP_BARS

    @property
    def gate_status(self) -> str:
        """Overall preflight gate status.

        - ``PASS``: all symbols have >= min_overlap_bars.
        - ``INCOMPLETE``: one or more symbols missing a file.
        - ``FAIL``: a symbol has both files but < min_overlap_bars.
        """
        statuses = {s.status for s in self.symbols}
        if FAIL in statuses:
            return FAIL
        if INCOMPLETE in statuses:
            return INCOMPLETE
        return PASS

    def summary_dict(self) -> dict:
        return {
            "gate_status": self.gate_status,
            "min_overlap_bars": self.min_overlap_bars,
            "symbols": [s.to_dict() for s in self.symbols],
        }

    def summary_text(self) -> str:
        lines = [f"Parity Prep Gate: {self.gate_status}"]
        for s in self.symbols:
            lines.append(
                f"  [{s.status:10s}] {s.symbol}: "
                f"overlap={s.overlap_bar_count} bars"
                f" ({s.first_common_date} to {s.last_common_date})"
            )
            for e in s.errors:
                lines.append(f"               {e}")
        return "\n".join(lines)

    def to_csv_df(self) -> pd.DataFrame:
        """Return a one-row-per-symbol DataFrame for CSV export."""
        rows = [s.to_dict() for s in self.symbols]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # errors list → semicolon-separated string for CSV.
        df["errors"] = df["errors"].apply(lambda x: "; ".join(x) if x else "")
        return df


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_db_file(
    symbol: str,
    db_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
) -> Optional[Path]:
    """Find the Databento continuous CSV for *symbol*."""
    path = db_dir / f"{symbol}_continuous.csv"
    return path if path.is_file() else None


def discover_ts_file(
    symbol: str,
    ts_dir: Path = DEFAULT_TS_DIR,
) -> Optional[Path]:
    """Find the TradeStation reference CSV for *symbol*.

    Glob pattern: ``TS_{symbol}_1D_*.csv``. If multiple matches, use
    the first sorted match and log a warning.
    """
    matches = sorted(ts_dir.glob(f"TS_{symbol}_1D_*.csv"))
    if not matches:
        # Also try exact match without wildcard suffix.
        exact = ts_dir / f"TS_{symbol}_1D.csv"
        if exact.is_file():
            return exact
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple TS files for %s: %s — using %s",
            symbol,
            [m.name for m in matches],
            matches[0].name,
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Schema validation & loading
# ---------------------------------------------------------------------------

def load_and_validate(
    path: Path,
    label: str,
) -> tuple:
    """Load a CSV and validate required columns.

    Parameters
    ----------
    path : Path
        CSV file to load.
    label : str
        Human label for error messages (e.g. ``"DB ES"``).

    Returns
    -------
    (df, errors) where df is a DataFrame with ``Date`` parsed as datetime
    (or None on failure), and errors is a list of strings.
    """
    errors: List[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, [f"{label}: read error — {exc}"]

    # Normalise column names: lowercase → title-case mapping.
    col_map = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=col_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"{label}: missing columns {missing}")
        return None, errors

    # Parse dates.
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as exc:
        errors.append(f"{label}: date parse error — {exc}")
        return None, errors

    # Sort and deduplicate.
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df, errors


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def compute_overlap(
    db_df: pd.DataFrame,
    ts_df: pd.DataFrame,
) -> dict:
    """Compute the overlap window between two date-indexed DataFrames.

    Returns
    -------
    dict with keys ``first_common_date``, ``last_common_date``,
    ``overlap_bar_count``.
    """
    db_dates = set(db_df["Date"].dt.date)
    ts_dates = set(ts_df["Date"].dt.date)
    common = sorted(db_dates & ts_dates)

    if not common:
        return {
            "first_common_date": None,
            "last_common_date": None,
            "overlap_bar_count": 0,
        }
    return {
        "first_common_date": str(common[0]),
        "last_common_date": str(common[-1]),
        "overlap_bar_count": len(common),
    }


# ---------------------------------------------------------------------------
# Per-symbol prep
# ---------------------------------------------------------------------------

def prep_symbol(
    symbol: str,
    db_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    ts_dir: Path = DEFAULT_TS_DIR,
    min_overlap: int = MIN_OVERLAP_BARS,
) -> SymbolPrepResult:
    """Run parity prep checks for one symbol.

    Parameters
    ----------
    symbol : str
        Root symbol (e.g. ``"ES"``).
    db_dir : Path
        Directory containing ``{SYM}_continuous.csv``.
    ts_dir : Path
        Directory containing ``TS_{SYM}_1D_*.csv``.
    min_overlap : int
        Minimum overlapping bars for PASS.

    Returns
    -------
    SymbolPrepResult
    """
    result = SymbolPrepResult(symbol=symbol, status=INCOMPLETE)

    # --- Databento discovery ---
    db_path = discover_db_file(symbol, db_dir)
    if db_path is None:
        result.errors.append(f"Databento file not found: {db_dir / f'{symbol}_continuous.csv'}")
        return result

    db_df, db_errors = load_and_validate(db_path, f"DB {symbol}")
    result.db_file = db_path.name
    if db_errors:
        result.errors.extend(db_errors)
        result.status = FAIL
        return result

    result.db_bars = len(db_df)
    result.db_first_date = str(db_df["Date"].min().date())
    result.db_last_date = str(db_df["Date"].max().date())

    # --- TradeStation discovery ---
    ts_path = discover_ts_file(symbol, ts_dir)
    if ts_path is None:
        result.errors.append(
            f"TradeStation file not found: {ts_dir / f'TS_{symbol}_1D_*.csv'}"
        )
        return result  # status stays INCOMPLETE

    ts_df, ts_errors = load_and_validate(ts_path, f"TS {symbol}")
    result.ts_file = ts_path.name
    if ts_errors:
        result.errors.extend(ts_errors)
        result.status = FAIL
        return result

    result.ts_bars = len(ts_df)
    result.ts_first_date = str(ts_df["Date"].min().date())
    result.ts_last_date = str(ts_df["Date"].max().date())

    # --- Overlap ---
    overlap = compute_overlap(db_df, ts_df)
    result.first_common_date = overlap["first_common_date"]
    result.last_common_date = overlap["last_common_date"]
    result.overlap_bar_count = overlap["overlap_bar_count"]

    # --- Gate ---
    if result.overlap_bar_count >= min_overlap:
        result.status = PASS
    else:
        result.status = FAIL
        result.errors.append(
            f"Overlap {result.overlap_bar_count} < {min_overlap} required"
        )

    return result


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

def run_parity_prep(
    db_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    ts_dir: Path = DEFAULT_TS_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    symbols: tuple = PARITY_SYMBOLS,
    min_overlap: int = MIN_OVERLAP_BARS,
    save: bool = True,
) -> ParityPrepReport:
    """Run parity prep for all symbols and optionally save reports.

    Parameters
    ----------
    db_dir : Path
        Databento continuous CSV directory.
    ts_dir : Path
        TradeStation reference CSV directory.
    report_dir : Path
        Output directory for reports.
    symbols : tuple
        Root symbols to check.
    min_overlap : int
        Minimum overlapping bars for PASS.
    save : bool
        Whether to write report files.

    Returns
    -------
    ParityPrepReport
    """
    report = ParityPrepReport(min_overlap_bars=min_overlap)
    for sym in symbols:
        report.symbols.append(
            prep_symbol(sym, db_dir, ts_dir, min_overlap)
        )

    if save:
        save_prep_report(report, report_dir)

    return report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_prep_report(
    report: ParityPrepReport,
    out_dir: Path = DEFAULT_REPORT_DIR,
) -> Dict[str, Path]:
    """Save parity prep report as JSON and CSV.

    Returns
    -------
    dict with keys ``json_path`` and ``csv_path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "parity_prep_report.json"
    csv_path = out_dir / "parity_prep_report.csv"

    with open(json_path, "w") as f:
        json.dump(report.summary_dict(), f, indent=2)

    report.to_csv_df().to_csv(csv_path, index=False)

    return {"json_path": json_path, "csv_path": csv_path}
