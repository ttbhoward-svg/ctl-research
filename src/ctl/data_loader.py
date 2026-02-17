"""Data loader for Phase 1a OHLCV data.

Handles two data sources:
  1. TradeStation CSV exports (28 tradable symbols)
  2. yfinance downloads (SBSW research_only)

All loaded data is normalised into a consistent DataFrame schema before
being written to data/processed/.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ctl.universe import Universe

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Canonical column order after normalisation.
OHLCV_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# TradeStation CSV common column aliases.
_TS_COL_MAP = {
    "date": "Date",
    "time": "Time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "vol": "Volume",
    "volume": "Volume",
    "up": "Volume",  # TS sometimes labels volume as "Up"
}

# Timeframe labels used in file naming.
TIMEFRAMES = ("daily", "weekly", "monthly", "h4")

# yfinance interval mapping.
_YF_INTERVALS = {
    "daily": "1d",
    "weekly": "1wk",
    "monthly": "1mo",
}


# ---------------------------------------------------------------------------
# TradeStation CSV loader
# ---------------------------------------------------------------------------


def load_tradestation_csv(
    path: Path,
    timeframe: str,
) -> pd.DataFrame:
    """Load a single TradeStation OHLCV CSV export into a normalised DataFrame.

    Parameters
    ----------
    path : Path
        CSV file exported from TradeStation.
    timeframe : str
        One of 'daily', 'weekly', 'monthly', 'h4'.

    Returns
    -------
    pd.DataFrame
        Columns: Date (datetime64), Open, High, Low, Close, Volume (float64).
        Sorted by Date ascending, duplicates removed.
    """
    df = pd.read_csv(path)

    # Normalise column names.
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=_TS_COL_MAP)

    # Parse date.
    if "Date" not in df.columns:
        raise ValueError(f"No 'Date' column found in {path}. Got: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure numeric OHLCV.
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[OHLCV_COLS].copy()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


def load_all_tradestation(
    universe: Universe,
    raw_dir: Path = RAW_DIR,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Scan raw_dir for TradeStation CSVs and load all symbols/timeframes.

    Expected file naming convention (case-insensitive):
        {symbol}_{timeframe}.csv
    Examples:
        ES_daily.csv, PA_daily.csv, XLE_weekly.csv, GS_h4.csv

    Futures symbols: strip the leading slash for filename matching
    (/PA -> PA_daily.csv).

    Returns
    -------
    Dict[symbol, Dict[timeframe, DataFrame]]
    """
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    missing: List[str] = []

    for sym in universe.all_symbols:
        file_stem = sym.lstrip("/")
        result[sym] = {}
        for tf in TIMEFRAMES:
            candidates = [
                raw_dir / f"{file_stem}_{tf}.csv",
                raw_dir / f"{file_stem.upper()}_{tf}.csv",
                raw_dir / f"{file_stem.lower()}_{tf}.csv",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found is None:
                # SBSW is loaded via yfinance, not TradeStation.
                if sym != "SBSW":
                    missing.append(f"{sym}/{tf}")
                continue
            try:
                df = load_tradestation_csv(found, tf)
                result[sym][tf] = df
                logger.info("Loaded %s/%s: %d rows from %s", sym, tf, len(df), found.name)
            except Exception:
                logger.exception("Failed to load %s", found)
                missing.append(f"{sym}/{tf} (parse error)")

    if missing:
        logger.warning(
            "Missing data files (%d). These must be exported from TradeStation:\n  %s",
            len(missing),
            "\n  ".join(missing),
        )

    return result


# ---------------------------------------------------------------------------
# yfinance loader (SBSW only in Phase 1a)
# ---------------------------------------------------------------------------


def download_yfinance(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for a single ticker from yfinance.

    Returns dict of timeframe -> DataFrame for daily, weekly, monthly.
    H4 data is NOT available from yfinance — returns empty DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for SBSW download. Install: pip install yfinance"
        )

    if end is None:
        end = date.today().isoformat()

    result: Dict[str, pd.DataFrame] = {}

    for tf, yf_interval in _YF_INTERVALS.items():
        raw = yf.download(ticker, start=start, end=end, interval=yf_interval, progress=False)
        if raw.empty:
            logger.warning("yfinance returned no data for %s/%s", ticker, tf)
            result[tf] = pd.DataFrame(columns=OHLCV_COLS)
            continue

        df = raw.reset_index()
        # yfinance column names vary by version; normalise.
        col_map = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl == "date":
                col_map[c] = "Date"
            elif cl == "open":
                col_map[c] = "Open"
            elif cl == "high":
                col_map[c] = "High"
            elif cl == "low":
                col_map[c] = "Low"
            elif cl in ("close", "adj close"):
                col_map[c] = "Close"
            elif cl == "volume":
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        # If both Close and Adj Close mapped, prefer Close.
        for col in OHLCV_COLS:
            if col not in df.columns:
                if col == "Volume":
                    df["Volume"] = 0.0
                else:
                    raise ValueError(f"Missing column {col} for {ticker}/{tf}")

        df["Date"] = pd.to_datetime(df["Date"])
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[OHLCV_COLS].copy()
        df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
        result[tf] = df
        logger.info("yfinance %s/%s: %d rows", ticker, tf, len(df))

    # H4 not available from yfinance.
    result["h4"] = pd.DataFrame(columns=OHLCV_COLS)
    logger.info("yfinance %s/h4: not available, set to empty", ticker)

    return result


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_processed(
    data: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: Path = PROCESSED_DIR,
) -> List[Path]:
    """Save normalised DataFrames as parquet files under out_dir.

    Naming: {symbol}_{timeframe}.parquet  (symbol with / stripped).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for sym, tfs in data.items():
        stem = sym.lstrip("/")
        for tf, df in tfs.items():
            if df.empty:
                continue
            p = out_dir / f"{stem}_{tf}.parquet"
            df.to_parquet(p, index=False)
            written.append(p)
    logger.info("Saved %d processed files to %s", len(written), out_dir)
    return written


def load_processed(
    symbol: str,
    timeframe: str,
    proc_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Load a single processed parquet file."""
    stem = symbol.lstrip("/")
    p = proc_dir / f"{stem}_{timeframe}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Processed data not found: {p}")
    return pd.read_parquet(p)
