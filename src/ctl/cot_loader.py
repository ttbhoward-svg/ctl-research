"""COT (Commitments of Traders) data loader for Phase 1a.

Loads pre-processed CFTC COT data and computes two features per futures symbol:
  - COT_20D_Delta: 4-week change in commercial net position (~20 trading days)
  - COT_ZScore_1Y: z-score of commercial net position over trailing 52 weeks

Only applicable to futures symbols. ETFs and equities have no COT data.
See docs/notes/Task7_assumptions.md for data source expectations and lag rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Weekly observations corresponding to ~20 trading days.
_DELTA_WEEKS = 4

# Trailing window for z-score (52 weeks = 1 year).
_ZSCORE_WINDOW = 52


def load_cot_csv(path: Path) -> pd.DataFrame:
    """Load pre-processed COT data from CSV.

    Expected columns (case-insensitive):
      - publication_date : datetime — Friday release date
      - symbol           : str     — futures symbol (e.g., /ES, /CL)
      - commercial_net   : float   — commercial long minus short

    Returns
    -------
    pd.DataFrame
        Normalised columns: publication_date (datetime64), symbol (str),
        commercial_net (float64). Sorted by (symbol, publication_date).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"publication_date", "symbol", "commercial_net"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"COT CSV missing columns: {missing}. Got: {list(df.columns)}")

    df["publication_date"] = pd.to_datetime(df["publication_date"])
    df["commercial_net"] = pd.to_numeric(df["commercial_net"], errors="coerce")
    df = df[["publication_date", "symbol", "commercial_net"]].copy()
    df = (
        df.sort_values(["symbol", "publication_date"])
        .drop_duplicates(subset=["symbol", "publication_date"])
        .reset_index(drop=True)
    )
    logger.info("Loaded COT data: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


def compute_cot_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute COT_20D_Delta and COT_ZScore_1Y from raw commercial net positions.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``load_cot_csv``: publication_date, symbol, commercial_net.

    Returns
    -------
    pd.DataFrame
        Columns: publication_date, symbol, commercial_net,
        cot_20d_delta (float or NaN), cot_zscore_1y (float or NaN).
        Sorted by (symbol, publication_date).
    """
    df = raw.sort_values(["symbol", "publication_date"]).copy()

    # 4-week delta per symbol.
    df["cot_20d_delta"] = df.groupby("symbol")["commercial_net"].diff(periods=_DELTA_WEEKS)

    # 52-week rolling z-score per symbol.
    grouped = df.groupby("symbol")["commercial_net"]
    roll_mean = grouped.transform(
        lambda s: s.rolling(window=_ZSCORE_WINDOW, min_periods=_ZSCORE_WINDOW).mean()
    )
    roll_std = grouped.transform(
        lambda s: s.rolling(window=_ZSCORE_WINDOW, min_periods=_ZSCORE_WINDOW).std(ddof=1)
    )
    # Avoid division by zero: where std is 0 or NaN, z-score is NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["cot_zscore_1y"] = (df["commercial_net"] - roll_mean) / roll_std
    df.loc[roll_std == 0, "cot_zscore_1y"] = np.nan

    df = df.reset_index(drop=True)
    logger.info("Computed COT features: %d rows with valid delta", df["cot_20d_delta"].notna().sum())
    return df


def load_and_compute(path: Path) -> pd.DataFrame:
    """Convenience: load CSV and compute features in one call."""
    return compute_cot_features(load_cot_csv(path))
