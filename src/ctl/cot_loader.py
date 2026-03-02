"""COT (Commitments of Traders) data loader for Phase 1a.

Loads pre-processed CFTC COT data and computes canonical tracker features:
  - COT_Commercial_Pctile_3yr      (trailing 156-week percentile)
  - COT_Commercial_Zscore_1yr      (trailing 52-week z-score)
  - COT_Structural_Extreme_5yr     (near 5-year high/low boolean)

Backward-compatible aliases are also emitted:
  - cot_zscore_1y  (same values as cot_commercial_zscore_1yr)
  - cot_20d_delta  (legacy 4-week delta feature)

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
_PCTILE_WINDOW = 156
_EXTREME_WINDOW = 260


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


def _rolling_percentile_rank_last(values: pd.Series) -> float:
    """Percentile rank of last value within the window, in [0, 1]."""
    last = values.iloc[-1]
    return float((values <= last).mean())


def compute_cot_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute canonical and legacy COT features from commercial net positions.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``load_cot_csv``: publication_date, symbol, commercial_net.

    Returns
    -------
    pd.DataFrame
        Columns: publication_date, symbol, commercial_net,
        cot_commercial_pctile_3yr (float or NaN),
        cot_commercial_zscore_1yr (float or NaN),
        cot_structural_extreme_5yr (bool or NaN),
        plus backward-compatible aliases.
        Sorted by (symbol, publication_date).
    """
    df = raw.sort_values(["symbol", "publication_date"]).copy()

    # Legacy 4-week delta per symbol (~20 trading days).
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
        z = (df["commercial_net"] - roll_mean) / roll_std
    z.loc[roll_std == 0] = np.nan
    df["cot_commercial_zscore_1yr"] = z
    # Backward-compatible alias used by older consumers/tests.
    df["cot_zscore_1y"] = z

    # 3-year rolling percentile of commercial net (structural feature).
    df["cot_commercial_pctile_3yr"] = df.groupby("symbol")["commercial_net"].transform(
        lambda s: s.rolling(window=_PCTILE_WINDOW, min_periods=_PCTILE_WINDOW).apply(
            _rolling_percentile_rank_last,
            raw=False,
        )
    )

    # 5-year structural extreme flag: near rolling min/max in 5-year window.
    roll_min_5y = df.groupby("symbol")["commercial_net"].transform(
        lambda s: s.rolling(window=_EXTREME_WINDOW, min_periods=_EXTREME_WINDOW).min()
    )
    roll_max_5y = df.groupby("symbol")["commercial_net"].transform(
        lambda s: s.rolling(window=_EXTREME_WINDOW, min_periods=_EXTREME_WINDOW).max()
    )
    near_low = (df["commercial_net"] - roll_min_5y).abs() <= 1e-12
    near_high = (df["commercial_net"] - roll_max_5y).abs() <= 1e-12
    df["cot_structural_extreme_5yr"] = (near_low | near_high).where(
        roll_min_5y.notna() & roll_max_5y.notna(),
        np.nan,
    )

    df = df.reset_index(drop=True)
    logger.info("Computed COT features: %d rows with valid delta", df["cot_20d_delta"].notna().sum())
    return df


def load_and_compute(path: Path) -> pd.DataFrame:
    """Convenience: load CSV and compute features in one call."""
    return compute_cot_features(load_cot_csv(path))
