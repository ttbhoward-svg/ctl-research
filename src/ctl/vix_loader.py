"""VIX regime loader for Phase 1a.

Loads daily VIX close data and assigns a deterministic regime flag:
  - VIX_Regime = True  if VIX close < 20 (low-volatility environment)
  - VIX_Regime = False if VIX close >= 20

Phase 1a uses VIX-level-only classification.  If a ``vix3m_close`` column
is present, term-structure enrichment (contango/backwardation) can be
added in a future phase.

See docs/notes/Task7_assumptions.md for lag rules and fallback rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Spec §11 threshold.
_VIX_LOW_THRESHOLD = 20.0


def load_vix_csv(path: Path) -> pd.DataFrame:
    """Load daily VIX data from CSV.

    Expected columns (case-insensitive):
      - date      : datetime — trading day
      - vix_close : float   — VIX daily close

    Optional:
      - vix3m_close : float — VIX3M close (for future term-structure regime)

    Returns
    -------
    pd.DataFrame
        Normalised columns: date (datetime64), vix_close (float64).
        Optionally includes vix3m_close.  Sorted by date ascending.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "vix_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"VIX CSV missing columns: {missing}. Got: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df["vix_close"] = pd.to_numeric(df["vix_close"], errors="coerce")

    keep = ["date", "vix_close"]
    if "vix3m_close" in df.columns:
        df["vix3m_close"] = pd.to_numeric(df["vix3m_close"], errors="coerce")
        keep.append("vix3m_close")

    df = df[keep].copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    logger.info("Loaded VIX data: %d rows", len(df))
    return df


def compute_vix_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``vix_regime`` column to VIX DataFrame.

    Level-only classification (Phase 1a):
      True  if vix_close < 20
      False if vix_close >= 20
      None  if vix_close is NaN

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_vix_csv``.

    Returns
    -------
    pd.DataFrame
        Same columns plus ``vix_regime`` (bool or None for NaN rows).
    """
    out = df.copy()
    mask_valid = out["vix_close"].notna()
    out["vix_regime"] = None  # object column initially
    out.loc[mask_valid, "vix_regime"] = out.loc[mask_valid, "vix_close"] < _VIX_LOW_THRESHOLD
    logger.info(
        "VIX regime: %d low-vol, %d elevated/high, %d missing",
        (out["vix_regime"] == True).sum(),  # noqa: E712
        (out["vix_regime"] == False).sum(),  # noqa: E712
        out["vix_regime"].isna().sum(),
    )
    return out


def load_and_compute(path: Path) -> pd.DataFrame:
    """Convenience: load CSV and compute regime in one call."""
    return compute_vix_regime(load_vix_csv(path))
