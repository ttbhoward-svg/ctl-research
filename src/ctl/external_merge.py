"""Merge external features (COT, VIX) onto B1 triggers with strict no-lookahead.

This module is a separate pipeline step run AFTER trigger detection.  It
annotates B1Trigger objects with external data aligned by publication/effective
date, enforcing strict lag rules:

  - COT: most recent publication_date STRICTLY BEFORE trigger_date
  - VIX: most recent trading date STRICTLY BEFORE trigger_date (prior day close)

See docs/notes/Task7_assumptions.md for full lag rules.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ctl.b1_detector import B1Trigger
from ctl.universe import Universe

logger = logging.getLogger(__name__)


def _build_cot_lookup(
    cot_features: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-index COT features per symbol for fast date lookup.

    Returns dict of symbol -> (dates, deltas, zscores) as sorted numpy arrays.
    """
    lookup: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, grp in cot_features.groupby("symbol"):
        g = grp.sort_values("publication_date")
        lookup[sym] = (
            g["publication_date"].values,
            g["cot_20d_delta"].values.astype(float),
            g["cot_zscore_1y"].values.astype(float),
        )
    return lookup


def _build_vix_lookup(
    vix_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-index VIX data for fast date lookup.

    Returns (dates, regimes) as sorted numpy arrays.
    Regimes are stored as float: 1.0 = True (low vol), 0.0 = False, NaN = missing.
    """
    v = vix_data.sort_values("date")
    regimes = np.full(len(v), np.nan)
    mask_true = v["vix_regime"] == True  # noqa: E712
    mask_false = v["vix_regime"] == False  # noqa: E712
    regimes[mask_true.values] = 1.0
    regimes[mask_false.values] = 0.0
    return v["date"].values, regimes


def lookup_cot(
    symbol: str,
    trigger_date: pd.Timestamp,
    cot_lookup: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[Optional[float], Optional[float]]:
    """Find most recent COT features published STRICTLY BEFORE trigger_date.

    Returns (cot_20d_delta, cot_zscore_1y).  Both None if no data available.
    """
    if symbol not in cot_lookup:
        return None, None

    dates, deltas, zscores = cot_lookup[symbol]
    # searchsorted with side='left' gives the index where trigger_date would be
    # inserted to keep order.  idx-1 is the last date < trigger_date.
    idx = int(np.searchsorted(dates, np.datetime64(trigger_date), side="left")) - 1
    if idx < 0:
        return None, None

    delta = deltas[idx]
    zscore = zscores[idx]
    return (
        None if np.isnan(delta) else float(delta),
        None if np.isnan(zscore) else float(zscore),
    )


def lookup_vix(
    trigger_date: pd.Timestamp,
    vix_lookup: Tuple[np.ndarray, np.ndarray],
) -> Optional[bool]:
    """Find VIX regime for the trading day STRICTLY BEFORE trigger_date.

    Returns True (low vol), False (elevated/high), or None (no data).
    """
    dates, regimes = vix_lookup
    idx = int(np.searchsorted(dates, np.datetime64(trigger_date), side="left")) - 1
    if idx < 0:
        return None

    val = regimes[idx]
    if np.isnan(val):
        return None
    return bool(val == 1.0)


def merge_external_features(
    triggers: List[B1Trigger],
    cot_features: pd.DataFrame | None,
    vix_data: pd.DataFrame | None,
    universe: Universe,
) -> List[B1Trigger]:
    """Annotate triggers with COT and VIX features.

    Mutates trigger objects in-place and returns them.

    Parameters
    ----------
    triggers : list of B1Trigger
    cot_features : pd.DataFrame or None
        Output of ``cot_loader.compute_cot_features``.
        Columns: publication_date, symbol, cot_20d_delta, cot_zscore_1y.
    vix_data : pd.DataFrame or None
        Output of ``vix_loader.compute_vix_regime``.
        Columns: date, vix_close, vix_regime.
    universe : Universe
        Used to determine which symbols are futures (COT-applicable).
    """
    # Build fast lookup structures.
    cot_lookup = _build_cot_lookup(cot_features) if cot_features is not None else {}
    vix_lookup = _build_vix_lookup(vix_data) if vix_data is not None else None

    n_cot = 0
    n_vix = 0

    for trig in triggers:
        # COT: futures only.
        sym_info = universe.symbols.get(trig.symbol)
        if sym_info is not None and sym_info.is_future and cot_lookup:
            delta, zscore = lookup_cot(trig.symbol, trig.trigger_date, cot_lookup)
            trig.cot_20d_delta = delta
            trig.cot_zscore_1y = zscore
            if delta is not None:
                n_cot += 1
        # Non-futures: leave as None (structurally expected).

        # VIX: all symbols.
        if vix_lookup is not None:
            trig.vix_regime = lookup_vix(trig.trigger_date, vix_lookup)
            if trig.vix_regime is not None:
                n_vix += 1

    logger.info(
        "Merged external features: %d/%d COT, %d/%d VIX",
        n_cot, len(triggers), n_vix, len(triggers),
    )
    return triggers
