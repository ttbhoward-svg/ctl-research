"""B1 (10-EMA Retest) signal detector.

Implements trigger detection per B1_Strategy_Logic_Spec_v2.md Sections 2-5.
This module identifies *triggers* and resolves *entry timing*.  Trade
simulation (stop/TP evaluation, R-multiple accounting) lives in a separate
simulator module (Task 3).

Scope: Phase 1a — B1 only, static stop, no gap filter, no pyramiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ctl.indicators import atr, ema, slope_pct, williams_r

# ---------------------------------------------------------------------------
# Default parameters (Spec §11)
# ---------------------------------------------------------------------------

@dataclass
class B1Params:
    """All tuneable B1 parameters with Phase 1a defaults."""

    ema_period: int = 10
    slope_lookback: int = 20
    slope_threshold: float = 8.0
    min_bars_of_air: int = 6
    max_bars_of_air_lookback: int = 50
    breakdown_buffer_atr: float = 0.5
    entry_grace_bars: int = 3
    swing_lookback_daily: int = 20
    swing_lookback_weekly: int = 12
    atr_period: int = 14
    williams_r_period: int = 10


# ---------------------------------------------------------------------------
# Trigger row dataclass
# ---------------------------------------------------------------------------

@dataclass
class B1Trigger:
    """One detected B1 trigger with all computed fields."""

    # Identity
    trigger_bar_idx: int          # positional index in the DataFrame
    trigger_date: pd.Timestamp
    symbol: str
    timeframe: str

    # Conditions at trigger
    slope_20: float
    bars_of_air: int
    ema10_at_trigger: float
    atr14_at_trigger: float

    # Trade levels (set at trigger time)
    stop_price: float             # Low[N]
    swing_high: float
    tp1: float
    tp2: float
    tp3: float
    tp4: float
    tp5: float

    # Entry resolution
    confirmed: bool = False
    confirmation_bar_idx: Optional[int] = None
    entry_bar_idx: Optional[int] = None
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None   # Open of entry bar (no slippage yet)

    # Flags
    expired: bool = False

    # MTFA flags (Spec §9) — None if HTF data not provided
    weekly_trend_aligned: Optional[bool] = None
    monthly_trend_aligned: Optional[bool] = None


# ---------------------------------------------------------------------------
# Indicator pre-computation
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, params: B1Params) -> pd.DataFrame:
    """Add all indicator columns needed for B1 detection.

    Mutates *df* in-place and returns it for convenience.
    Requires columns: Date, Open, High, Low, Close, Volume.
    """
    p = params

    df["EMA10"] = ema(df["Close"], p.ema_period)
    df["ATR14"] = atr(df["High"], df["Low"], df["Close"], p.atr_period)
    df["Slope_20"] = slope_pct(df["EMA10"], p.slope_lookback)
    df["WilliamsR"] = williams_r(
        df["High"], df["Low"], df["Close"], p.williams_r_period,
    )
    return df


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _count_bars_of_air(
    lows: np.ndarray,
    ema10s: np.ndarray,
    n: int,
    max_lookback: int,
) -> int:
    """Count consecutive bars before bar *n* where Low > EMA10.

    Spec §2 C2: start at N-1 and look backward.
    """
    count = 0
    for i in range(1, max_lookback + 1):
        j = n - i
        if j < 0:
            break
        if lows[j] <= ema10s[j]:
            return i - 1
        count = i
    return count


def _find_swing_high(
    highs: np.ndarray,
    end_exclusive: int,
    lookback: int,
) -> float:
    """Highest High in the *lookback* bars ending at end_exclusive-1.

    Spec §5.1: SwingHigh = Highest(High[N-1], SwingLookback).
    Tie-break: most recent bar wins (np.argmax returns first occurrence
    when scanning left-to-right, so we reverse the slice).
    """
    start = max(0, end_exclusive - lookback)
    window = highs[start:end_exclusive]
    if len(window) == 0:
        return float("nan")
    # Reverse to make argmax pick the rightmost (most recent) tie.
    rev = window[::-1]
    best_rev_idx = int(np.argmax(rev))
    return float(rev[best_rev_idx])


# ---------------------------------------------------------------------------
# MTFA helpers (Spec §9)
# ---------------------------------------------------------------------------

def _prepare_htf(df: pd.DataFrame, params: B1Params) -> tuple:
    """Pre-compute EMA10 and Slope_20 on a higher-timeframe DataFrame.

    Returns (dates, close, ema10, slope20) as numpy arrays for fast lookup.
    The input DataFrame must have Date and Close columns.  Rows are sorted
    by Date to ensure correct searchsorted behavior.
    """
    work = df.sort_values("Date").reset_index(drop=True)
    ema10 = ema(work["Close"], params.ema_period)
    slope20 = slope_pct(ema10, params.slope_lookback)
    return (
        work["Date"].values,
        work["Close"].values.astype(float),
        ema10.values.astype(float),
        slope20.values.astype(float),
    )


def _monthly_cutoff(trigger_date: pd.Timestamp) -> pd.Timestamp:
    """Last calendar day of the month before *trigger_date*'s month.

    Ensures only fully completed monthly bars are used (no current-month
    in-progress bar).
    """
    return trigger_date.replace(day=1) - pd.Timedelta(days=1)


def _htf_aligned(
    htf_dates: np.ndarray,
    htf_close: np.ndarray,
    htf_ema10: np.ndarray,
    htf_slope20: np.ndarray,
    cutoff: pd.Timestamp,
    slope_threshold: float,
) -> bool:
    """Check HTF trend alignment per Spec §9.

    Returns True if the last HTF bar at or before *cutoff* has:
      (1) Slope_20 >= slope_threshold, AND
      (2) Close > EMA10.
    Returns False if no bar exists or indicators are NaN (insufficient history).
    """
    idx = int(np.searchsorted(htf_dates, np.datetime64(cutoff), side="right")) - 1
    if idx < 0:
        return False
    c = htf_close[idx]
    e = htf_ema10[idx]
    s = htf_slope20[idx]
    if np.isnan(e) or np.isnan(s):
        return False
    return bool(s >= slope_threshold and c > e)


def detect_triggers(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: B1Params | None = None,
    weekly_df: pd.DataFrame | None = None,
    monthly_df: pd.DataFrame | None = None,
) -> List[B1Trigger]:
    """Scan a fully-indicator-enriched DataFrame for B1 triggers.

    Parameters
    ----------
    df : DataFrame
        Must already have EMA10, ATR14, Slope_20 columns
        (call ``compute_indicators`` first).
    symbol : str
    timeframe : str
        'daily' or 'weekly' — controls swing lookback.
    params : B1Params, optional
    weekly_df : DataFrame, optional
        Weekly OHLCV data for WeeklyTrendAligned flag (Spec §9.1).
        Must have Date and Close columns.  If None, flag is left as None.
    monthly_df : DataFrame, optional
        Monthly OHLCV data for MonthlyTrendAligned flag (Spec §9.2).
        Must have Date and Close columns.  If None, flag is left as None.

    Returns
    -------
    List of B1Trigger objects, one per detected trigger.
    Entry resolution (confirmation + entry bar) is included.
    """
    if params is None:
        params = B1Params()

    # Prepare HTF data for MTFA flags (Spec §9).
    weekly_arrays = _prepare_htf(weekly_df, params) if weekly_df is not None else None
    monthly_arrays = _prepare_htf(monthly_df, params) if monthly_df is not None else None

    swing_lookback = (
        params.swing_lookback_daily
        if timeframe == "daily"
        else params.swing_lookback_weekly
    )

    # Pre-extract numpy arrays for speed.
    dates = df["Date"].values
    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    ema10 = df["EMA10"].values.astype(float)
    atr14 = df["ATR14"].values.astype(float)
    slope20 = df["Slope_20"].values.astype(float)
    n_bars = len(df)

    # Warmup: need at least slope_lookback + ema warmup bars.
    min_bar = max(params.slope_lookback, 200)

    triggers: List[B1Trigger] = []

    # State: pending trigger being confirmed.
    pending: Optional[B1Trigger] = None
    pending_expiry_idx: int = -1
    # State: currently in an open position (entry resolved, not yet exited).
    # The *simulator* tracks exits; here we only suppress re-triggers.
    in_position: bool = False
    position_entry_idx: int = -1

    for n in range(min_bar, n_bars):
        # --- resolve pending entry first -----------------------------------
        if pending is not None and not pending.confirmed:
            grace_offset = n - pending.trigger_bar_idx
            if 1 <= grace_offset <= params.entry_grace_bars:
                # Check confirmation: Close > EMA10 on this bar.
                if closes[n] > ema10[n]:
                    pending.confirmed = True
                    pending.confirmation_bar_idx = n
                    entry_idx = n + 1
                    if entry_idx < n_bars:
                        pending.entry_bar_idx = entry_idx
                        pending.entry_date = pd.Timestamp(dates[entry_idx])
                        pending.entry_price = float(opens[entry_idx])
                    else:
                        # Confirmed but entry bar beyond data — still flag it.
                        pending.entry_bar_idx = None
                        pending.entry_date = None
                        pending.entry_price = None
                    triggers.append(pending)
                    in_position = True
                    position_entry_idx = pending.entry_bar_idx or n
                    pending = None
                    continue
            elif grace_offset > params.entry_grace_bars:
                # Expired without confirmation.
                pending.expired = True
                triggers.append(pending)
                pending = None

        # --- retrigger suppression (Spec §3.4, §3.5) ----------------------
        if pending is not None:
            # Still waiting on confirmation — ignore new triggers.
            continue
        if in_position:
            # Phase 1a: ignore new triggers while in a position on same
            # symbol+timeframe.  Position exit is resolved by the simulator;
            # for detection we conservatively assume the position is still
            # open.  The simulator will call ``release_position`` to re-enable.
            # For a pure detector pass, we leave in_position sticky — this
            # means the detector under-counts triggers when called standalone
            # without a simulator.  That's acceptable; the combined
            # detect+simulate pipeline is the canonical path.
            #
            # To make the detector usable standalone for trigger *counting*,
            # we reset in_position after a generous max-hold window (60 bars).
            if n - position_entry_idx > 60:
                in_position = False
            else:
                continue

        # --- evaluate C1-C4 -----------------------------------------------
        if np.isnan(slope20[n]) or np.isnan(ema10[n]) or np.isnan(atr14[n]):
            continue

        # C1: Uptrend.
        if slope20[n] < params.slope_threshold:
            continue

        # C3: EMA violation (check before C2 — cheap test first).
        if lows[n] > ema10[n]:
            continue

        # C2: Bars of air.
        air = _count_bars_of_air(
            lows, ema10, n, params.max_bars_of_air_lookback,
        )
        if air < params.min_bars_of_air:
            continue

        # C4: Not breakdown.
        buffer = atr14[n] * params.breakdown_buffer_atr
        if closes[n] <= ema10[n] - buffer:
            continue

        # --- all four conditions met — trigger fires -----------------------
        swing_high = _find_swing_high(highs, n, swing_lookback)
        stop = float(lows[n])
        rng = swing_high - stop

        trig = B1Trigger(
            trigger_bar_idx=n,
            trigger_date=pd.Timestamp(dates[n]),
            symbol=symbol,
            timeframe=timeframe,
            slope_20=float(slope20[n]),
            bars_of_air=air,
            ema10_at_trigger=float(ema10[n]),
            atr14_at_trigger=float(atr14[n]),
            stop_price=stop,
            swing_high=swing_high,
            tp1=stop + rng * 0.618,
            tp2=stop + rng * 0.786,
            tp3=stop + rng * 1.000,
            tp4=stop + rng * 1.618,
            tp5=stop + rng * 2.618,
        )

        # Annotate MTFA flags (Spec §9).
        if weekly_arrays is not None:
            trig.weekly_trend_aligned = _htf_aligned(
                *weekly_arrays, trig.trigger_date, params.slope_threshold,
            )
        if monthly_arrays is not None:
            trig.monthly_trend_aligned = _htf_aligned(
                *monthly_arrays, _monthly_cutoff(trig.trigger_date),
                params.slope_threshold,
            )

        pending = trig
        # Confirmation window starts next bar.

    # Handle any pending trigger left at end of data.
    if pending is not None and not pending.confirmed:
        pending.expired = True
        triggers.append(pending)

    return triggers


# ---------------------------------------------------------------------------
# Convenience: full pipeline from raw DataFrame
# ---------------------------------------------------------------------------

def run_b1_detection(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: B1Params | None = None,
    weekly_df: pd.DataFrame | None = None,
    monthly_df: pd.DataFrame | None = None,
) -> List[B1Trigger]:
    """Compute indicators then detect triggers. Non-mutating wrapper."""
    work = df.copy()
    if params is None:
        params = B1Params()
    compute_indicators(work, params)
    return detect_triggers(work, symbol, timeframe, params, weekly_df, monthly_df)
