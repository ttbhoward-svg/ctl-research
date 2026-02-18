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
    gap_scan_window: int = 100


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

    # Confluence flags (Spec §8) — computed at trigger time
    wr_divergence: bool = False
    clean_pullback: bool = False
    volume_declining: bool = False
    gap_fill_below: bool = False
    multi_year_highs: bool = False
    single_bar_pullback: bool = False
    fib_confluence: Optional[bool] = None   # None if weekly data not provided

    # MTFA flags (Spec §9) — None if HTF data not provided
    weekly_trend_aligned: Optional[bool] = None
    monthly_trend_aligned: Optional[bool] = None

    # External features (Task 7) — set by external_merge, not by detector
    cot_20d_delta: Optional[float] = None
    cot_zscore_1y: Optional[float] = None
    vix_regime: Optional[bool] = None


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


# ---------------------------------------------------------------------------
# Confluence flag helpers (Spec §8)
# ---------------------------------------------------------------------------

def _wr_divergence(
    lows: np.ndarray,
    wr: np.ndarray,
    ema10: np.ndarray,
    n: int,
    max_lookback: int = 100,
) -> bool:
    """Williams %R bullish divergence (Spec §8.1).

    Find the most recent bar before N where Low <= EMA10 (prior EMA touch).
    Divergence = price made a lower/equal low but %R made a higher low.
    """
    for i in range(1, max_lookback + 1):
        j = n - i
        if j < 0:
            return False
        if lows[j] <= ema10[j]:
            # Found prior touch at bar j.
            if np.isnan(wr[n]) or np.isnan(wr[j]):
                return False
            return bool(wr[n] > wr[j] and lows[n] <= lows[j])
    return False


def _clean_pullback(highs: np.ndarray, lows: np.ndarray, n: int) -> bool:
    """Three bars of orderly stair-step decline (Spec §8.2).

    CleanPullback = High[N-1] < High[N-2] < High[N-3]
                AND Low[N-1]  < Low[N-2]  < Low[N-3]
    """
    if n < 3:
        return False
    return bool(
        highs[n - 1] < highs[n - 2] < highs[n - 3]
        and lows[n - 1] < lows[n - 2] < lows[n - 3]
    )


def _volume_declining(volumes: np.ndarray, n: int) -> bool:
    """Recent 3-bar avg volume < prior 3-bar avg volume (Spec §8.3)."""
    if n < 6:
        return False
    recent = (volumes[n - 1] + volumes[n - 2] + volumes[n - 3]) / 3.0
    prior = (volumes[n - 4] + volumes[n - 5] + volumes[n - 6]) / 3.0
    if np.isnan(recent) or np.isnan(prior) or prior == 0:
        return False
    return bool(recent < prior)


def _gap_fill_below(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    stop_price: float,
    n: int,
    scan_window: int,
) -> bool:
    """Unfilled gap-down within 2% below StopPrice (Spec §8.4)."""
    for i in range(1, scan_window + 1):
        j = n - i
        if j < 1:
            break
        if opens[j] < closes[j - 1]:  # gap down
            gap_top = closes[j - 1]
            gap_bottom = opens[j]
            # Gap bottom within 2% below stop.
            if gap_bottom < stop_price and gap_bottom >= stop_price * 0.98:
                # Check if gap has been filled between j and N.
                filled = False
                for k in range(j, n + 1):
                    if highs[k] >= gap_top:
                        filled = True
                        break
                if not filled:
                    return True
    return False


def _multi_year_highs(
    highs: np.ndarray,
    swing_high: float,
    n: int,
    lookback: int = 252,
    threshold: float = 0.95,
) -> bool:
    """SwingHigh within 5% of 252-day high (Spec §8.5)."""
    if n < 1:
        return False
    start = max(0, n - lookback)
    yearly_high = float(np.nanmax(highs[start:n]))
    if np.isnan(yearly_high):
        return False
    return bool(swing_high >= yearly_high * threshold)


def _single_bar_pullback(
    highs: np.ndarray,
    n: int,
    swing_lookback: int,
) -> bool:
    """Bar N-1 is the swing high — only 1 bar of pullback (Spec §8.7).

    Caution flag, not positive confluence.
    """
    if n < 1:
        return False
    start = max(0, n - swing_lookback)
    window_max = float(np.nanmax(highs[start:n]))
    if np.isnan(window_max):
        return False
    return bool(highs[n - 1] == window_max)


def _fib_confluence(
    tp1: float,
    weekly_dates: np.ndarray,
    weekly_highs: np.ndarray,
    weekly_lows: np.ndarray,
    trigger_date: pd.Timestamp,
    swing_lookback: int,
    tolerance: float = 0.01,
) -> bool:
    """Daily TP1 aligns with weekly 0.618 fib level (Spec §8.6).

    Uses ``swing_lookback * 2`` weekly bars for HTF swing range.
    """
    idx = int(
        np.searchsorted(weekly_dates, np.datetime64(trigger_date), side="right")
    ) - 1
    if idx < 0:
        return False

    window = swing_lookback * 2
    start = max(0, idx - window + 1)

    htf_high = float(np.nanmax(weekly_highs[start : idx + 1]))
    htf_low = float(np.nanmin(weekly_lows[start : idx + 1]))
    if np.isnan(htf_high) or np.isnan(htf_low):
        return False

    rng = htf_high - htf_low
    if rng <= 0:
        return False

    htf_fib618 = htf_low + rng * 0.618

    if tp1 <= 0 or np.isnan(tp1):
        return False

    return bool(abs(tp1 - htf_fib618) / tp1 < tolerance)


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

    # Prepare weekly swing data for FibConfluence (Spec §8.6).
    weekly_swing: tuple | None = None
    if weekly_df is not None:
        w_sorted = weekly_df.sort_values("Date").reset_index(drop=True)
        weekly_swing = (
            w_sorted["Date"].values,
            w_sorted["High"].values.astype(float),
            w_sorted["Low"].values.astype(float),
        )

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
    volumes = df["Volume"].values.astype(float)
    wr = df["WilliamsR"].values.astype(float)
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

        # Confluence flags (Spec §8).
        trig.wr_divergence = _wr_divergence(lows, wr, ema10, n)
        trig.clean_pullback = _clean_pullback(highs, lows, n)
        trig.volume_declining = _volume_declining(volumes, n)
        trig.gap_fill_below = _gap_fill_below(
            opens, closes, highs, stop, n, params.gap_scan_window,
        )
        trig.multi_year_highs = _multi_year_highs(highs, swing_high, n)
        trig.single_bar_pullback = _single_bar_pullback(highs, n, swing_lookback)
        if weekly_swing is not None:
            trig.fib_confluence = _fib_confluence(
                trig.tp1, *weekly_swing, trig.trigger_date, swing_lookback,
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
