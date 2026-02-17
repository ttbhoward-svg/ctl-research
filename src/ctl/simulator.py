"""B1 trade simulator — walks forward through OHLCV data bar-by-bar.

Implements the locked execution conventions from B1_Strategy_Logic_Spec_v2.md:
  - Entry fill:  Open of entry bar + slippage
  - Stop eval:   Close-based.  Breach when Close[bar] < StopPrice
  - Stop fill:   Next-bar open - slippage
  - TP eval:     High-based.   Hit when High[bar] >= TP level
  - TP fill:     At TP level exactly (resting limit order)
  - Collision:   TP wins (default, configurable)
  - Stop mgmt:   Mode 1 (static) only in Phase 1a

Produces a TradeResult per confirmed B1Trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ctl.b1_detector import B1Trigger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CollisionRule = Literal["tp_wins", "stop_wins"]


@dataclass
class SimConfig:
    """Simulation parameters that are NOT part of the B1 signal logic."""

    slippage_per_side: float = 0.0      # dollar amount, set per instrument
    collision_rule: CollisionRule = "tp_wins"


# ---------------------------------------------------------------------------
# Per-trade result
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    """Full outcome record for one simulated trade."""

    # --- identity (copied from trigger) ------------------------------------
    symbol: str
    timeframe: str
    setup_type: str = "B1"

    # --- trigger context ---------------------------------------------------
    trigger_date: Optional[pd.Timestamp] = None
    trigger_bar_idx: int = -1
    slope_20: float = 0.0
    bars_of_air: int = 0
    ema10_at_trigger: float = 0.0
    atr14_at_trigger: float = 0.0

    # --- trade levels ------------------------------------------------------
    entry_date: Optional[pd.Timestamp] = None
    entry_bar_idx: int = -1
    entry_price: float = 0.0          # Open + slippage
    stop_price: float = 0.0           # Low[N] of trigger bar
    swing_high: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    tp4: float = 0.0
    tp5: float = 0.0

    # --- exit --------------------------------------------------------------
    exit_date: Optional[pd.Timestamp] = None
    exit_bar_idx: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""             # "TP1", "TP2", "TP3", "Stop"

    # --- R-multiples -------------------------------------------------------
    risk_per_unit: float = 0.0        # EntryPrice - StopPrice
    r_multiple_actual: float = 0.0    # weighted actual R (account-constrained)
    theoretical_r: float = 0.0        # weighted ideal-thirds R
    mfe_r: float = 0.0                # (Highest High - Entry) / risk
    mae_r: float = 0.0                # (Entry - Lowest Low) / risk

    # --- flags / research --------------------------------------------------
    day1_fail: bool = False
    same_bar_collision: bool = False
    exit_on_last_bar: bool = False
    trade_outcome: str = ""           # "Win" or "Loss"
    hold_bars: int = 0

    # --- TP hit tracking (for TheoreticalR) --------------------------------
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False

    # --- placeholder columns (NULL in Phase 1a) ----------------------------
    # These exist so the output schema matches Spec §10.  Populated later.
    score_bucket: Optional[str] = None          # tercile after scoring
    asset_cluster: str = ""
    tradable_status: str = "tradable"


# ---------------------------------------------------------------------------
# Simulator core
# ---------------------------------------------------------------------------

def simulate_trade(
    trigger: B1Trigger,
    df: pd.DataFrame,
    config: SimConfig | None = None,
) -> Optional[TradeResult]:
    """Simulate a single confirmed B1 trade through the OHLCV data.

    Parameters
    ----------
    trigger : B1Trigger
        Must have ``confirmed=True`` and ``entry_bar_idx`` set.
    df : DataFrame
        The same OHLCV DataFrame used for detection (with Date, O, H, L, C).
    config : SimConfig, optional

    Returns
    -------
    TradeResult, or None if the trigger is unconfirmed / has no entry bar.
    """
    if not trigger.confirmed or trigger.entry_bar_idx is None:
        return None

    if config is None:
        config = SimConfig()

    dates = df["Date"].values
    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n_bars = len(df)

    entry_idx = trigger.entry_bar_idx
    if entry_idx >= n_bars:
        return None

    entry_price = float(opens[entry_idx]) + config.slippage_per_side
    stop_price = trigger.stop_price
    risk_per_unit = entry_price - stop_price
    if risk_per_unit <= 0:
        # Degenerate: entry at or below stop — skip.
        return None

    tp_levels = [trigger.tp1, trigger.tp2, trigger.tp3]

    # Track MFE / MAE.
    highest_high = float(highs[entry_idx])
    lowest_low = float(lows[entry_idx])

    # Day 1 Fail flag.
    day1_fail = bool(lows[entry_idx] < stop_price)

    # Track which TPs have been hit (for TheoreticalR).
    tp_hit = [False, False, False]

    exit_idx: Optional[int] = None
    exit_price_final = 0.0
    exit_reason = ""
    same_bar_collision = False
    exit_on_last_bar = False

    # Walk forward bar by bar starting from entry bar.
    for bar in range(entry_idx, n_bars):
        h = float(highs[bar])
        lo = float(lows[bar])
        c = float(closes[bar])

        # Update MFE / MAE.
        if h > highest_high:
            highest_high = h
        if lo < lowest_low:
            lowest_low = lo

        # Skip entry bar for exit evaluation — trade is just opened.
        if bar == entry_idx:
            continue

        # --- TP evaluation (High-based) ------------------------------------
        # Check highest TP first (§5.4: multi-TP same bar → highest wins).
        tp_hit_this_bar = -1
        for tp_idx in (2, 1, 0):
            if not tp_hit[tp_idx] and h >= tp_levels[tp_idx]:
                tp_hit_this_bar = tp_idx
                break  # highest qualifying TP found

        # --- Stop evaluation (Close-based) ---------------------------------
        stop_breached = c < stop_price

        # --- Collision resolution ------------------------------------------
        if tp_hit_this_bar >= 0 and stop_breached:
            same_bar_collision = True
            if config.collision_rule == "tp_wins":
                stop_breached = False
            else:
                tp_hit_this_bar = -1

        # --- Resolve exit --------------------------------------------------
        if tp_hit_this_bar >= 0:
            # Mark all TPs up to and including the hit level as hit.
            for j in range(tp_hit_this_bar + 1):
                tp_hit[j] = True

            # If TP3 (or TP2/TP1 covering remaining) -> full exit.
            if tp_hit_this_bar == 2:
                # TP3 hit — all out.
                exit_idx = bar
                exit_price_final = tp_levels[2]
                exit_reason = "TP3"
                break
            elif tp_hit_this_bar == 1 and tp_hit[0]:
                # TP2 hit (and TP1 already hit) — 2/3 exited at TPs.
                # Remaining 1/3 still running.  Continue.
                pass
            elif tp_hit_this_bar == 1 and not tp_hit[0]:
                # TP2 hit in one jump (skipping TP1) — TP1 also fills.
                tp_hit[0] = True
                pass
            elif tp_hit_this_bar == 0:
                # TP1 hit — 1/3 exited.  Continue.
                pass

            # Check if all three TPs are now hit.
            if all(tp_hit):
                exit_idx = bar
                exit_price_final = tp_levels[2]
                exit_reason = "TP3"
                break

        if stop_breached:
            # Stop fill: next-bar open - slippage.
            next_bar = bar + 1
            if next_bar < n_bars:
                exit_idx = next_bar
                exit_price_final = float(opens[next_bar]) - config.slippage_per_side
            else:
                # Stop on last bar of data — use close as fallback.
                exit_idx = bar
                exit_price_final = c
                exit_on_last_bar = True
            exit_reason = "Stop"
            break

    # If we walked to the end of data without an exit, mark as open.
    if exit_idx is None:
        # Trade still open at end of data.  Record with exit at last close.
        exit_idx = n_bars - 1
        exit_price_final = float(closes[n_bars - 1])
        exit_reason = "Open"
        exit_on_last_bar = True

    # --- Compute R-multiples -----------------------------------------------
    actual_r = (exit_price_final - entry_price) / risk_per_unit

    # TheoreticalR: perfect thirds.
    theoretical_r = _compute_theoretical_r(
        tp_hit, tp_levels, entry_price, stop_price,
        exit_price_final, exit_reason, risk_per_unit,
    )

    mfe_r = (highest_high - entry_price) / risk_per_unit
    mae_r = (entry_price - lowest_low) / risk_per_unit

    # Trade outcome.
    trade_outcome = "Win" if tp_hit[0] else "Loss"
    if exit_reason == "Open":
        trade_outcome = "Open"

    hold_bars = exit_idx - entry_idx

    return TradeResult(
        symbol=trigger.symbol,
        timeframe=trigger.timeframe,
        trigger_date=trigger.trigger_date,
        trigger_bar_idx=trigger.trigger_bar_idx,
        slope_20=trigger.slope_20,
        bars_of_air=trigger.bars_of_air,
        ema10_at_trigger=trigger.ema10_at_trigger,
        atr14_at_trigger=trigger.atr14_at_trigger,
        entry_date=pd.Timestamp(dates[entry_idx]),
        entry_bar_idx=entry_idx,
        entry_price=entry_price,
        stop_price=stop_price,
        swing_high=trigger.swing_high,
        tp1=trigger.tp1,
        tp2=trigger.tp2,
        tp3=trigger.tp3,
        tp4=trigger.tp4,
        tp5=trigger.tp5,
        exit_date=pd.Timestamp(dates[exit_idx]),
        exit_bar_idx=exit_idx,
        exit_price=exit_price_final,
        exit_reason=exit_reason,
        risk_per_unit=risk_per_unit,
        r_multiple_actual=actual_r,
        theoretical_r=theoretical_r,
        mfe_r=mfe_r,
        mae_r=mae_r,
        day1_fail=day1_fail,
        same_bar_collision=same_bar_collision,
        exit_on_last_bar=exit_on_last_bar,
        trade_outcome=trade_outcome,
        hold_bars=hold_bars,
        tp1_hit=tp_hit[0],
        tp2_hit=tp_hit[1],
        tp3_hit=tp_hit[2],
    )


def _compute_theoretical_r(
    tp_hit: List[bool],
    tp_levels: List[float],
    entry_price: float,
    stop_price: float,
    exit_price: float,
    exit_reason: str,
    risk_per_unit: float,
) -> float:
    """Compute Theoretical R assuming perfect thirds (Spec §7).

    Each third either exits at its TP level or at the final exit price.
    """
    third = 1.0 / 3.0
    parts: List[float] = []

    if tp_hit[0]:
        parts.append((tp_levels[0] - entry_price) / risk_per_unit)
    else:
        parts.append((exit_price - entry_price) / risk_per_unit)

    if tp_hit[1]:
        parts.append((tp_levels[1] - entry_price) / risk_per_unit)
    else:
        parts.append((exit_price - entry_price) / risk_per_unit)

    if tp_hit[2]:
        parts.append((tp_levels[2] - entry_price) / risk_per_unit)
    else:
        parts.append((exit_price - entry_price) / risk_per_unit)

    return sum(p * third for p in parts)


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_all(
    triggers: List[B1Trigger],
    df: pd.DataFrame,
    config: SimConfig | None = None,
) -> List[TradeResult]:
    """Simulate all confirmed triggers and return trade results."""
    results: List[TradeResult] = []
    for trig in triggers:
        result = simulate_trade(trig, df, config)
        if result is not None:
            results.append(result)
    return results


def results_to_dataframe(results: List[TradeResult]) -> pd.DataFrame:
    """Convert a list of TradeResult to a flat DataFrame (Spec §10 schema)."""
    if not results:
        return pd.DataFrame()

    records: List[Dict] = []
    for r in results:
        records.append({
            # Identification
            "Date": r.trigger_date,
            "Ticker": r.symbol,
            "Timeframe": r.timeframe,
            "SetupType": r.setup_type,
            # Trade levels
            "EntryPrice": r.entry_price,
            "StopPrice": r.stop_price,
            "TP1": r.tp1,
            "TP2": r.tp2,
            "TP3": r.tp3,
            "TP4": r.tp4,
            "TP5": r.tp5,
            # Technical features at trigger
            "BarsOfAir": r.bars_of_air,
            "Slope_20": r.slope_20,
            # Cluster (populated by pipeline, placeholder here)
            "AssetCluster": r.asset_cluster,
            "TradableStatus": r.tradable_status,
            # Outcome
            "RMultiple_Actual": r.r_multiple_actual,
            "TheoreticalR": r.theoretical_r,
            "MFE_R": r.mfe_r,
            "MAE_R": r.mae_r,
            "Day1Fail": r.day1_fail,
            "TradeOutcome": r.trade_outcome,
            "ExitDate": r.exit_date,
            "ExitPrice": r.exit_price,
            "ExitReason": r.exit_reason,
            # Research / meta
            "HoldBars": r.hold_bars,
            "SameBarCollision": r.same_bar_collision,
            "ExitOnLastBar": r.exit_on_last_bar,
            "TP1_Hit": r.tp1_hit,
            "TP2_Hit": r.tp2_hit,
            "TP3_Hit": r.tp3_hit,
            "EntryDate": r.entry_date,
            "EntryBarIdx": r.entry_bar_idx,
            "TriggerBarIdx": r.trigger_bar_idx,
            "SwingHigh": r.swing_high,
            "RiskPerUnit": r.risk_per_unit,
            "ScoreBucket": r.score_bucket,
        })
    return pd.DataFrame(records)
