"""Cutover parity test harness (Data Cutover Task D).

Compares primary (Databento) vs reference (TradeStation) OHLCV data on
three S-criteria:

1. **EMA10 reproduction** — max per-bar divergence percentage.
2. **Trigger date parity** — exact-match of B1 trigger dates.
3. **Trade outcome parity** — R-multiple difference for matched triggers.

Each criterion returns PASS/FAIL.  An ALERT-level reconciliation (Task C)
should block this harness from running; this module assumes data has already
passed reconciliation.

See docs/notes/TaskD_assumptions.md for design rationale.

Usage
-----
>>> from ctl.cutover_parity import run_parity_suite
>>> results = run_parity_suite(primary_df, reference_df, "/ES")
>>> results.summary_dict()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ctl.b1_detector import B1Params, compute_indicators, detect_triggers
from ctl.indicators import ema
from ctl.normalization import AssetClass, NormalizationMode, normalize_ohlcv
from ctl.overlap import align_to_overlap, compute_overlap_window, validate_min_overlap
from ctl.simulator import SimConfig, TradeResult, simulate_all

# ---------------------------------------------------------------------------
# Thresholds (fixed — no changes per Task D constraints)
# ---------------------------------------------------------------------------

#: Maximum allowed EMA divergence percentage.
EMA_MAX_DIVERGENCE_PCT = 0.01

#: Maximum allowed R-multiple difference for matched trades.
R_DIFF_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class EmaParityResult:
    """EMA10 reproduction check result."""

    passed: bool
    max_divergence_pct: float
    threshold_pct: float
    n_compared: int
    n_primary_only: int
    n_reference_only: int
    detail_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class TriggerParityResult:
    """Trigger date parity check result."""

    passed: bool
    n_primary: int
    n_reference: int
    n_matched: int
    extra_in_primary: List[str]
    extra_in_reference: List[str]
    detail_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class TradeParityResult:
    """Trade outcome parity check result."""

    passed: bool
    max_r_diff: float
    threshold_r: float
    n_compared: int
    n_primary_only: int
    n_reference_only: int
    detail_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class ParitySuiteResult:
    """Aggregate result from all three parity checks."""

    symbol: str
    ema: EmaParityResult
    triggers: TriggerParityResult
    trades: TradeParityResult

    @property
    def all_passed(self) -> bool:
        return self.ema.passed and self.triggers.passed and self.trades.passed

    def summary_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "all_passed": self.all_passed,
            "ema_parity": {
                "passed": self.ema.passed,
                "max_divergence_pct": self.ema.max_divergence_pct,
                "threshold_pct": self.ema.threshold_pct,
                "n_compared": self.ema.n_compared,
                "n_primary_only": self.ema.n_primary_only,
                "n_reference_only": self.ema.n_reference_only,
            },
            "trigger_parity": {
                "passed": self.triggers.passed,
                "n_primary": self.triggers.n_primary,
                "n_reference": self.triggers.n_reference,
                "n_matched": self.triggers.n_matched,
                "extra_in_primary": self.triggers.extra_in_primary,
                "extra_in_reference": self.triggers.extra_in_reference,
            },
            "trade_parity": {
                "passed": self.trades.passed,
                "max_r_diff": self.trades.max_r_diff,
                "threshold_r": self.trades.threshold_r,
                "n_compared": self.trades.n_compared,
                "n_primary_only": self.trades.n_primary_only,
                "n_reference_only": self.trades.n_reference_only,
            },
        }


# ---------------------------------------------------------------------------
# EMA parity
# ---------------------------------------------------------------------------

def check_ema_parity(
    primary: pd.DataFrame,
    reference: pd.DataFrame,
    ema_period: int = 10,
    max_divergence_pct: float = EMA_MAX_DIVERGENCE_PCT,
) -> EmaParityResult:
    """Compare EMA10 reproduction between primary and reference.

    Parameters
    ----------
    primary, reference : pd.DataFrame
        OHLCV DataFrames with at least ``Date`` and ``Close`` columns.
    ema_period : int
        EMA period (default 10).
    max_divergence_pct : float
        Maximum allowed divergence in percent.

    Returns
    -------
    EmaParityResult
    """
    # Compute EMA on each dataset.
    ema_p = ema(primary["Close"], ema_period)
    ema_r = ema(reference["Close"], ema_period)

    # Build comparison frames keyed on Date.
    df_p = pd.DataFrame({"Date": primary["Date"], "ema_primary": ema_p})
    df_r = pd.DataFrame({"Date": reference["Date"], "ema_reference": ema_r})

    merged = pd.merge(df_p, df_r, on="Date", how="outer", indicator=True)

    n_primary_only = int((merged["_merge"] == "left_only").sum())
    n_reference_only = int((merged["_merge"] == "right_only").sum())

    matched = merged[merged["_merge"] == "both"].copy()

    # Exclude warmup bars (where EMA hasn't stabilised) and zero-reference.
    matched = matched.iloc[ema_period - 1:]
    mask = matched["ema_reference"].abs() > 0
    compared = matched.loc[mask].copy()

    if compared.empty:
        detail = pd.DataFrame(columns=["Date", "ema_primary", "ema_reference", "divergence_pct"])
        return EmaParityResult(
            passed=True,
            max_divergence_pct=0.0,
            threshold_pct=max_divergence_pct,
            n_compared=0,
            n_primary_only=n_primary_only,
            n_reference_only=n_reference_only,
            detail_df=detail,
        )

    compared["divergence_pct"] = (
        (compared["ema_primary"] - compared["ema_reference"]).abs()
        / compared["ema_reference"].abs()
        * 100
    )

    max_div = float(compared["divergence_pct"].max())

    detail = compared[["Date", "ema_primary", "ema_reference", "divergence_pct"]].reset_index(drop=True)

    return EmaParityResult(
        passed=max_div <= max_divergence_pct,
        max_divergence_pct=round(max_div, 8),
        threshold_pct=max_divergence_pct,
        n_compared=len(compared),
        n_primary_only=n_primary_only,
        n_reference_only=n_reference_only,
        detail_df=detail,
    )


# ---------------------------------------------------------------------------
# Trigger parity
# ---------------------------------------------------------------------------

def check_trigger_parity(
    primary: pd.DataFrame,
    reference: pd.DataFrame,
    symbol: str,
    timeframe: str = "daily",
    params: Optional[B1Params] = None,
) -> TriggerParityResult:
    """Compare B1 trigger dates between primary and reference datasets.

    Both datasets go through the full detection pipeline (indicator
    computation + trigger scan) using identical ``B1Params``.

    Parameters
    ----------
    primary, reference : pd.DataFrame
        OHLCV DataFrames with Date, Open, High, Low, Close, Volume.
    symbol : str
        CTL canonical symbol.
    timeframe : str
        Detection timeframe (default ``"daily"``).
    params : B1Params, optional
        Shared detection parameters.

    Returns
    -------
    TriggerParityResult
    """
    if params is None:
        params = B1Params()

    # Run detection on each.
    p_df = primary.copy()
    r_df = reference.copy()
    compute_indicators(p_df, params)
    compute_indicators(r_df, params)
    triggers_p = detect_triggers(p_df, symbol, timeframe, params)
    triggers_r = detect_triggers(r_df, symbol, timeframe, params)

    dates_p: Set[str] = {str(t.trigger_date.date()) for t in triggers_p}
    dates_r: Set[str] = {str(t.trigger_date.date()) for t in triggers_r}

    extra_p = sorted(dates_p - dates_r)
    extra_r = sorted(dates_r - dates_p)
    matched = sorted(dates_p & dates_r)

    # Build detail frame.
    all_dates = sorted(dates_p | dates_r)
    rows = []
    for d in all_dates:
        rows.append({
            "trigger_date": d,
            "in_primary": d in dates_p,
            "in_reference": d in dates_r,
            "matched": d in dates_p and d in dates_r,
        })
    detail = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["trigger_date", "in_primary", "in_reference", "matched"]
    )

    return TriggerParityResult(
        passed=len(extra_p) == 0 and len(extra_r) == 0,
        n_primary=len(triggers_p),
        n_reference=len(triggers_r),
        n_matched=len(matched),
        extra_in_primary=extra_p,
        extra_in_reference=extra_r,
        detail_df=detail,
    )


# ---------------------------------------------------------------------------
# Trade outcome parity
# ---------------------------------------------------------------------------

def check_trade_parity(
    primary: pd.DataFrame,
    reference: pd.DataFrame,
    symbol: str,
    timeframe: str = "daily",
    params: Optional[B1Params] = None,
    sim_config: Optional[SimConfig] = None,
    r_threshold: float = R_DIFF_THRESHOLD,
) -> TradeParityResult:
    """Compare trade R-multiples between primary and reference datasets.

    Runs full detection + simulation on both datasets, then compares
    R-multiples for trigger dates that appear in both.

    Parameters
    ----------
    primary, reference : pd.DataFrame
        OHLCV DataFrames.
    symbol : str
        CTL canonical symbol.
    timeframe : str
        Detection timeframe.
    params : B1Params, optional
    sim_config : SimConfig, optional
    r_threshold : float
        Maximum allowed absolute R-multiple difference.

    Returns
    -------
    TradeParityResult
    """
    if params is None:
        params = B1Params()
    if sim_config is None:
        sim_config = SimConfig()

    # Detect + simulate on each.
    p_df = primary.copy()
    r_df = reference.copy()
    compute_indicators(p_df, params)
    compute_indicators(r_df, params)
    triggers_p = detect_triggers(p_df, symbol, timeframe, params)
    triggers_r = detect_triggers(r_df, symbol, timeframe, params)
    trades_p = simulate_all(triggers_p, p_df, sim_config)
    trades_r = simulate_all(triggers_r, r_df, sim_config)

    # Index trades by trigger date.
    by_date_p: Dict[str, TradeResult] = {
        str(t.trigger_date.date()): t for t in trades_p if t.trigger_date is not None
    }
    by_date_r: Dict[str, TradeResult] = {
        str(t.trigger_date.date()): t for t in trades_r if t.trigger_date is not None
    }

    dates_p = set(by_date_p.keys())
    dates_r = set(by_date_r.keys())
    matched_dates = sorted(dates_p & dates_r)

    rows = []
    for d in matched_dates:
        r_p = by_date_p[d].r_multiple_actual
        r_r = by_date_r[d].r_multiple_actual
        rows.append({
            "trigger_date": d,
            "r_primary": round(r_p, 6),
            "r_reference": round(r_r, 6),
            "r_diff": round(abs(r_p - r_r), 6),
        })
    detail = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["trigger_date", "r_primary", "r_reference", "r_diff"]
    )

    max_diff = float(detail["r_diff"].max()) if not detail.empty else 0.0

    return TradeParityResult(
        passed=max_diff <= r_threshold,
        max_r_diff=round(max_diff, 6),
        threshold_r=r_threshold,
        n_compared=len(matched_dates),
        n_primary_only=len(dates_p - dates_r),
        n_reference_only=len(dates_r - dates_p),
        detail_df=detail,
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_parity_suite(
    primary: pd.DataFrame,
    reference: pd.DataFrame,
    symbol: str,
    timeframe: str = "daily",
    params: Optional[B1Params] = None,
    sim_config: Optional[SimConfig] = None,
    ema_period: int = 10,
    max_divergence_pct: float = EMA_MAX_DIVERGENCE_PCT,
    r_threshold: float = R_DIFF_THRESHOLD,
    primary_mode: NormalizationMode = "raw",
    reference_mode: NormalizationMode = "raw",
    primary_asset_class: Optional[AssetClass] = None,
    reference_asset_class: Optional[AssetClass] = None,
    enforce_overlap: bool = False,
    min_overlap_bars: int = 0,
) -> ParitySuiteResult:
    """Run all three parity checks for one symbol.

    Parameters
    ----------
    primary, reference : pd.DataFrame
        OHLCV DataFrames with Date, Open, High, Low, Close, Volume.
    symbol : str
        CTL canonical symbol.
    timeframe : str
        Detection timeframe (default ``"daily"``).
    params : B1Params, optional
    sim_config : SimConfig, optional
    ema_period : int
        EMA period for parity check.
    max_divergence_pct : float
        EMA divergence threshold in percent.
    r_threshold : float
        R-multiple difference threshold.
    primary_mode : NormalizationMode
        Normalization mode for primary frame (default ``"raw"``).
    reference_mode : NormalizationMode
        Normalization mode for reference frame (default ``"raw"``).
    primary_asset_class : AssetClass, optional
        Asset class for primary frame.  Required if mode != ``"raw"``.
    reference_asset_class : AssetClass, optional
        Asset class for reference frame.  Required if mode != ``"raw"``.
    enforce_overlap : bool
        If True, trim both frames to their overlapping date range
        before running parity checks (default False).
    min_overlap_bars : int
        Minimum number of overlapping bars required.  Only enforced
        when ``enforce_overlap=True``.  Raises ``ValueError`` if
        the overlap is insufficient (default 0 — no minimum).

    Returns
    -------
    ParitySuiteResult
    """
    # Apply normalization when an asset class is specified.
    if primary_asset_class is not None:
        primary = normalize_ohlcv(
            primary, asset_class=primary_asset_class,
            mode=primary_mode, source="primary",
        )
    if reference_asset_class is not None:
        reference = normalize_ohlcv(
            reference, asset_class=reference_asset_class,
            mode=reference_mode, source="reference",
        )

    # Overlap enforcement.
    if enforce_overlap:
        _, _, n_overlap = compute_overlap_window(primary, reference)
        if min_overlap_bars > 0:
            validate_min_overlap(n_overlap, min_overlap_bars)
        primary, reference = align_to_overlap(primary, reference)

    ema_result = check_ema_parity(
        primary, reference, ema_period, max_divergence_pct,
    )
    trigger_result = check_trigger_parity(
        primary, reference, symbol, timeframe, params,
    )
    trade_result = check_trade_parity(
        primary, reference, symbol, timeframe, params, sim_config, r_threshold,
    )
    return ParitySuiteResult(
        symbol=symbol,
        ema=ema_result,
        triggers=trigger_result,
        trades=trade_result,
    )


# ---------------------------------------------------------------------------
# Multi-symbol runner
# ---------------------------------------------------------------------------

def run_cutover_suite(
    primary_frames: Dict[str, pd.DataFrame],
    reference_frames: Dict[str, pd.DataFrame],
    symbols: Optional[List[str]] = None,
    timeframe: str = "daily",
    params: Optional[B1Params] = None,
    sim_config: Optional[SimConfig] = None,
    primary_mode: NormalizationMode = "raw",
    reference_mode: NormalizationMode = "raw",
    primary_asset_class: Optional[AssetClass] = None,
    reference_asset_class: Optional[AssetClass] = None,
    enforce_overlap: bool = False,
    min_overlap_bars: int = 0,
) -> Dict[str, ParitySuiteResult]:
    """Run parity checks across multiple symbols.

    Parameters
    ----------
    primary_frames : dict
        ``{symbol: ohlcv_df}`` from the primary provider.
    reference_frames : dict
        ``{symbol: ohlcv_df}`` from the reference archive.
    symbols : list of str, optional
        Symbols to check. Defaults to intersection of both dicts.
    primary_mode : NormalizationMode
        Normalization mode for primary frames (default ``"raw"``).
    reference_mode : NormalizationMode
        Normalization mode for reference frames (default ``"raw"``).
    primary_asset_class : AssetClass, optional
        Asset class for primary frames.
    reference_asset_class : AssetClass, optional
        Asset class for reference frames.
    enforce_overlap : bool
        If True, trim frames to overlapping date range (default False).
    min_overlap_bars : int
        Minimum overlap bars when ``enforce_overlap=True`` (default 0).

    Returns
    -------
    dict mapping symbol → ParitySuiteResult
    """
    if symbols is None:
        symbols = sorted(set(primary_frames) & set(reference_frames))

    results: Dict[str, ParitySuiteResult] = {}
    for sym in symbols:
        p = primary_frames.get(sym)
        r = reference_frames.get(sym)
        if p is None or r is None:
            continue
        results[sym] = run_parity_suite(
            p, r, sym, timeframe, params, sim_config,
            primary_mode=primary_mode,
            reference_mode=reference_mode,
            primary_asset_class=primary_asset_class,
            reference_asset_class=reference_asset_class,
            enforce_overlap=enforce_overlap,
            min_overlap_bars=min_overlap_bars,
        )
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_parity_artifacts(
    result: ParitySuiteResult,
    out_dir: Path,
    prefix: str = "",
) -> Dict[str, Path]:
    """Save parity check artifacts as CSV + summary JSON.

    Returns
    -------
    dict with keys ``"ema_csv"``, ``"trigger_csv"``, ``"trade_csv"``,
    ``"summary_json"`` pointing to saved paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pfx = f"{prefix}_" if prefix else ""

    ema_csv = out_dir / f"{pfx}ema_reproduction.csv"
    trigger_csv = out_dir / f"{pfx}trigger_parity_report.csv"
    trade_csv = out_dir / f"{pfx}trade_outcome_parity.csv"
    summary_json = out_dir / f"{pfx}cutover_parity_summary.json"

    result.ema.detail_df.to_csv(ema_csv, index=False)
    result.triggers.detail_df.to_csv(trigger_csv, index=False)
    result.trades.detail_df.to_csv(trade_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump(result.summary_dict(), f, indent=2)

    return {
        "ema_csv": ema_csv,
        "trigger_csv": trigger_csv,
        "trade_csv": trade_csv,
        "summary_json": summary_json,
    }
