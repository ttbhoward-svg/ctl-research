"""Continuous futures series builder (Data Cutover Task F).

Reads Databento outright contract CSVs, detects volume-based rolls,
applies Panama-style additive back-adjustment, and outputs clean
daily continuous series.

Supports two adjustment conventions:
- ``"subtract"`` (default): historical prices shifted DOWN when new > old.
- ``"add"``: historical prices shifted UP when new > old.

See docs/notes/TaskF_assumptions.md for design rationale.

Usage
-----
>>> from ctl.continuous_builder import build_continuous
>>> result = build_continuous("ES", data_dir)
>>> result.continuous.to_csv("ES_continuous.csv", index=False)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard futures month codes → 1-based month number.
MONTH_CODES: Dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

#: Reverse mapping: month number → code.
MONTH_NAMES: Dict[int, str] = {v: k for k, v in MONTH_CODES.items()}

#: Year digit → four-digit year (data range 2018-2026).
YEAR_MAP: Dict[str, int] = {
    "8": 2018, "9": 2019, "0": 2020, "1": 2021, "2": 2022,
    "3": 2023, "4": 2024, "5": 2025, "6": 2026, "7": 2027,
}

#: Consecutive days of volume crossover required to trigger a roll.
ROLL_CONSECUTIVE_DAYS = 2

#: Supported adjustment conventions.
#: "subtract" — historical OHLC = raw - cumulative_adj (default).
#: "add"      — historical OHLC = raw + cumulative_adj (inverted).
ADJUSTMENT_CONVENTIONS = ("subtract", "add")

#: Type alias for adjustment convention.
AdjustmentConvention = Literal["subtract", "add"]

#: Contract code regex: root (1-3 uppercase letters) + month (1 letter) + year (1 digit).
CONTRACT_RE = re.compile(r"^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d)$")


# ---------------------------------------------------------------------------
# Contract parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class ContractSpec:
    """Parsed futures contract specification, sortable by expiration."""

    sort_key: int  # year * 12 + month, for natural ordering
    root: str
    month_code: str
    month: int
    year: int
    symbol: str

    @classmethod
    def from_symbol(cls, symbol: str) -> "ContractSpec":
        """Parse a Databento symbol like ``'ESH5'`` into a ContractSpec."""
        m = CONTRACT_RE.match(symbol)
        if m is None:
            raise ValueError(f"Cannot parse contract symbol: '{symbol}'")
        root, month_code, year_digit = m.groups()
        if month_code not in MONTH_CODES:
            raise ValueError(f"Invalid month code '{month_code}' in '{symbol}'")
        if year_digit not in YEAR_MAP:
            raise ValueError(f"Unmapped year digit '{year_digit}' in '{symbol}'")
        month = MONTH_CODES[month_code]
        year = YEAR_MAP[year_digit]
        return cls(
            sort_key=year * 12 + month,
            root=root,
            month_code=month_code,
            month=month,
            year=year,
            symbol=symbol,
        )


def parse_contracts(symbols: List[str]) -> List[ContractSpec]:
    """Parse and sort contract symbols by expiration order."""
    specs = [ContractSpec.from_symbol(s) for s in symbols]
    return sorted(specs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_contract_data(
    data_dir: Path,
    root: str,
) -> Dict[str, pd.DataFrame]:
    """Load all outright contract CSVs for a root symbol.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``*.csv.zst`` files for one root symbol.
    root : str
        Root symbol (e.g. ``"ES"``).

    Returns
    -------
    dict mapping contract symbol → DataFrame with columns
    ``[date, open, high, low, close, volume]``.
    """
    data_dir = Path(data_dir)
    contracts: Dict[str, pd.DataFrame] = {}

    for fpath in sorted(data_dir.glob("*.csv.zst")):
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            logger.warning("Skipping unreadable file %s: %s", fpath.name, exc)
            continue

        if "symbol" not in df.columns or "ts_event" not in df.columns:
            logger.debug("Skipping %s: missing symbol/ts_event columns", fpath.name)
            continue

        sym = df["symbol"].iloc[0]
        try:
            spec = ContractSpec.from_symbol(sym)
        except ValueError:
            logger.debug("Skipping unparseable symbol '%s' in %s", sym, fpath.name)
            continue
        if spec.root != root:
            continue

        # Normalise.
        df["date"] = pd.to_datetime(df["ts_event"]).dt.date
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df = df.reset_index(drop=True)

        # Sanity guard: skip empty or single-row contracts.
        if len(df) < 2:
            logger.debug("Skipping %s: only %d bars", sym, len(df))
            continue

        # Sanity guard: skip contracts with all-zero volume.
        if (df["volume"] == 0).all():
            logger.debug("Skipping %s: all-zero volume", sym)
            continue

        contracts[sym] = df

    return contracts


# ---------------------------------------------------------------------------
# Roll detection
# ---------------------------------------------------------------------------

@dataclass
class RollEvent:
    """One roll from contract A to contract B."""

    date: object  # datetime.date
    from_contract: str
    to_contract: str
    from_close: float
    to_close: float
    adjustment: float  # to_close - from_close
    cumulative_adjustment: float = 0.0


def detect_rolls(
    contracts: Dict[str, pd.DataFrame],
    contract_order: List[ContractSpec],
    consecutive_days: int = ROLL_CONSECUTIVE_DAYS,
) -> Tuple[List[RollEvent], pd.DataFrame]:
    """Detect volume-based roll dates across the contract chain.

    Parameters
    ----------
    contracts : dict
        ``{symbol: df}`` with columns ``[date, open, high, low, close, volume]``.
    contract_order : list of ContractSpec
        Contracts sorted by expiration.
    consecutive_days : int
        Number of consecutive days where next-contract volume must exceed
        current to trigger a roll.

    Returns
    -------
    (rolls, active_series)
        rolls: list of RollEvent
        active_series: DataFrame with ``[date, contract]`` showing which
        contract is active on each date.
    """
    if len(contract_order) < 1:
        return [], pd.DataFrame(columns=["date", "contract"])

    # Index each contract's volume by date for fast lookup.
    vol_by_date: Dict[str, Dict] = {}
    close_by_date: Dict[str, Dict] = {}
    for sym, df in contracts.items():
        vol_by_date[sym] = dict(zip(df["date"], df["volume"]))
        close_by_date[sym] = dict(zip(df["date"], df["close"]))

    # Collect all unique dates across all contracts, sorted.
    all_dates = sorted(set().union(*(set(df["date"]) for df in contracts.values())))

    rolls: List[RollEvent] = []
    active_idx = 0
    crossover_count = 0

    active_records: List[Dict] = []

    for d in all_dates:
        current_sym = contract_order[active_idx].symbol
        next_idx = active_idx + 1

        # Check if a roll should happen.
        if next_idx < len(contract_order):
            next_sym = contract_order[next_idx].symbol
            cur_vol = vol_by_date.get(current_sym, {}).get(d, 0)
            nxt_vol = vol_by_date.get(next_sym, {}).get(d, 0)

            if nxt_vol > cur_vol and nxt_vol > 0:
                crossover_count += 1
            else:
                crossover_count = 0

            if crossover_count >= consecutive_days:
                # Roll!
                cur_close = close_by_date.get(current_sym, {}).get(d, np.nan)
                nxt_close = close_by_date.get(next_sym, {}).get(d, np.nan)
                adj = 0.0
                if not (np.isnan(cur_close) or np.isnan(nxt_close)):
                    adj = nxt_close - cur_close
                rolls.append(RollEvent(
                    date=d,
                    from_contract=current_sym,
                    to_contract=next_sym,
                    from_close=cur_close,
                    to_close=nxt_close,
                    adjustment=adj,
                ))
                active_idx = next_idx
                crossover_count = 0

        # Only record dates where the active contract has data.
        if d in vol_by_date.get(contract_order[active_idx].symbol, {}):
            active_records.append({
                "date": d,
                "contract": contract_order[active_idx].symbol,
            })

    active_df = pd.DataFrame(active_records)
    return rolls, active_df


# ---------------------------------------------------------------------------
# Panama-style back-adjustment
# ---------------------------------------------------------------------------

def apply_panama_adjustment(
    contracts: Dict[str, pd.DataFrame],
    active_series: pd.DataFrame,
    rolls: List[RollEvent],
    convention: AdjustmentConvention = "subtract",
) -> pd.DataFrame:
    """Apply additive back-adjustment to build the continuous series.

    Works backwards from the most recent bar: the last contract's prices
    are unadjusted, and each earlier segment is shifted by the cumulative
    roll adjustments.

    Parameters
    ----------
    contracts : dict
        Raw contract DataFrames.
    active_series : DataFrame
        ``[date, contract]`` from ``detect_rolls``.
    rolls : list of RollEvent
        Roll events from ``detect_rolls``.
    convention : {"subtract", "add"}
        ``"subtract"`` (default): ``price = raw - cumulative_adj``.
        ``"add"``: ``price = raw + cumulative_adj`` (inverted).

    Returns
    -------
    DataFrame with ``[Date, Open, High, Low, Close, Volume, contract, adjustment]``.
    """
    if convention not in ADJUSTMENT_CONVENTIONS:
        raise ValueError(
            f"Unknown convention '{convention}'; expected one of {ADJUSTMENT_CONVENTIONS}"
        )
    if active_series.empty:
        return pd.DataFrame(
            columns=["Date", "Open", "High", "Low", "Close", "Volume",
                      "contract", "adjustment"]
        )

    # Compute cumulative adjustment (backwards from last roll).
    cumulative = 0.0
    for roll in reversed(rolls):
        cumulative += roll.adjustment
        roll.cumulative_adjustment = cumulative

    # Build segment map: for each roll, everything BEFORE that roll's date
    # gets that roll's cumulative adjustment.
    # Process: assign each date its cumulative adjustment.
    # Dates after the last roll → adjustment = 0.
    # Dates on/before roll[i] but after roll[i-1] → cumulative from roll[i] onwards.

    # Convert to a lookup: date → cumulative adjustment to subtract.
    roll_dates = [(r.date, r.cumulative_adjustment) for r in rolls]
    roll_dates.sort()  # oldest to newest

    rows = []
    for _, row in active_series.iterrows():
        d = row["date"]
        sym = row["contract"]
        cdf = contracts.get(sym)
        if cdf is None:
            continue
        bar = cdf.loc[cdf["date"] == d]
        if bar.empty:
            continue
        bar = bar.iloc[0]

        # Determine adjustment: sum of adjustments for all rolls AFTER this date.
        adj = 0.0
        for rd, cum_adj in roll_dates:
            if d < rd:
                adj = cum_adj
                break

        sign = -1.0 if convention == "subtract" else 1.0
        rows.append({
            "Date": d,
            "Open": float(bar["open"]) + sign * adj,
            "High": float(bar["high"]) + sign * adj,
            "Low": float(bar["low"]) + sign * adj,
            "Close": float(bar["close"]) + sign * adj,
            "Volume": int(bar["volume"]),
            "contract": sym,
            "adjustment": round(adj, 6),
        })

    result = pd.DataFrame(rows)
    result["Date"] = pd.to_datetime(result["Date"])
    return result.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ContinuousResult:
    """Output from the continuous series builder."""

    root: str
    continuous: pd.DataFrame  # The adjusted continuous series
    roll_log: pd.DataFrame    # Roll events
    n_contracts: int = 0
    n_rolls: int = 0
    convention: str = "subtract"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_continuous(
    root: str,
    data_dir: Path,
    consecutive_days: int = ROLL_CONSECUTIVE_DAYS,
    convention: AdjustmentConvention = "subtract",
) -> ContinuousResult:
    """Build a continuous back-adjusted series for one root symbol.

    Parameters
    ----------
    root : str
        Root symbol (e.g. ``"ES"``).
    data_dir : Path
        Directory containing per-contract ``*.csv.zst`` files.
    consecutive_days : int
        Consecutive volume-crossover days to trigger a roll.
    convention : {"subtract", "add"}
        Adjustment convention (see ``apply_panama_adjustment``).

    Returns
    -------
    ContinuousResult
    """
    contracts = load_contract_data(data_dir, root)
    if not contracts:
        return ContinuousResult(
            root=root,
            continuous=pd.DataFrame(
                columns=["Date", "Open", "High", "Low", "Close", "Volume",
                          "contract", "adjustment"]
            ),
            roll_log=pd.DataFrame(
                columns=["date", "from_contract", "to_contract",
                          "from_close", "to_close", "adjustment",
                          "cumulative_adjustment"]
            ),
            convention=convention,
        )

    contract_order = parse_contracts(list(contracts.keys()))

    rolls, active_series = detect_rolls(contracts, contract_order, consecutive_days)
    continuous = apply_panama_adjustment(contracts, active_series, rolls, convention)

    roll_log = _build_roll_log(rolls, len(contracts))

    return ContinuousResult(
        root=root,
        continuous=continuous,
        roll_log=roll_log,
        n_contracts=len(contracts),
        n_rolls=len(rolls),
        convention=convention,
    )


def _build_roll_log(
    rolls: List[RollEvent],
    n_contracts: int,
) -> pd.DataFrame:
    """Build a roll diagnostics DataFrame from roll events.

    Includes per-roll: date, from/to contract, from/to close, adjustment,
    cumulative_adjustment, and active_contract_count (contracts remaining
    in the chain after the roll).
    """
    if not rolls:
        return pd.DataFrame(
            columns=["date", "from_contract", "to_contract",
                      "from_close", "to_close", "adjustment",
                      "cumulative_adjustment", "active_contract_count"]
        )
    rows = []
    for i, r in enumerate(rolls):
        rows.append({
            "date": r.date,
            "from_contract": r.from_contract,
            "to_contract": r.to_contract,
            "from_close": r.from_close,
            "to_close": r.to_close,
            "adjustment": r.adjustment,
            "cumulative_adjustment": r.cumulative_adjustment,
            "active_contract_count": n_contracts - (i + 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-symbol pipeline
# ---------------------------------------------------------------------------

def build_all(
    symbols: List[str],
    base_dir: Path,
    out_dir: Path,
    convention: AdjustmentConvention = "subtract",
) -> Dict[str, ContinuousResult]:
    """Build continuous series for multiple symbols and save outputs.

    Parameters
    ----------
    symbols : list of str
        Root symbols (e.g. ``["ES", "CL", "PA"]``).
    base_dir : Path
        Parent directory containing ``{symbol}/`` subdirectories.
    out_dir : Path
        Output directory for continuous CSVs and roll log.
    convention : {"subtract", "add"}
        Adjustment convention (see ``apply_panama_adjustment``).

    Returns
    -------
    dict mapping root symbol → ContinuousResult
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, ContinuousResult] = {}
    all_roll_rows: List[pd.DataFrame] = []

    for sym in symbols:
        sym_dir = Path(base_dir) / sym
        result = build_continuous(sym, sym_dir, convention=convention)
        results[sym] = result

        # Save continuous CSV.
        csv_path = out_dir / f"{sym}_continuous.csv"
        result.continuous.to_csv(csv_path, index=False)

        # Accumulate roll log entries.
        if not result.roll_log.empty:
            rl = result.roll_log.copy()
            rl.insert(0, "root", sym)
            all_roll_rows.append(rl)

    # Save combined roll log.
    if all_roll_rows:
        combined_rl = pd.concat(all_roll_rows, ignore_index=True)
    else:
        combined_rl = pd.DataFrame(
            columns=["root", "date", "from_contract", "to_contract",
                      "from_close", "to_close", "adjustment",
                      "cumulative_adjustment", "active_contract_count"]
        )
    combined_rl.to_csv(out_dir / "roll_log.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# Parity calibration helper
# ---------------------------------------------------------------------------

@dataclass
class ConventionScore:
    """Comparison score for one adjustment convention against a reference."""

    convention: str
    mean_close_diff: float
    max_close_diff: float
    mean_ema10_diff: float
    max_ema10_diff: float
    overlap_bars: int


@dataclass
class CalibrationResult:
    """Result of calibrating adjustment convention for one symbol."""

    symbol: str
    recommended: str  # "subtract" or "add"
    scores: Dict[str, ConventionScore]


def calibrate_convention(
    contracts: Dict[str, pd.DataFrame],
    contract_order: List[ContractSpec],
    reference_df: pd.DataFrame,
    consecutive_days: int = ROLL_CONSECUTIVE_DAYS,
    ema_period: int = 10,
) -> CalibrationResult:
    """Build both conventions and compare against a reference series.

    Parameters
    ----------
    contracts : dict
        ``{symbol: df}`` raw contract data (same format as ``load_contract_data``).
    contract_order : list of ContractSpec
        Sorted contract chain.
    reference_df : pd.DataFrame
        Reference OHLCV with ``Date`` and ``Close`` columns (e.g. TradeStation).
    consecutive_days : int
        Roll trigger threshold.
    ema_period : int
        EMA period for comparison metric.

    Returns
    -------
    CalibrationResult with per-convention scores and recommendation.
    """
    rolls, active_series = detect_rolls(contracts, contract_order, consecutive_days)

    # Parse reference dates.
    ref = reference_df.copy()
    ref["Date"] = pd.to_datetime(ref["Date"])
    ref = ref.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    ref_dates = set(ref["Date"].dt.date)

    scores: Dict[str, ConventionScore] = {}

    for conv in ADJUSTMENT_CONVENTIONS:
        # Re-detect cumulative adjustments for fresh RollEvent state.
        rolls_copy = []
        for r in rolls:
            rolls_copy.append(RollEvent(
                date=r.date,
                from_contract=r.from_contract,
                to_contract=r.to_contract,
                from_close=r.from_close,
                to_close=r.to_close,
                adjustment=r.adjustment,
            ))

        continuous = apply_panama_adjustment(
            contracts, active_series, rolls_copy, convention=conv,
        )

        if continuous.empty:
            scores[conv] = ConventionScore(
                convention=conv,
                mean_close_diff=np.inf,
                max_close_diff=np.inf,
                mean_ema10_diff=np.inf,
                max_ema10_diff=np.inf,
                overlap_bars=0,
            )
            continue

        # Align on overlapping dates.
        cont_dates = set(continuous["Date"].dt.date)
        common = sorted(cont_dates & ref_dates)
        if not common:
            scores[conv] = ConventionScore(
                convention=conv,
                mean_close_diff=np.inf,
                max_close_diff=np.inf,
                mean_ema10_diff=np.inf,
                max_ema10_diff=np.inf,
                overlap_bars=0,
            )
            continue

        # Merge on date.
        cont_keyed = continuous.copy()
        cont_keyed["_date"] = cont_keyed["Date"].dt.date
        ref_keyed = ref.copy()
        ref_keyed["_date"] = ref_keyed["Date"].dt.date

        merged = pd.merge(
            cont_keyed[["_date", "Close"]].rename(columns={"Close": "close_db"}),
            ref_keyed[["_date", "Close"]].rename(columns={"Close": "close_ref"}),
            on="_date",
        ).sort_values("_date").reset_index(drop=True)

        close_diff = (merged["close_db"] - merged["close_ref"]).abs()

        # EMA comparison.
        if len(merged) >= ema_period:
            ema_db = merged["close_db"].ewm(span=ema_period, adjust=False).mean()
            ema_ref = merged["close_ref"].ewm(span=ema_period, adjust=False).mean()
            ema_diff = (ema_db - ema_ref).abs().iloc[ema_period - 1:]
            mean_ema = float(ema_diff.mean())
            max_ema = float(ema_diff.max())
        else:
            mean_ema = float(close_diff.mean())
            max_ema = float(close_diff.max())

        scores[conv] = ConventionScore(
            convention=conv,
            mean_close_diff=round(float(close_diff.mean()), 6),
            max_close_diff=round(float(close_diff.max()), 6),
            mean_ema10_diff=round(mean_ema, 6),
            max_ema10_diff=round(max_ema, 6),
            overlap_bars=len(merged),
        )

    # Recommend the convention with lower mean EMA diff.
    best = min(scores.values(), key=lambda s: s.mean_ema10_diff)

    symbol = contract_order[0].root if contract_order else "?"
    return CalibrationResult(
        symbol=symbol,
        recommended=best.convention,
        scores=scores,
    )
