"""Phase 1a dataset assembler — merges triggers, trade results, and metadata.

Combines B1Trigger fields (confluence, MTFA, external) with TradeResult fields
(outcomes, R-multiples) and universe metadata (cluster, tradable status) into
the canonical output schema per B1_Strategy_Logic_Spec_v2.md §10.

Produces an immutable artifact with deterministic sort order, stable schema,
and SHA-256 hash manifest.

See docs/notes/Task8_assumptions.md for design rationale.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ctl.b1_detector import B1Trigger
from ctl.simulator import TradeResult
from ctl.universe import Universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema — fixed column order for reproducibility
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = [
    # Identification
    "Date",
    "Ticker",
    "Timeframe",
    "SetupType",
    # Trade levels
    "EntryPrice",
    "StopPrice",
    "TP1",
    "TP2",
    "TP3",
    "TP4",
    "TP5",
    # Technical features at trigger
    "BarsOfAir",
    "Slope_20",
    # Confluence flags (Spec §8)
    "WR_Divergence",
    "CleanPullback",
    "VolumeDeclining",
    "FibConfluence",
    "GapFillBelow",
    "MultiYearHighs",
    "SingleBarPullback",
    # MTFA flags (Spec §9)
    "WeeklyTrendAligned",
    "MonthlyTrendAligned",
    # External data (Task 7)
    "COT_20D_Delta",
    "COT_ZScore_1Y",
    "VIX_Regime",
    # Cluster (Spec §10)
    "AssetCluster",
    "TradableStatus",
    # Outcome
    "RMultiple_Actual",
    "TheoreticalR",
    "MFE_R",
    "MAE_R",
    "Day1Fail",
    "TradeOutcome",
    "ExitDate",
    "ExitPrice",
    "ExitReason",
    # Research / meta
    "HoldBars",
    "SameBarCollision",
    "ExitOnLastBar",
    "TP1_Hit",
    "TP2_Hit",
    "TP3_Hit",
    "EntryDate",
    "EntryBarIdx",
    "TriggerBarIdx",
    "SwingHigh",
    "RiskPerUnit",
    "ScoreBucket",
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    triggers: List[B1Trigger],
    results: List[TradeResult],
    universe: Universe,
) -> pd.DataFrame:
    """Merge triggers, trade results, and universe metadata into the final dataset.

    Parameters
    ----------
    triggers : list of B1Trigger
        All detected triggers (confirmed and unconfirmed).  Only those with
        a matching TradeResult appear in the output.
    results : list of TradeResult
        Simulated trade outcomes for confirmed triggers.
    universe : Universe
        Provides AssetCluster and TradableStatus per symbol.

    Returns
    -------
    pd.DataFrame
        One row per trade, columns in ``SCHEMA_COLUMNS`` order,
        sorted by (Date, Ticker) ascending.
    """
    # Build trigger lookup by (symbol, trigger_bar_idx).
    trigger_map: Dict[Tuple[str, int], B1Trigger] = {
        (t.symbol, t.trigger_bar_idx): t for t in triggers
    }

    records: List[Dict] = []
    for r in results:
        key = (r.symbol, r.trigger_bar_idx)
        trig = trigger_map.get(key)

        # Universe metadata.
        sym_info = universe.symbols.get(r.symbol)
        cluster = sym_info.cluster if sym_info else ""
        status = sym_info.status if sym_info else "tradable"

        record = {
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
            # Confluence flags (from trigger, if matched)
            "WR_Divergence": trig.wr_divergence if trig else None,
            "CleanPullback": trig.clean_pullback if trig else None,
            "VolumeDeclining": trig.volume_declining if trig else None,
            "FibConfluence": trig.fib_confluence if trig else None,
            "GapFillBelow": trig.gap_fill_below if trig else None,
            "MultiYearHighs": trig.multi_year_highs if trig else None,
            "SingleBarPullback": trig.single_bar_pullback if trig else None,
            # MTFA flags (from trigger)
            "WeeklyTrendAligned": trig.weekly_trend_aligned if trig else None,
            "MonthlyTrendAligned": trig.monthly_trend_aligned if trig else None,
            # External data (from trigger, set by external_merge)
            "COT_20D_Delta": trig.cot_20d_delta if trig else None,
            "COT_ZScore_1Y": trig.cot_zscore_1y if trig else None,
            "VIX_Regime": trig.vix_regime if trig else None,
            # Cluster
            "AssetCluster": cluster,
            "TradableStatus": status,
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
        }
        records.append(record)

    if not records:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame(records)
    # Enforce stable column order.
    df = df[SCHEMA_COLUMNS]
    # Deterministic sort: Date ascending, then Ticker ascending.
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    logger.info("Assembled dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Hash manifest
# ---------------------------------------------------------------------------

def compute_manifest(df: pd.DataFrame) -> Dict:
    """Compute SHA-256 hash and summary statistics for a dataset.

    The hash is computed on the CSV byte representation to ensure
    deterministic reproducibility regardless of in-memory representation.

    Returns
    -------
    dict with keys: sha256, n_rows, n_columns, columns, date_range.
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    sha256 = hashlib.sha256(csv_bytes).hexdigest()
    date_min = str(df["Date"].min()) if not df.empty else ""
    date_max = str(df["Date"].max()) if not df.empty else ""
    return {
        "sha256": sha256,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "date_range": [date_min, date_max],
    }


# ---------------------------------------------------------------------------
# Save immutable artifact
# ---------------------------------------------------------------------------

def save_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    version: str = "v1",
) -> Tuple[Path, Dict]:
    """Save dataset as immutable CSV artifact with JSON hash manifest.

    File naming: ``phase1a_triggers_{version}_{YYYYMMDD}.csv``
    Never overwrites — new versions get new filenames.

    Returns
    -------
    (csv_path, manifest_dict)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    base = f"phase1a_triggers_{version}_{today}"
    csv_path = out_dir / f"{base}.csv"
    manifest_path = out_dir / f"{base}_manifest.json"

    manifest = compute_manifest(df)

    df.to_csv(csv_path, index=False)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved dataset: %s (%d rows, SHA-256: %s)", csv_path.name, len(df), manifest["sha256"])
    return csv_path, manifest
