#!/usr/bin/env python3
"""Build immutable Phase 1a trigger dataset with health checks.

Tracker mapping:
- Task 7: external COT/VIX merge with strict lag
- Task 8: final DB assembly + health check + immutable artifact + SHA manifest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.b1_detector import run_b1_detection  # noqa: E402
from ctl.cot_loader import load_and_compute as load_cot_and_compute  # noqa: E402
from ctl.dataset_assembler import assemble_dataset, save_dataset  # noqa: E402
from ctl.external_merge import merge_external_features  # noqa: E402
from ctl.health_check import run_health_checks  # noqa: E402
from ctl.parity_prep import discover_ts_file, load_and_validate  # noqa: E402
from ctl.run_orchestrator import _build_htf_ohlcv  # noqa: E402
from ctl.simulator import SimConfig, simulate_trade  # noqa: E402
from ctl.universe import Universe  # noqa: E402
from ctl.vix_loader import load_and_compute as load_vix_and_compute  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_phase1a_dataset")

DEFAULT_DB_CONTINUOUS_DIR = (
    REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
)
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "datasets"


def _base_symbol(symbol: str) -> str:
    return symbol.lstrip("/$")


def _load_symbol_ohlcv(symbol: str, data_dir: Path, ts_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    base = _base_symbol(symbol)
    csv_path = data_dir / f"{base}_continuous.csv"
    df, errors = load_and_validate(csv_path, f"DB {symbol}")
    if not errors:
        return df, None

    ts_path = discover_ts_file(base, ts_dir)
    if ts_path is None:
        return None, f"{symbol}: no continuous CSV and no TS fallback"

    ts_df, ts_errors = load_and_validate(ts_path, f"TS {symbol}")
    if not ts_errors:
        return ts_df, None

    # TS fallback alias for Vol -> Volume.
    try:
        raw = pd.read_csv(ts_path)
        col_map = {c: c.strip().title() for c in raw.columns}
        raw = raw.rename(columns=col_map)
        if "Vol" in raw.columns and "Volume" not in raw.columns:
            raw = raw.rename(columns={"Vol": "Volume"})
        req = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if req.issubset(set(raw.columns)):
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
            raw = raw.dropna(subset=["Date"])
            raw = raw.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
            return raw, None
    except Exception:
        pass
    return None, f"{symbol}: TS fallback load failed ({'; '.join(ts_errors)})"


def _simulate_symbol(symbol: str, df: pd.DataFrame, timeframe: str, slippage: float):
    weekly_df = _build_htf_ohlcv(df, "weekly")
    monthly_df = _build_htf_ohlcv(df, "monthly")
    detect_df = df if timeframe == "daily" else weekly_df.copy()
    if len(detect_df) < 50:
        return [], [], f"{symbol}: insufficient {timeframe} bars ({len(detect_df)})"

    triggers = run_b1_detection(
        detect_df,
        symbol,
        timeframe,
        weekly_df=weekly_df,
        monthly_df=monthly_df,
    )
    confirmed = sorted(
        [t for t in triggers if t.confirmed and t.entry_bar_idx is not None],
        key=lambda t: int(t.entry_bar_idx),
    )

    selected = []
    results = []
    last_exit_idx = -1
    cfg = SimConfig(slippage_per_side=slippage)
    for trig in confirmed:
        if int(trig.entry_bar_idx) <= last_exit_idx:
            continue
        trade = simulate_trade(trig, detect_df, cfg)
        if trade is None:
            continue
        selected.append(trig)
        results.append(trade)
        last_exit_idx = int(trade.exit_bar_idx)
    return selected, results, None


def main() -> None:
    p = argparse.ArgumentParser(description="Build immutable Phase 1a dataset")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols; empty=all universe")
    p.add_argument("--timeframe", type=str, default="daily", choices=["daily", "weekly"])
    p.add_argument("--version", type=str, default="v1")
    p.add_argument("--db-dir", type=Path, default=DEFAULT_DB_CONTINUOUS_DIR)
    p.add_argument("--ts-dir", type=Path, default=DEFAULT_TS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cot-csv", type=Path, default=None, help="Optional COT CSV path")
    p.add_argument("--vix-csv", type=Path, default=None, help="Optional VIX CSV path")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    universe = Universe.from_yaml()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols.strip()
        else list(universe.all_symbols)
    )

    cot_features = load_cot_and_compute(args.cot_csv) if args.cot_csv else None
    vix_data = load_vix_and_compute(args.vix_csv) if args.vix_csv else None

    all_triggers = []
    all_results = []
    warnings = []

    for sym in symbols:
        df, err = _load_symbol_ohlcv(sym, args.db_dir, args.ts_dir)
        if err:
            warnings.append(err)
            continue
        slippage = 0.0
        trigs, results, sim_err = _simulate_symbol(sym, df, args.timeframe, slippage)
        if sim_err:
            warnings.append(sim_err)
            continue
        trigs = merge_external_features(trigs, cot_features, vix_data, universe)
        all_triggers.extend(trigs)
        all_results.extend(results)

    dataset = assemble_dataset(all_triggers, all_results, universe)
    report = run_health_checks(dataset, universe)
    csv_path, manifest = save_dataset(dataset, args.out_dir, version=args.version)

    payload = {
        "dataset_path": str(csv_path),
        "version": args.version,
        "timeframe": args.timeframe,
        "symbols_requested": len(symbols),
        "triggers": len(all_triggers),
        "trades": len(all_results),
        "rows": len(dataset),
        "sha256": manifest["sha256"],
        "health_all_passed": report.all_passed,
        "health_passed": report.n_passed,
        "health_failed": report.n_failed,
        "warnings": warnings,
    }
    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("Phase1a Dataset Build")
    print(f"dataset: {csv_path}")
    print(f"rows/trades/triggers: {len(dataset)}/{len(all_results)}/{len(all_triggers)}")
    print(f"sha256: {manifest['sha256']}")
    print(f"health: {'PASS' if report.all_passed else 'FAIL'} ({report.n_passed} passed, {report.n_failed} failed)")
    if warnings:
        print("warnings:")
        for w in warnings[:20]:
            print(f" - {w}")


if __name__ == "__main__":
    main()
