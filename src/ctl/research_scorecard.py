"""Research-tier confidence scorecard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ctl.canonical_acceptance import acceptance_from_diagnostics
from ctl.cutover_diagnostics import run_diagnostics
from ctl.operating_profile import discover_ts_custom_file
from ctl.parity_prep import load_and_validate
from ctl.research_registry import ResearchTickerRegistry
from ctl.roll_reconciliation import load_roll_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_CONTINUOUS_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"
DEFAULT_RESEARCH_RUN_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "research_runs"


@dataclass(frozen=True)
class ResearchScorecardRow:
    symbol: str
    run_status: str
    trigger_count: Optional[int]
    trade_count: Optional[int]
    total_r: Optional[float]
    win_rate: Optional[float]
    mtfa_weekly_rate: Optional[float]
    mtfa_monthly_rate: Optional[float]
    diagnostics_available: bool
    diagnostics_status: str
    decision: Optional[str]
    mean_gap_diff: Optional[float]
    mean_drift: Optional[float]
    confidence_score: float
    confidence_band: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_latest_research_batch(summary_dir: Path = DEFAULT_RESEARCH_RUN_DIR) -> Optional[dict]:
    summary_dir = Path(summary_dir)
    files = sorted(summary_dir.glob("*_research_batch.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _extract_symbol_rows(batch_summary: dict) -> List[dict]:
    # Supports both wrapped payload and raw saved summary JSON.
    if "summary" in batch_summary and isinstance(batch_summary.get("summary"), dict):
        return list(batch_summary.get("summary", {}).get("symbol_results", []))
    return list(batch_summary.get("symbol_results", []))


def _load_ohlcv_with_vol_alias(path: Path, label: str):
    df, errs = load_and_validate(path, label)
    if not errs:
        return df, errs
    # Handle TS files that use Vol instead of Volume.
    try:
        import pandas as pd  # local import to avoid module-global dependency expansion
        raw = pd.read_csv(path)
        col_map = {c: c.strip().title() for c in raw.columns}
        raw = raw.rename(columns=col_map)
        if "Vol" in raw.columns and "Volume" not in raw.columns:
            raw = raw.rename(columns={"Vol": "Volume"})
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if required.issubset(set(raw.columns)):
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
            raw = raw.dropna(subset=["Date"])
            raw = raw.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
            return raw, []
    except Exception:
        pass
    return None, errs


def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _compute_score(run_row: dict, diag_decision: Optional[str], diag_available: bool) -> float:
    status = str(run_row.get("status", "")).upper()
    if status != "EXECUTED":
        return 0.10

    base = 0.30
    trade_count = int(run_row.get("trade_count") or 0)
    win_rate = float(run_row.get("win_rate") or 0.0)
    total_r = float(run_row.get("total_r") or 0.0)

    # Execution quality component.
    base += min(trade_count, 10) * 0.02
    base += min(max(win_rate, 0.0), 1.0) * 0.20
    if total_r > 0:
        base += 0.10

    # Diagnostics component.
    if diag_available:
        if diag_decision == "ACCEPT":
            base += 0.25
        elif diag_decision == "WATCH":
            base += 0.15
        elif diag_decision == "REJECT":
            base -= 0.35
    else:
        base += 0.05  # execution-only track

    return max(0.0, min(base, 1.0))


def _run_optional_diagnostics(
    symbol: str,
    tick_size: Optional[float],
    max_day_delta: Optional[int],
    db_dir: Path,
    ts_dir: Path,
) -> Dict[str, object]:
    # Need explicit metadata + futures-style files.
    if tick_size is None or max_day_delta is None:
        return {"available": False, "status": "SKIP", "note": "missing tick_size/max_day_delta"}

    can_path = db_dir / f"{symbol}_continuous.csv"
    manifest_path = db_dir / f"{symbol}_roll_manifest.json"
    if not can_path.is_file() or not manifest_path.is_file():
        return {"available": False, "status": "SKIP", "note": "missing continuous/manifest"}

    ts_adj_path = discover_ts_custom_file(symbol, ts_dir, "ADJ")
    ts_unadj_path = discover_ts_custom_file(symbol, ts_dir, "UNADJ")
    if ts_adj_path is None or ts_unadj_path is None:
        return {"available": False, "status": "SKIP", "note": "missing TS CUSTOM ADJ/UNADJ"}

    can_df, e0 = _load_ohlcv_with_vol_alias(can_path, f"DB {symbol}")
    ts_adj_df, e1 = _load_ohlcv_with_vol_alias(ts_adj_path, f"TS {symbol} ADJ")
    ts_unadj_df, e2 = _load_ohlcv_with_vol_alias(ts_unadj_path, f"TS {symbol} UNADJ")
    if e0 or e1 or e2:
        return {"available": False, "status": "ERROR", "note": "; ".join(e0 + e1 + e2)}

    manifest = load_roll_manifest(manifest_path)
    diag = run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest,
        ts_unadj_df=ts_unadj_df,
        symbol=symbol,
        tick_size=float(tick_size),
        max_day_delta=int(max_day_delta),
    )
    acc = acceptance_from_diagnostics(diag)
    return {
        "available": True,
        "status": f"{diag.strict_status}/{diag.policy_status}",
        "decision": acc.decision,
        "mean_gap_diff": float(diag.l3.mean_gap_diff),
        "mean_drift": float(diag.l4.mean_drift),
        "note": "; ".join(acc.reasons) if acc.reasons else "",
    }


def build_research_scorecard(
    registry: ResearchTickerRegistry,
    batch_summary: dict,
    db_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    ts_dir: Path = DEFAULT_TS_DIR,
) -> List[ResearchScorecardRow]:
    by_symbol = {row.get("symbol"): row for row in _extract_symbol_rows(batch_summary)}
    reg_map = {s.symbol: s for s in registry.symbols}

    out: List[ResearchScorecardRow] = []
    for sym in registry.enabled_symbols():
        run_row = by_symbol.get(sym, {"symbol": sym, "status": "MISSING"})
        rs = reg_map[sym]

        d = _run_optional_diagnostics(
            symbol=sym,
            tick_size=rs.tick_size,
            max_day_delta=rs.max_day_delta,
            db_dir=db_dir,
            ts_dir=ts_dir,
        )

        score = _compute_score(
            run_row=run_row,
            diag_decision=d.get("decision"),
            diag_available=bool(d.get("available")),
        )
        out.append(
            ResearchScorecardRow(
                symbol=sym,
                run_status=str(run_row.get("status", "MISSING")),
                trigger_count=run_row.get("trigger_count"),
                trade_count=run_row.get("trade_count"),
                total_r=run_row.get("total_r"),
                win_rate=run_row.get("win_rate"),
                mtfa_weekly_rate=run_row.get("mtfa_weekly_rate"),
                mtfa_monthly_rate=run_row.get("mtfa_monthly_rate"),
                diagnostics_available=bool(d.get("available")),
                diagnostics_status=str(d.get("status", "SKIP")),
                decision=d.get("decision"),
                mean_gap_diff=d.get("mean_gap_diff"),
                mean_drift=d.get("mean_drift"),
                confidence_score=round(score, 4),
                confidence_band=_confidence_band(score),
                notes=str(d.get("note", "")),
            )
        )
    return out
