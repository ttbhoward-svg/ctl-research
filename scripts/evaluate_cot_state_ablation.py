#!/usr/bin/env python3
"""Evaluate state-model COT replacement vs baseline and fusion_mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.oos_evaluation import evaluate_oos  # noqa: E402
from ctl.regression import train_model  # noqa: E402


def _latest_dataset(dataset_dir: Path) -> Path:
    legacy = sorted(dataset_dir.glob("phase1a_triggers_v1_full_universe_real_cot_legacy_*.csv"))
    if legacy:
        return legacy[-1]
    files = sorted(dataset_dir.glob("phase1a_triggers_v1_full_universe_real*.csv"))
    if not files:
        raise FileNotFoundError(f"No strict dataset found under: {dataset_dir}")
    return files[-1]


def _split_is_oos(df: pd.DataFrame, is_end: str = "2024-12-31") -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    is_mask = data["Date"] <= pd.Timestamp(is_end)
    return (
        data.loc[is_mask].reset_index(drop=True),
        data.loc[~is_mask].reset_index(drop=True),
    )


def _variant_dataset(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    d = df.copy()
    if variant == "baseline_legacy":
        return d
    if variant == "fusion_mean":
        d["COT_20D_Delta"] = d["fusion_mean_delta"]
        d["COT_ZScore_1Y"] = d["fusion_mean_z"]
        return d
    if variant == "fusion_mean_state":
        d["COT_20D_Delta"] = d["state_score_delta"]
        d["COT_ZScore_1Y"] = d["state_score_z"]
        return d
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate COT state-model variant.")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument(
        "--fusion-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_features_latest.csv",
    )
    p.add_argument(
        "--state-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_state_features_latest.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_state_ablation_latest.json",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    dataset_path = args.dataset or _latest_dataset(
        REPO_ROOT / "data" / "processed" / "cutover_v1" / "datasets"
    )
    base = pd.read_csv(dataset_path)
    fusion = pd.read_csv(args.fusion_csv)[
        ["Date", "Ticker", "fusion_mean_delta", "fusion_mean_z"]
    ]
    state = pd.read_csv(args.state_csv)[
        ["Date", "Ticker", "state_score_delta", "state_score_z"]
    ]
    merged = base.merge(fusion, how="left", on=["Date", "Ticker"], validate="one_to_one")
    merged = merged.merge(state, how="left", on=["Date", "Ticker"], validate="one_to_one")

    variants = ["baseline_legacy", "fusion_mean", "fusion_mean_state"]
    rows = []
    baseline = None
    fusion_mean = None
    for v in variants:
        d = _variant_dataset(merged, v)
        is_df, oos_df = _split_is_oos(d)
        model = train_model(is_df)
        oos = evaluate_oos(oos_df, model)
        rec: Dict[str, float | str] = {
            "variant": v,
            "is_r2": round(model.diagnostics.r_squared_is, 6),
            "oos_corr": round(oos.score_r_correlation, 6),
            "oos_spread": round(oos.tercile_spread, 6),
            "oos_top_avg_r": round(
                next((b.avg_r for b in oos.tercile_table if b.label == "top"), 0.0), 6
            ),
        }
        rows.append(rec)
        if v == "baseline_legacy":
            baseline = rec
        if v == "fusion_mean":
            fusion_mean = rec

    assert baseline is not None and fusion_mean is not None
    for r in rows:
        r["lift_corr_vs_baseline"] = round(float(r["oos_corr"]) - float(baseline["oos_corr"]), 6)
        r["lift_spread_vs_baseline"] = round(float(r["oos_spread"]) - float(baseline["oos_spread"]), 6)
        r["lift_top_r_vs_baseline"] = round(float(r["oos_top_avg_r"]) - float(baseline["oos_top_avg_r"]), 6)
        r["lift_corr_vs_fusion_mean"] = round(float(r["oos_corr"]) - float(fusion_mean["oos_corr"]), 6)
        r["lift_spread_vs_fusion_mean"] = round(float(r["oos_spread"]) - float(fusion_mean["oos_spread"]), 6)
        r["lift_top_r_vs_fusion_mean"] = round(
            float(r["oos_top_avg_r"]) - float(fusion_mean["oos_top_avg_r"]), 6
        )

    payload = {
        "dataset_path": str(dataset_path),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
