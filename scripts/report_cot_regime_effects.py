#!/usr/bin/env python3
"""Report OOS COT variant behavior by VIX regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.oos_evaluation import evaluate_oos  # noqa: E402
from ctl.regression import train_model  # noqa: E402


def _latest_dataset(dataset_dir: Path) -> Path:
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
    mapping: Dict[str, Tuple[str, str]] = {
        "baseline_legacy": ("COT_20D_Delta", "COT_ZScore_1Y"),
        "fusion_mean": ("fusion_mean_delta", "fusion_mean_z"),
        "fusion_median": ("fusion_median_delta", "fusion_median_z"),
        "fusion_shrink": ("fusion_shrink_delta", "fusion_shrink_z"),
        "fusion_consensus": ("fusion_consensus_delta", "fusion_consensus_z"),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown variant: {variant}")
    delta_col, z_col = mapping[variant]
    d["COT_20D_Delta"] = d[delta_col]
    d["COT_ZScore_1Y"] = d[z_col]
    return d


def _regime_stats(oos_df: pd.DataFrame, buckets: np.ndarray) -> list[dict]:
    out = []
    labels = [
        ("low_vol", oos_df["VIX_Regime"] == True),  # noqa: E712
        ("high_vol", oos_df["VIX_Regime"] == False),  # noqa: E712
        ("unknown_vix", oos_df["VIX_Regime"].isna()),
    ]
    for label, mask in labels:
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            continue
        s = oos_df.iloc[idx]
        b = buckets[idx]
        top = s.loc[b == "top", "TheoreticalR"]
        bot = s.loc[b == "bottom", "TheoreticalR"]
        spread = float(top.mean() - bot.mean()) if len(top) and len(bot) else np.nan
        out.append(
            {
                "regime": label,
                "n": int(len(s)),
                "avg_r": float(s["TheoreticalR"].mean()),
                "top_avg_r": float(top.mean()) if len(top) else np.nan,
                "bottom_avg_r": float(bot.mean()) if len(bot) else np.nan,
                "spread": spread,
                "win_rate": float((s["TheoreticalR"] > 0).mean()),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Report OOS regime effects for COT fusion variants.")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument(
        "--fusion-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_features_latest.csv",
    )
    p.add_argument(
        "--variants",
        default="baseline_legacy,fusion_mean,fusion_median,fusion_shrink,fusion_consensus",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_regime_effects_latest.json",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    dataset_path = args.dataset or _latest_dataset(
        REPO_ROOT / "data" / "processed" / "cutover_v1" / "datasets"
    )
    base = pd.read_csv(dataset_path)
    fusion = pd.read_csv(args.fusion_csv)
    cols = [
        "Date", "Ticker",
        "fusion_mean_delta", "fusion_mean_z",
        "fusion_median_delta", "fusion_median_z",
        "fusion_shrink_delta", "fusion_shrink_z",
        "fusion_consensus_delta", "fusion_consensus_z",
    ]
    merged = base.merge(fusion[cols], how="left", on=["Date", "Ticker"], validate="one_to_one")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for v in variants:
        d = _variant_dataset(merged, v)
        is_df, oos_df = _split_is_oos(d)
        model = train_model(is_df)
        oos = evaluate_oos(oos_df, model)
        rows.append(
            {
                "variant": v,
                "overall": {
                    "oos_corr": float(oos.score_r_correlation),
                    "oos_spread": float(oos.tercile_spread),
                },
                "regimes": _regime_stats(oos_df, oos.tercile_buckets),
            }
        )

    payload = {
        "dataset_path": str(dataset_path),
        "fusion_csv": str(args.fusion_csv),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json_output:
        print(json.dumps(payload, indent=2))
        return
    print(f"Saved regime report: {args.out}")
    for r in rows:
        print(
            f"{r['variant']}: corr={r['overall']['oos_corr']:.4f} spread={r['overall']['oos_spread']:.4f}"
        )


if __name__ == "__main__":
    main()
