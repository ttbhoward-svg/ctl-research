#!/usr/bin/env python3
"""Run end-to-end COT source benchmark and emit a ranked recommendation.

Pipeline:
1) Build canonical overlays (legacy/disagg/tff)
2) Build strict Phase 1a dataset per source
3) Run confluence ablation per dataset
4) Rank sources and persist JSON artifact
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
OUT_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis"

SOURCES = ("legacy", "disagg", "tff")


def _run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return json.loads(proc.stdout)


def _score_rows(rows: list[dict]) -> list[dict]:
    spreads = [r["full_oos_tercile_spread"] for r in rows]
    corrs = [r["full_oos_score_r_corr"] for r in rows]
    spread_min, spread_max = min(spreads), max(spreads)
    corr_min, corr_max = min(corrs), max(corrs)

    def _norm(v: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0
        return (v - lo) / (hi - lo)

    scored = []
    for row in rows:
        spread_n = _norm(row["full_oos_tercile_spread"], spread_min, spread_max)
        corr_n = _norm(row["full_oos_score_r_corr"], corr_min, corr_max)
        # Favor separation quality (spread) over correlation lift.
        score = (0.7 * spread_n) + (0.3 * corr_n)
        out = dict(row)
        out["rank_score"] = round(score, 6)
        scored.append(out)
    scored.sort(key=lambda x: x["rank_score"], reverse=True)
    return scored


def _find_variant(payload: dict, name: str) -> dict:
    return next(v for v in payload["variants"] if v["variant"] == name)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark COT source variants and rank recommendation.")
    p.add_argument(
        "--sources",
        default="legacy,disagg,tff",
        help="Comma-separated source list (subset of legacy,disagg,tff).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    invalid = [s for s in sources if s not in SOURCES]
    if invalid:
        raise SystemExit(f"Unsupported source(s): {invalid}. Allowed: {SOURCES}")

    overlays = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/build_cot_source_overlays.py",
            "--sources",
            ",".join(sources),
            "--json",
        ]
    )
    overlay_by_source = {x["source"]: x for x in overlays["sources"]}

    rows = []
    for src in sources:
        strict = _run_json(
            [
                str(PYTHON_BIN),
                "scripts/run_phase1a_strict_build.py",
                "--version",
                f"v1_full_universe_real_cot_{src}",
                "--cot-csv",
                str(REPO_ROOT / "data" / "raw" / "external" / f"cot_phase1a_{src}.csv"),
            ]
        )

        ablation = _run_json(
            [
                str(PYTHON_BIN),
                "scripts/evaluate_confluence_ablation.py",
                "--dataset",
                strict["dataset_path"],
                "--json",
            ]
        )
        full = _find_variant(ablation, "full")
        no_cot = _find_variant(ablation, "no_cot")

        rows.append(
            {
                "source": src,
                "cot_rows": overlay_by_source[src]["rows"],
                "cot_symbols": overlay_by_source[src]["symbols"],
                "dataset_path": strict["dataset_path"],
                "dataset_sha256": strict["sha256"],
                "full_is_r2": full["is_r2"],
                "full_oos_score_r_corr": full["oos_score_r_corr"],
                "full_oos_tercile_spread": full["oos_tercile_spread"],
                "full_oos_top_avg_r": full["oos_top_avg_r"],
                "delta_corr_vs_no_cot": round(
                    full["oos_score_r_corr"] - no_cot["oos_score_r_corr"], 6
                ),
                "delta_spread_vs_no_cot": round(
                    full["oos_tercile_spread"] - no_cot["oos_tercile_spread"], 6
                ),
                "delta_top_r_vs_no_cot": round(
                    full["oos_top_avg_r"] - no_cot["oos_top_avg_r"], 6
                ),
            }
        )

    ranked = _score_rows(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (OUT_DIR / f"{ts}_cot_source_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": ts,
        "sources": sources,
        "ranked": ranked,
        "recommended_source": ranked[0]["source"] if ranked else None,
        "recommendation_basis": "0.7*normalized_tercile_spread + 0.3*normalized_oos_corr",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print(f"Saved: {out_path}")
    print()
    print("Source   RankScore  OOS_Corr  Spread    TopR     dCorr(no_cot)  dSpread(no_cot)")
    print("------   ---------  --------  ------    ----     ------------    --------------")
    for r in ranked:
        print(
            f"{r['source']:7s} {r['rank_score']:9.4f}  "
            f"{r['full_oos_score_r_corr']:8.4f}  {r['full_oos_tercile_spread']:8.4f}  "
            f"{r['full_oos_top_avg_r']:7.4f}  {r['delta_corr_vs_no_cot']:12.4f}  "
            f"{r['delta_spread_vs_no_cot']:14.4f}"
        )
    print()
    print(f"Recommended: {payload['recommended_source']}")


if __name__ == "__main__":
    main()
