#!/usr/bin/env python3
"""Build canonical COT CSV overlays from legacy/disaggregated/TFF sources.

Outputs canonical files with schema:
  publication_date,symbol,commercial_net
for use by the existing cot_loader pipeline.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "data" / "raw" / "external" / "cot"
OUT_DIR = REPO_ROOT / "data" / "raw" / "external"


SYMBOL_PATTERNS = {
    "/ES": ["E-MINI S&P 500"],
    "/CL": ["CRUDE OIL, LIGHT SWEET"],
    "/PA": ["PALLADIUM"],
    "/PL": ["PLATINUM"],
    "/GC": ["GOLD"],
    "/HG": ["COPPER"],
    "/NG": ["NATURAL GAS"],
    "/NQ": ["E-MINI NASDAQ-100"],
    "/RTY": ["E-MINI RUSSELL 2000", "RUSSELL 2000"],
    "/SI": ["SILVER"],
    "/YM": ["DJIA", "DOW JONES"],
    "/ZB": ["U.S. TREASURY BONDS", "TREASURY BONDS"],
    "/ZN": ["10-YEAR U.S. TREASURY NOTES", "10-YEAR TREASURY NOTES"],
    "/ZC": ["CORN"],
    "/ZS": ["SOYBEANS"],
}


def _load_csv_from_zip(zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        resolved = member_name
        if resolved not in names:
            candidates = [n for n in names if n.lower().endswith(".txt")]
            if len(candidates) == 1:
                resolved = candidates[0]
            else:
                raise ValueError(
                    f"{zip_path}: expected member '{member_name}' not found; members={sorted(names)}"
                )
        with zf.open(resolved) as f:
            return pd.read_csv(
                f,
                encoding="utf-8",
                low_memory=False,
            )


def _load_legacy_frames(dir_path: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for p in sorted(dir_path.glob("*.txt")):
        frames.append(pd.read_csv(p))
    return frames


def _load_disagg_frames(dir_path: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for p in sorted(dir_path.glob("*.zip")):
        frames.append(_load_csv_from_zip(p, "c_year.txt"))
    return frames


def _load_tff_frames(dir_path: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for p in sorted(dir_path.glob("*.zip")):
        frames.append(_load_csv_from_zip(p, "FinFutYY.txt"))
    return frames


def _parse_source(
    raw: pd.DataFrame,
    *,
    source: str,
    name_col: str,
    date_col: str,
    long_col: str,
    short_col: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["publication_date", "symbol", "commercial_net"])

    for c in [name_col, date_col, long_col, short_col]:
        if c not in raw.columns:
            raise ValueError(f"{source}: missing required column: {c}")

    name = raw[name_col].astype(str).str.upper()
    rows = []
    for sym, patterns in SYMBOL_PATTERNS.items():
        mask = False
        for pat in patterns:
            mask = mask | name.str.contains(pat, na=False)
        m = raw[mask].copy()
        if m.empty:
            continue
        m["publication_date"] = pd.to_datetime(m[date_col], errors="coerce")
        m["commercial_net"] = (
            pd.to_numeric(m[long_col], errors="coerce")
            - pd.to_numeric(m[short_col], errors="coerce")
        )
        m["symbol"] = sym
        rows.append(m[["publication_date", "symbol", "commercial_net"]])

    if not rows:
        return pd.DataFrame(columns=["publication_date", "symbol", "commercial_net"])

    out = pd.concat(rows, ignore_index=True).dropna(subset=["publication_date", "commercial_net"])
    out = out.sort_values(["symbol", "publication_date"]).drop_duplicates(["symbol", "publication_date"])
    return out.reset_index(drop=True)


def _build_overlay(source: str, base_dir: Path) -> pd.DataFrame:
    if source == "legacy":
        frames = _load_legacy_frames(base_dir / "legacy_futures_only")
        raw = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
        return _parse_source(
            raw,
            source=source,
            name_col="Market and Exchange Names",
            date_col="As of Date in Form YYYY-MM-DD",
            long_col="Commercial Positions-Long (All)",
            short_col="Commercial Positions-Short (All)",
        )

    if source == "disagg":
        frames = _load_disagg_frames(base_dir / "disaggregated_futures_only")
        raw = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
        # Use Producer/Merchant net as commercial proxy in disaggregated report.
        return _parse_source(
            raw,
            source=source,
            name_col="Market_and_Exchange_Names",
            date_col="Report_Date_as_YYYY-MM-DD",
            long_col="Prod_Merc_Positions_Long_All",
            short_col="Prod_Merc_Positions_Short_All",
        )

    if source == "tff":
        frames = _load_tff_frames(base_dir / "tff_futures_only")
        raw = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
        # TFF has no commercial bucket; use Dealer net as financial-market proxy.
        return _parse_source(
            raw,
            source=source,
            name_col="Market_and_Exchange_Names",
            date_col="Report_Date_as_YYYY-MM-DD",
            long_col="Dealer_Positions_Long_All",
            short_col="Dealer_Positions_Short_All",
        )

    raise ValueError(f"Unknown source: {source}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build canonical COT overlays by source.")
    p.add_argument(
        "--sources",
        type=str,
        default="legacy,disagg,tff",
        help="Comma-separated sources: legacy,disagg,tff",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    payload = {"sources": []}
    for src in sources:
        df = _build_overlay(src, BASE_DIR)
        out = OUT_DIR / f"cot_phase1a_{src}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        payload["sources"].append(
            {
                "source": src,
                "path": str(out),
                "rows": int(len(df)),
                "symbols": int(df["symbol"].nunique() if not df.empty else 0),
            }
        )

    if args.json_output:
        print(json.dumps(payload, indent=2))
    else:
        for s in payload["sources"]:
            print(f"{s['source']:8s} rows={s['rows']:5d} symbols={s['symbols']:2d} -> {s['path']}")


if __name__ == "__main__":
    main()
