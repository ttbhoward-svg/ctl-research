"""Forward-safe PL gap feature model with hierarchical fallbacks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ctl.roll_reconciliation import RollManifestEntry

_CONTRACT_RE = re.compile(r"^[A-Z]+([FGHJKMNQUVXZ])\d+$")


@dataclass(frozen=True)
class FeatureBias:
    level: str
    regime: Optional[str]
    month_code: str
    gap_sign: Optional[int]
    n_rows: int
    median_signed_gap_delta: float
    mean_signed_gap_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureBiasModel:
    exact: Dict[Tuple[str, str, int], float]
    regime_month: Dict[Tuple[str, str], float]
    month: Dict[str, float]
    rows: List[FeatureBias]

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "n_exact": len(self.exact),
            "n_regime_month": len(self.regime_month),
            "n_month": len(self.month),
        }


def _extract_month_code(contract: str) -> Optional[str]:
    m = _CONTRACT_RE.match(str(contract))
    if not m:
        return None
    return m.group(1)


def _regime_label(date_value: pd.Timestamp) -> str:
    if date_value <= pd.Timestamp("2019-12-31"):
        return "pre_2020"
    if date_value >= pd.Timestamp("2024-01-01"):
        return "post_2024"
    return "mid_2020_2023"


def _gap_sign(x: float) -> int:
    return 1 if float(x) >= 0.0 else -1


def estimate_feature_bias_model(
    l2_detail_df: pd.DataFrame,
    train_end_date: str,
    min_rows: int = 2,
) -> FeatureBiasModel:
    """Estimate hierarchical feature biases from L2 training-window rows."""
    if l2_detail_df.empty:
        return FeatureBiasModel(exact={}, regime_month={}, month={}, rows=[])

    work = l2_detail_df.copy()
    mask = (
        work["status"].isin(["PASS", "WATCH"])
        & work["canonical_gap"].notna()
        & work["ts_gap"].notna()
        & work["canonical_date"].notna()
        & work["to_contract"].notna()
    )
    work = work.loc[mask].copy()
    if work.empty:
        return FeatureBiasModel(exact={}, regime_month={}, month={}, rows=[])

    work["canonical_date"] = pd.to_datetime(work["canonical_date"], errors="coerce")
    work = work.dropna(subset=["canonical_date"])
    work = work[work["canonical_date"] <= pd.Timestamp(train_end_date)]
    if work.empty:
        return FeatureBiasModel(exact={}, regime_month={}, month={}, rows=[])

    work["month_code"] = work["to_contract"].map(_extract_month_code)
    work = work.dropna(subset=["month_code"])
    work["regime"] = work["canonical_date"].map(_regime_label)
    work["gap_sign"] = work["canonical_gap"].map(_gap_sign)
    work["signed_gap_delta"] = work["canonical_gap"] - work["ts_gap"]
    if work.empty:
        return FeatureBiasModel(exact={}, regime_month={}, month={}, rows=[])

    rows: List[FeatureBias] = []
    exact: Dict[Tuple[str, str, int], float] = {}
    regime_month: Dict[Tuple[str, str], float] = {}
    month: Dict[str, float] = {}

    g1 = work.groupby(["regime", "month_code", "gap_sign"], sort=True)
    for (reg, mon, sign), g in g1:
        n = int(len(g))
        if n < min_rows:
            continue
        med = float(g["signed_gap_delta"].median())
        exact[(str(reg), str(mon), int(sign))] = med
        rows.append(
            FeatureBias(
                level="exact",
                regime=str(reg),
                month_code=str(mon),
                gap_sign=int(sign),
                n_rows=n,
                median_signed_gap_delta=med,
                mean_signed_gap_delta=float(g["signed_gap_delta"].mean()),
            )
        )

    g2 = work.groupby(["regime", "month_code"], sort=True)
    for (reg, mon), g in g2:
        n = int(len(g))
        if n < min_rows:
            continue
        med = float(g["signed_gap_delta"].median())
        regime_month[(str(reg), str(mon))] = med
        rows.append(
            FeatureBias(
                level="regime_month",
                regime=str(reg),
                month_code=str(mon),
                gap_sign=None,
                n_rows=n,
                median_signed_gap_delta=med,
                mean_signed_gap_delta=float(g["signed_gap_delta"].mean()),
            )
        )

    g3 = work.groupby(["month_code"], sort=True)
    for mon, g in g3:
        month_code = mon[0] if isinstance(mon, tuple) else mon
        n = int(len(g))
        if n < min_rows:
            continue
        med = float(g["signed_gap_delta"].median())
        month[str(month_code)] = med
        rows.append(
            FeatureBias(
                level="month",
                regime=None,
                month_code=str(month_code),
                gap_sign=None,
                n_rows=n,
                median_signed_gap_delta=med,
                mean_signed_gap_delta=float(g["signed_gap_delta"].mean()),
            )
        )

    return FeatureBiasModel(exact=exact, regime_month=regime_month, month=month, rows=rows)


def apply_feature_bias_model(
    manifest_entries: Iterable[RollManifestEntry],
    model: FeatureBiasModel,
    apply_start_date: Optional[str] = None,
) -> List[RollManifestEntry]:
    """Apply hierarchical feature bias correction to manifest entries."""
    start_ts = pd.Timestamp(apply_start_date) if apply_start_date else None

    out: List[RollManifestEntry] = []
    cumulative = 0.0
    for e in manifest_entries:
        d = pd.Timestamp(e.roll_date)
        bias = 0.0
        mon = _extract_month_code(e.to_contract)
        if mon is not None:
            reg = _regime_label(d)
            sign = _gap_sign(e.gap)
            if start_ts is None or d >= start_ts:
                key_exact = (reg, mon, sign)
                key_rm = (reg, mon)
                if key_exact in model.exact:
                    bias = model.exact[key_exact]
                elif key_rm in model.regime_month:
                    bias = model.regime_month[key_rm]
                elif mon in model.month:
                    bias = model.month[mon]

        new_gap = float(e.gap) - bias
        cumulative += new_gap
        out.append(
            RollManifestEntry(
                roll_date=e.roll_date,
                from_contract=e.from_contract,
                to_contract=e.to_contract,
                from_close=float(e.from_close),
                to_close=float(e.from_close + new_gap),
                gap=float(new_gap),
                cumulative_adj=float(cumulative),
                trigger_reason=e.trigger_reason,
                confirmation_days=e.confirmation_days,
                convention=e.convention,
                session_template=e.session_template,
                close_type=e.close_type,
            )
        )
    return out
