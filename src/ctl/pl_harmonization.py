"""PL harmonization helpers for integrated evaluation paths.

This module wires previously offline correction prototypes into a single
optional transform that can be used by evaluation scripts without changing
default behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from ctl.pl_basis_correction import apply_regime_offsets, derive_regime_offsets
from ctl.pl_gap_correction import apply_gap_bias, estimate_gap_bias
from ctl.pl_gap_window_correction import apply_window_biases, select_top_window_biases
from ctl.roll_reconciliation import RollManifestEntry


PL_HARMONIZATION_MODES = (
    "none",
    "drift_only",
    "gap_bias",
    "combined",
    "window_gap",
    "window_combined",
)


DEFAULT_PL_REGIMES: tuple[tuple[str, str, str], ...] = (
    ("pre_2020", "2018-01-01", "2019-12-31"),
    ("post_2024", "2024-01-01", "2026-02-17"),
)

PL_REGIME_PRESETS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "legacy": DEFAULT_PL_REGIMES,
    "yearly_2020_2022": (
        ("pre_2020", "2018-01-01", "2019-12-31"),
        ("y2020", "2020-01-01", "2020-12-31"),
        ("y2021", "2021-01-01", "2021-12-31"),
        ("y2022", "2022-01-01", "2022-12-31"),
        ("post_2023", "2023-01-01", "2026-02-17"),
    ),
}


@dataclass(frozen=True)
class PLHarmonizationResult:
    mode: str
    top_k: int
    regime_offsets: List[dict]
    signed_gap_bias: float
    gap_bias_rows: int
    window_biases: List[dict]
    regime_preset: str = "legacy"

    def to_dict(self) -> dict:
        return asdict(self)


def apply_pl_harmonization(
    canonical_df: pd.DataFrame,
    manifest_entries: List[RollManifestEntry],
    ts_adj_df: pd.DataFrame,
    l2_detail_df: pd.DataFrame,
    *,
    mode: str = "none",
    top_k: int = 5,
    regimes: Sequence[Tuple[str, str, str]] = DEFAULT_PL_REGIMES,
    regime_preset: str = "legacy",
) -> tuple[pd.DataFrame, List[RollManifestEntry], PLHarmonizationResult]:
    """Apply optional PL harmonization transforms.

    Parameters
    ----------
    canonical_df
        Canonical adjusted series with Date/Close.
    manifest_entries
        Canonical roll manifest entries.
    ts_adj_df
        TradeStation adjusted reference used for signed-diff regimes.
    l2_detail_df
        L2 comparison table used for gap-bias estimation.
    mode
        Harmonization mode; defaults to ``none``.
    top_k
        Window count for ``window_gap``/``window_combined``.
    regimes
        Regime windows for drift harmonization.
    """
    if mode not in PL_HARMONIZATION_MODES:
        raise ValueError(f"Unknown PL harmonization mode '{mode}'")

    can_out = canonical_df.copy()
    manifest_out = list(manifest_entries)
    regime_offsets: List[dict] = []
    signed_gap_bias = 0.0
    gap_bias_rows = 0
    window_biases: List[dict] = []

    if mode in ("drift_only", "combined", "window_combined"):
        offsets = derive_regime_offsets(can_out, ts_adj_df, list(regimes))
        can_out = apply_regime_offsets(can_out, offsets)
        regime_offsets = [o.to_dict() for o in offsets]

    if mode in ("gap_bias", "combined"):
        gap_bias = estimate_gap_bias(l2_detail_df)
        manifest_out = apply_gap_bias(manifest_out, gap_bias.median_signed_gap_delta)
        signed_gap_bias = float(gap_bias.median_signed_gap_delta)
        gap_bias_rows = int(gap_bias.n_rows)

    if mode in ("window_gap", "window_combined"):
        biases = select_top_window_biases(l2_detail_df, top_k=top_k)
        manifest_out = apply_window_biases(manifest_out, biases)
        window_biases = [b.to_dict() for b in biases]

    meta = PLHarmonizationResult(
        mode=mode,
        top_k=top_k,
        regime_offsets=regime_offsets,
        signed_gap_bias=signed_gap_bias,
        gap_bias_rows=gap_bias_rows,
        window_biases=window_biases,
        regime_preset=regime_preset,
    )
    return can_out, manifest_out, meta


def resolve_pl_regimes(preset: str) -> tuple[tuple[str, str, str], ...]:
    """Resolve a named PL regime preset into date windows."""
    if preset not in PL_REGIME_PRESETS:
        raise ValueError(f"Unknown PL regime preset '{preset}'")
    return PL_REGIME_PRESETS[preset]
