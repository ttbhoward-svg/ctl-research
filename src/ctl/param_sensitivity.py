"""Parameter sensitivity sweep for Phase 1a B1 detection parameters.

Sweeps pre-registered parameter ranges to identify robustness plateaus
and flag fragile single-point optima.  All analysis is IS-only.

See docs/notes/Task11_assumptions.md for design rationale.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ctl.b1_detector import B1Params, run_b1_detection
from ctl.simulator import SimConfig, TradeResult, simulate_all

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default sweep grid (pre-registered ranges from pre_registration_v1.yaml)
# ---------------------------------------------------------------------------

DEFAULT_SWEEP: Dict[str, List] = {
    "slope_threshold": [4, 6, 8, 10, 12, 14],
    "min_bars_of_air": [3, 4, 5, 6, 7, 8, 9, 10],
    "entry_grace_bars": [1, 2, 3, 4, 5],
    "swing_lookback_daily": [15, 20, 25],
}

PARAM_NAMES = list(DEFAULT_SWEEP.keys())


# ---------------------------------------------------------------------------
# Sweep metrics
# ---------------------------------------------------------------------------

@dataclass
class SweepMetrics:
    """Summary metrics for one parameter combination."""

    params: Dict[str, float]
    n_trades: int
    avg_r: float
    total_r: float
    win_rate: float
    max_dd_r: float
    mar_proxy: float


@dataclass
class RobustnessResult:
    """Robustness analysis for one parameter combination."""

    params: Dict[str, float]
    total_r: float
    robustness_score: float       # mean neighbor total_r / own total_r
    n_neighbors: int
    is_fragile: bool


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_grid(
    sweep: Optional[Dict[str, List]] = None,
    max_combos: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Generate parameter grid from sweep specification.

    Parameters
    ----------
    sweep : dict mapping parameter name -> list of values.
        Defaults to DEFAULT_SWEEP.
    max_combos : int or None
        If set, subsample the grid to at most this many combos
        (deterministic, evenly spaced).

    Returns
    -------
    List of dicts, each mapping param name -> value.
    """
    if sweep is None:
        sweep = DEFAULT_SWEEP

    names = list(sweep.keys())
    value_lists = [sweep[n] for n in names]

    grid = [
        dict(zip(names, combo))
        for combo in itertools.product(*value_lists)
    ]

    if max_combos is not None and len(grid) > max_combos:
        step = len(grid) / max_combos
        indices = [int(i * step) for i in range(max_combos)]
        grid = [grid[i] for i in indices]

    return grid


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _make_params(combo: Dict[str, float], base: Optional[B1Params] = None) -> B1Params:
    """Create B1Params from a sweep combo, filling non-swept from base."""
    p = base or B1Params()
    return B1Params(
        ema_period=p.ema_period,
        slope_lookback=p.slope_lookback,
        slope_threshold=combo.get("slope_threshold", p.slope_threshold),
        min_bars_of_air=int(combo.get("min_bars_of_air", p.min_bars_of_air)),
        max_bars_of_air_lookback=p.max_bars_of_air_lookback,
        breakdown_buffer_atr=p.breakdown_buffer_atr,
        entry_grace_bars=int(combo.get("entry_grace_bars", p.entry_grace_bars)),
        swing_lookback_daily=int(combo.get("swing_lookback_daily", p.swing_lookback_daily)),
        swing_lookback_weekly=p.swing_lookback_weekly,
        atr_period=p.atr_period,
        williams_r_period=p.williams_r_period,
        gap_scan_window=p.gap_scan_window,
    )


def _max_drawdown_r(r_values: Sequence[float]) -> float:
    """Max drawdown from a sequence of per-trade R-multiples."""
    if len(r_values) == 0:
        return 0.0
    cumulative = np.cumsum(r_values)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown))


def run_single_combo(
    combo: Dict[str, float],
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    timeframe: str = "daily",
    sim_config: Optional[SimConfig] = None,
    base_params: Optional[B1Params] = None,
) -> SweepMetrics:
    """Run detection + simulation for one parameter combination.

    Parameters
    ----------
    combo : dict
        Parameter name -> value for this grid point.
    data_by_symbol : dict
        Symbol -> OHLCV DataFrame.
    symbols : list of str
        Symbols to run detection on.
    timeframe : str
        Timeframe label (default "daily").
    sim_config : SimConfig or None
        Simulation config (default: no slippage).
    base_params : B1Params or None
        Base parameters for non-swept fields.

    Returns
    -------
    SweepMetrics for this parameter combo.
    """
    params = _make_params(combo, base_params)
    config = sim_config or SimConfig()

    all_results: List[TradeResult] = []
    for sym in symbols:
        df = data_by_symbol.get(sym)
        if df is None or len(df) < 50:
            continue
        triggers = run_b1_detection(df, sym, timeframe, params)
        results = simulate_all(triggers, df, config)
        all_results.extend(results)

    n_trades = len(all_results)
    if n_trades == 0:
        return SweepMetrics(
            params=dict(combo),
            n_trades=0,
            avg_r=0.0,
            total_r=0.0,
            win_rate=0.0,
            max_dd_r=0.0,
            mar_proxy=float("nan"),
        )

    r_values = [t.theoretical_r for t in all_results]
    total_r = sum(r_values)
    avg_r = total_r / n_trades
    win_rate = sum(1 for r in r_values if r > 0) / n_trades
    max_dd = _max_drawdown_r(r_values)
    mar = total_r / max_dd if max_dd > 0 else float("nan")

    return SweepMetrics(
        params=dict(combo),
        n_trades=n_trades,
        avg_r=avg_r,
        total_r=total_r,
        win_rate=win_rate,
        max_dd_r=max_dd,
        mar_proxy=mar,
    )


def run_sweep(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    sweep: Optional[Dict[str, List]] = None,
    max_combos: Optional[int] = None,
    timeframe: str = "daily",
    sim_config: Optional[SimConfig] = None,
    base_params: Optional[B1Params] = None,
) -> pd.DataFrame:
    """Run the full parameter sensitivity sweep.

    Parameters
    ----------
    data_by_symbol : dict
        Symbol -> OHLCV DataFrame.
    symbols : list of str
        Symbols to include.
    sweep : dict or None
        Parameter grid specification (default: DEFAULT_SWEEP).
    max_combos : int or None
        Cap on total grid points.
    timeframe : str
    sim_config : SimConfig or None
    base_params : B1Params or None

    Returns
    -------
    pd.DataFrame with one row per combo.  Columns: parameter names +
    n_trades, avg_r, total_r, win_rate, max_dd_r, mar_proxy.
    """
    grid = build_grid(sweep, max_combos)
    logger.info("Sensitivity sweep: %d grid points", len(grid))

    rows: List[Dict] = []
    for i, combo in enumerate(grid):
        metrics = run_single_combo(
            combo, data_by_symbol, symbols, timeframe, sim_config, base_params,
        )
        row = dict(metrics.params)
        row.update({
            "n_trades": metrics.n_trades,
            "avg_r": metrics.avg_r,
            "total_r": metrics.total_r,
            "win_rate": metrics.win_rate,
            "max_dd_r": metrics.max_dd_r,
            "mar_proxy": metrics.mar_proxy,
        })
        rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info("  completed %d / %d combos", i + 1, len(grid))

    df = pd.DataFrame(rows)
    logger.info("Sweep complete: %d combos evaluated", len(df))
    return df


# ---------------------------------------------------------------------------
# Plateau / robustness analysis
# ---------------------------------------------------------------------------

def _are_neighbors(
    a: Dict[str, float],
    b: Dict[str, float],
    steps: Dict[str, float],
) -> bool:
    """True if a and b differ by exactly one step in exactly one parameter."""
    diffs = 0
    for param in steps:
        delta = abs(a[param] - b[param])
        if delta > 1e-9:
            if abs(delta - steps[param]) < 1e-9:
                diffs += 1
            else:
                return False  # more than one step apart in this param
    return diffs == 1


def _infer_steps(sweep: Dict[str, List]) -> Dict[str, float]:
    """Infer step size per parameter from the sweep grid values."""
    steps = {}
    for name, vals in sweep.items():
        sorted_vals = sorted(vals)
        if len(sorted_vals) >= 2:
            steps[name] = sorted_vals[1] - sorted_vals[0]
        else:
            steps[name] = 1.0
    return steps


def analyze_robustness(
    results_df: pd.DataFrame,
    sweep: Optional[Dict[str, List]] = None,
    fragility_percentile: float = 0.75,
    fragility_ratio: float = 0.5,
) -> List[RobustnessResult]:
    """Analyze plateau stability across the parameter grid.

    For each combo, computes a robustness score = mean neighbor total_r / own
    total_r.  Combos with high performance but low robustness are flagged as
    fragile.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_sweep()``.
    sweep : dict or None
        Parameter grid used (for step inference).
    fragility_percentile : float
        Combos above this percentile of total_r are candidates for fragility.
    fragility_ratio : float
        Robustness score below this triggers fragile flag.

    Returns
    -------
    List of RobustnessResult, one per grid point.
    """
    if sweep is None:
        sweep = DEFAULT_SWEEP
    steps = _infer_steps(sweep)
    param_names = list(sweep.keys())

    # Build lookup: combo tuple -> total_r.
    combos = []
    for _, row in results_df.iterrows():
        combo = {p: row[p] for p in param_names}
        combos.append(combo)

    total_r_values = results_df["total_r"].values
    threshold = np.nanpercentile(total_r_values, fragility_percentile * 100)

    results: List[RobustnessResult] = []
    for i, combo in enumerate(combos):
        own_total_r = total_r_values[i]

        # Find neighbors.
        neighbor_totals = []
        for j, other in enumerate(combos):
            if i == j:
                continue
            if _are_neighbors(combo, other, steps):
                neighbor_totals.append(total_r_values[j])

        n_neighbors = len(neighbor_totals)
        if n_neighbors == 0 or abs(own_total_r) < 1e-10:
            robustness = float("nan")
        else:
            robustness = np.mean(neighbor_totals) / own_total_r

        is_fragile = (
            own_total_r >= threshold
            and not np.isnan(robustness)
            and robustness < fragility_ratio
        )

        results.append(RobustnessResult(
            params=dict(combo),
            total_r=own_total_r,
            robustness_score=robustness,
            n_neighbors=n_neighbors,
            is_fragile=is_fragile,
        ))

    return results


def identify_plateau(
    robustness: List[RobustnessResult],
    min_robustness: float = 0.7,
) -> List[RobustnessResult]:
    """Return grid points in stable regions (robustness >= threshold)."""
    return [
        r for r in robustness
        if not np.isnan(r.robustness_score) and r.robustness_score >= min_robustness
    ]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_1d_sensitivity(
    results_df: pd.DataFrame,
    param_name: str,
    metric: str = "avg_r",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Plot 1D sensitivity curve for a single parameter.

    Averages the metric over all other parameter values.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_sweep()``.
    param_name : str
        Parameter to plot on x-axis.
    metric : str
        Column to plot on y-axis (default "avg_r").
    output_dir : Path or None
        Directory to save plot.  If None, plot is not saved.

    Returns
    -------
    Path to saved file, or None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = results_df.groupby(param_name)[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped[param_name], grouped[metric], "o-", linewidth=2, markersize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f"1D Sensitivity: {metric} vs {param_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"1d_{param_name}.png"
        fig.savefig(path, dpi=100)
        logger.info("Saved 1D plot: %s", path)

    plt.close(fig)
    return path


def plot_2d_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "avg_r",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Plot 2D heatmap for a pair of parameters.

    Averages the metric over all non-plotted parameters.

    Parameters
    ----------
    results_df : pd.DataFrame
    param_x, param_y : str
        Parameters for x and y axes.
    metric : str
        Column to display (default "avg_r").
    output_dir : Path or None

    Returns
    -------
    Path to saved file, or None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = results_df.groupby([param_x, param_y])[metric].mean().reset_index()
    pivot_table = pivot.pivot(index=param_y, columns=param_x, values=metric)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot_table.values,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
    )
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot_table.columns])
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot_table.index])
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"2D Sensitivity: {metric} ({param_x} vs {param_y})")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()

    path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"2d_{param_x}_vs_{param_y}.png"
        fig.savefig(path, dpi=100)
        logger.info("Saved 2D heatmap: %s", path)

    plt.close(fig)
    return path


def generate_all_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
    sweep: Optional[Dict[str, List]] = None,
    metric: str = "avg_r",
) -> List[Path]:
    """Generate all 1D + 2D plots for the sweep.

    Parameters
    ----------
    results_df : pd.DataFrame
    output_dir : Path
    sweep : dict or None
    metric : str

    Returns
    -------
    List of saved file paths.
    """
    if sweep is None:
        sweep = DEFAULT_SWEEP
    param_names = list(sweep.keys())
    paths: List[Path] = []

    # 1D curves.
    for name in param_names:
        p = plot_1d_sensitivity(results_df, name, metric, output_dir)
        if p:
            paths.append(p)

    # 2D heatmaps for all parameter pairs.
    for i, px in enumerate(param_names):
        for py in param_names[i + 1:]:
            p = plot_2d_heatmap(results_df, px, py, metric, output_dir)
            if p:
                paths.append(p)

    return paths
