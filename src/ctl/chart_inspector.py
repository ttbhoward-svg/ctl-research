"""Chart Inspector — per-trade Plotly candlestick visualisation.

Generates interactive charts showing price bars, 10 EMA, entry/exit markers,
TP1-TP3 levels, and stop level for visual validation during chart study.

Two outputs:
  1. A JSON-serialisable data payload (for downstream use / storage).
  2. A standalone HTML file via Plotly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ctl.simulator import TradeResult

# Context bars before trigger and after exit for chart framing.
_PAD_BEFORE = 30
_PAD_AFTER = 15


def build_chart_payload(
    trade: TradeResult,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build a data dict for one trade's chart.

    Parameters
    ----------
    trade : TradeResult
    df : DataFrame
        OHLCV + EMA10 columns.

    Returns
    -------
    Dict with keys: bars, ema10, markers, levels, meta.
    """
    start = max(0, trade.trigger_bar_idx - _PAD_BEFORE)
    end = min(len(df), trade.exit_bar_idx + _PAD_AFTER + 1)
    window = df.iloc[start:end].copy()

    bars = {
        "date": window["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": window["Open"].tolist(),
        "high": window["High"].tolist(),
        "low": window["Low"].tolist(),
        "close": window["Close"].tolist(),
    }

    ema10 = (
        window["EMA10"].tolist() if "EMA10" in window.columns else []
    )

    markers = {
        "trigger": {
            "date": trade.trigger_date.strftime("%Y-%m-%d") if trade.trigger_date else None,
            "price": trade.stop_price,  # Low of trigger bar
            "label": "Trigger",
        },
        "entry": {
            "date": trade.entry_date.strftime("%Y-%m-%d") if trade.entry_date else None,
            "price": trade.entry_price,
            "label": "Entry",
        },
        "exit": {
            "date": trade.exit_date.strftime("%Y-%m-%d") if trade.exit_date else None,
            "price": trade.exit_price,
            "label": f"Exit ({trade.exit_reason})",
        },
    }

    levels = {
        "stop": trade.stop_price,
        "tp1": trade.tp1,
        "tp2": trade.tp2,
        "tp3": trade.tp3,
        "entry": trade.entry_price,
    }

    meta = {
        "symbol": trade.symbol,
        "timeframe": trade.timeframe,
        "r_actual": round(trade.r_multiple_actual, 3),
        "r_theoretical": round(trade.theoretical_r, 3),
        "mfe_r": round(trade.mfe_r, 3),
        "mae_r": round(trade.mae_r, 3),
        "outcome": trade.trade_outcome,
        "hold_bars": trade.hold_bars,
        "day1_fail": trade.day1_fail,
        "bars_of_air": trade.bars_of_air,
        "slope_20": round(trade.slope_20, 2),
    }

    return {
        "bars": bars,
        "ema10": ema10,
        "markers": markers,
        "levels": levels,
        "meta": meta,
    }


def render_chart_html(
    trade: TradeResult,
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> str:
    """Render an interactive Plotly candlestick chart for one trade.

    Parameters
    ----------
    trade : TradeResult
    df : DataFrame
        OHLCV + EMA10.
    out_path : Path, optional
        If provided, writes a standalone HTML file.

    Returns
    -------
    HTML string.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for Chart Inspector.  pip install plotly")

    payload = build_chart_payload(trade, df)
    bars = payload["bars"]
    ema10 = payload["ema10"]
    markers = payload["markers"]
    levels = payload["levels"]
    meta = payload["meta"]

    fig = go.Figure()

    # Candlesticks.
    fig.add_trace(go.Candlestick(
        x=bars["date"],
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        close=bars["close"],
        name="Price",
    ))

    # EMA10 overlay.
    if ema10:
        fig.add_trace(go.Scatter(
            x=bars["date"],
            y=ema10,
            mode="lines",
            name="EMA10",
            line=dict(color="blue", width=1.5),
        ))

    # Horizontal lines for levels.
    _add_hline(fig, levels["stop"], "Stop", "red", bars["date"])
    _add_hline(fig, levels["entry"], "Entry", "gray", bars["date"])
    _add_hline(fig, levels["tp1"], "TP1 (0.618)", "green", bars["date"])
    _add_hline(fig, levels["tp2"], "TP2 (0.786)", "limegreen", bars["date"])
    _add_hline(fig, levels["tp3"], "TP3 (1.000)", "darkgreen", bars["date"])

    # Entry marker.
    if markers["entry"]["date"]:
        fig.add_trace(go.Scatter(
            x=[markers["entry"]["date"]],
            y=[markers["entry"]["price"]],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="blue"),
            text=["Entry"],
            textposition="bottom center",
            name="Entry",
            showlegend=False,
        ))

    # Exit marker.
    if markers["exit"]["date"]:
        color = "green" if meta["outcome"] == "Win" else "red"
        fig.add_trace(go.Scatter(
            x=[markers["exit"]["date"]],
            y=[markers["exit"]["price"]],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color=color),
            text=[markers["exit"]["label"]],
            textposition="top center",
            name="Exit",
            showlegend=False,
        ))

    # Title.
    title = (
        f"{meta['symbol']} {meta['timeframe']} B1 — "
        f"{meta['outcome']} | R={meta['r_actual']:.2f} "
        f"(Theo={meta['r_theoretical']:.2f}) | "
        f"Hold={meta['hold_bars']}bars"
    )
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
    )

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)

    return html


def _add_hline(
    fig: Any,
    price: float,
    label: str,
    color: str,
    x_range: list,
) -> None:
    """Add a dashed horizontal level line spanning the chart."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(
        x=[x_range[0], x_range[-1]],
        y=[price, price],
        mode="lines",
        line=dict(color=color, width=1, dash="dash"),
        name=label,
        showlegend=True,
    ))
