"""HTML performance report using plotly."""

from __future__ import annotations

import webbrowser
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from futuresquant.backtest.engine import BacktestResult


def build_report(result: "BacktestResult", open_browser: bool = True) -> Path:
    """Generate an interactive HTML report and optionally open it."""
    from futuresquant.backtest.analyzer import PerformanceAnalyzer

    analyzer = PerformanceAnalyzer(result.account.equity_curve(), result.config.initial_capital)
    equity = analyzer.equity
    drawdown = analyzer.drawdown_series()
    metrics = result.metrics

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        subplot_titles=["Equity Curve", "Drawdown", "Close Price"],
        vertical_spacing=0.07,
    )

    # Equity
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Equity", line=dict(color="#1f77b4", width=1.5),
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        name="Drawdown %", fill="tozeroy",
        line=dict(color="#d62728", width=1),
        fillcolor="rgba(214,39,40,0.2)",
    ), row=2, col=1)

    # Price
    close = result.klines["close"]
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        name="Close", line=dict(color="#2ca02c", width=1),
    ), row=3, col=1)

    # Metrics annotation
    metric_text = "<br>".join(
        f"<b>{k}</b>: {v:.4f}" if isinstance(v, float) else f"<b>{k}</b>: {v}"
        for k, v in metrics.items()
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=1.01, y=0.95,
        text=metric_text, showarrow=False, align="left",
        bordercolor="#888", borderwidth=1, bgcolor="white",
        font=dict(size=11),
    )

    fig.update_layout(
        title=f"Backtest Report — {result.config.symbol}",
        height=900,
        margin=dict(r=220),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Yuan", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)

    out_path = Path(tempfile.mktemp(suffix=".html", prefix="backtest_"))
    fig.write_html(str(out_path))

    if open_browser:
        webbrowser.open(out_path.as_uri())

    return out_path
