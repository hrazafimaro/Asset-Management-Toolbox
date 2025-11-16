# src/visualization.py
"""
Portfolio Asset-Management-Toolbox
Module: visualization.py

Visualisation avancée avec Plotly :
- Efficient Frontier
- Allocation Pie / Treemap
- Drawdown Waterfall
- Rolling Sharpe / Volatility
- Performance Attribution
- Correlation Heatmap
- NAV vs Benchmark
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Any, Tuple
from src.portfolio import Portfolio
from src.risk_metrics import RiskMetrics
from src.optimization import PortfolioOptimizer

# ===================================================================
# Classe principale de visualisation
# ===================================================================
class PortfolioVisualizer:
    """
    Outil de visualisation institutionnelle pour portefeuilles.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        benchmark_nav: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ):
        self.portfolio = portfolio
        self.benchmark_nav = benchmark_nav
        self.risk_free_rate = risk_free_rate
        self.returns = portfolio.nav_history.pct_change().dropna()
        self.risk = RiskMetrics(self.returns, risk_free_rate=risk_free_rate)

    # ===================================================================
    # 1. NAV vs Benchmark
    # ===================================================================
    def plot_nav_vs_benchmark(
        self,
        title: str = "Évolution de la NAV vs Benchmark",
        height: int = 500
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.portfolio.nav_history.index,
            y=self.portfolio.nav_history,
            name="Portefeuille",
            line=dict(width=3)
        ))
        if self.benchmark_nav is not None:
            bench_scaled = self.benchmark_nav / self.benchmark_nav.iloc[0] * self.portfolio.nav_history.iloc[0]
            fig.add_trace(go.Scatter(
                x=bench_scaled.index,
                y=bench_scaled,
                name="Benchmark",
                line=dict(dash="dash", width=2)
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="NAV (€)",
            template="plotly_white",
            height=height,
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        return fig

    # ===================================================================
    # 2. Efficient Frontier
    # ===================================================================
    def plot_efficient_frontier(
        self,
        optimizer: PortfolioOptimizer,
        n_points: int = 50,
        highlight_portfolio: bool = True,
        height: int = 600
    ) -> go.Figure:
        frontier = optimizer.efficient_frontier(n_points=n_points)
        fig = go.Figure()

        # Frontière
        fig.add_trace(go.Scatter(
            x=frontier["risk"],
            y=frontier["return"],
            mode="lines+markers",
            name="Frontière Efficiente",
            line=dict(color="royalblue"),
            hovertemplate="Rendement: %{y:.1%}<br>Risque: %{x:.1%}"
        ))

        # Portefeuille actuel
        if highlight_portfolio:
            port_ret = self.portfolio.cagr()
            port_risk = self.portfolio.volatility()
            fig.add_trace(go.Scatter(
                x=[port_risk],
                y=[port_ret],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                name="Votre Portefeuille"
            ))

        # Max Sharpe
        max_sharpe = optimizer.max_sharpe()
        fig.add_trace(go.Scatter(
            x=[max_sharpe["risk"]],
            y=[max_sharpe["return"]],
            mode="markers",
            marker=dict(color="green", size=10, symbol="diamond"),
            name="Max Sharpe"
        ))

        fig.update_layout(
            title="Frontière Efficiente (Markowitz)",
            xaxis_title="Risque (Volatilité annualisée)",
            yaxis_title="Rendement attendu (CAGR)",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=height,
            showlegend=True
        )
        return fig

    # ===================================================================
    # 3. Allocation (Pie + Treemap)
    # ===================================================================
    def plot_allocation_pie(
        self,
        date: Optional[pd.Timestamp] = None,
        title: str = "Allocation du Portefeuille"
    ) -> go.Figure:
        if date is None:
            date = self.portfolio.nav_history.index[-1]
        prices = self.portfolio.prices.loc[date]
        values = {t: prices[t] * w for t, w in self.portfolio.holdings.items()}
        df = pd.DataFrame(list(values.items()), columns=["Actif", "Valeur"])
        df["Pourcentage"] = df["Valeur"] / df["Valeur"].sum()

        fig = px.pie(
            df,
            values="Pourcentage",
            names="Actif",
            title=title,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    def plot_allocation_treemap(self) -> go.Figure:
        values = [w for w in self.portfolio.holdings.values()]
        labels = list(self.portfolio.holdings.keys())
        parents = [""] * len(labels)

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+value+percent parent",
            marker_colors=px.colors.sequential.Greens
        ))
        fig.update_layout(title="Allocation en Treemap")
        return fig

    # ===================================================================
    # 4. Drawdown Waterfall
    # ===================================================================
    def plot_drawdown_waterfall(self, height: int = 400) -> go.Figure:
        dd = self.risk.drawdown_series()
        fig = go.Figure(go.Waterfall(
            x=dd.index,
            y=dd,
            measure=["relative"] * len(dd),
            text=[f"{x:.1%}" for x in dd],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(
            title="Drawdown Cumulé (Waterfall)",
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=height
        )
        return fig

    # ===================================================================
    # 5. Rolling Metrics
    # ===================================================================
    def plot_rolling_sharpe(
        self,
        window: int = 252,
        height: int = 400
    ) -> go.Figure:
        rolling_excess = self.returns - self.risk_free_rate / 252
        rolling_sharpe = rolling_excess.rolling(window).mean() / rolling_excess.rolling(window).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            name=f"Sharpe Glissant ({window} jours)",
            line=dict(color="purple")
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Sharpe > 1")
        fig.update_layout(
            title="Sharpe Ratio Glissant",
            yaxis_title="Sharpe",
            template="plotly_white",
            height=height
        )
        return fig

    # ===================================================================
    # 6. Correlation Heatmap
    # ===================================================================
    def plot_correlation_heatmap(self) -> go.Figure:
        corr = self.portfolio.returns.corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            annotation_text=np.round(corr.values, 2),
            colorscale="RdYlBu",
            showscale=True
        )
        fig.update_layout(
            title="Matrice de Corrélation",
            xaxis_title="Actifs",
            yaxis_title="Actifs",
            height=500
        )
        return fig

    # ===================================================================
    # 7. Performance Attribution (par actif)
    # ===================================================================
    def plot_attribution(self) -> go.Figure:
        weights = pd.Series(self.portfolio.holdings)
        asset_returns = self.portfolio.returns.mean() * 252
        contribution = weights * asset_returns
        df = pd.DataFrame({
            "Actif": contribution.index,
            "Contribution": contribution.values,
            "Poids": weights.values
        }).sort_values("Contribution", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Actif"],
            y=df["Contribution"],
            name="Contribution au rendement",
            marker_color="lightblue"
        ))
        fig.add_trace(go.Bar(
            x=df["Actif"],
            y=df["Poids"],
            name="Poids",
            marker_color="lightcoral",
            yaxis="y2"
        ))
        fig.update_layout(
            title="Attribution de Performance par Actif",
            barmode="relative",
            yaxis=dict(title="Contribution (%)"),
            yaxis2=dict(title="Poids", overlaying="y", side="right", tickformat=".0%"),
            template="plotly_white"
        )
        return fig

    # ===================================================================
    # 8. Dashboard Complet (Subplots)
    # ===================================================================
    def dashboard(self) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "NAV vs Benchmark", "Allocation", "Drawdown",
                "Sharpe Glissant", "Corrélation", "Attribution"
            ),
            specs=[
                [{"secondary_y": False}, {"type": "pie"}],
                [{"secondary_y": False}, {"type": "heatmap"}],
                [{"secondary_y": False}, {"type": "bar"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. NAV
        nav_fig = self.plot_nav_vs_benchmark()
        for trace in nav_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # 2. Pie
        pie_fig = self.plot_allocation_pie()
        for trace in pie_fig.data:
            fig.add_trace(trace, row=1, col=2)

        # 3. Drawdown
        dd_fig = self.plot_drawdown_waterfall()
        for trace in dd_fig.data:
            fig.add_trace(trace, row=2, col=1)

        # 4. Sharpe
        sharpe_fig = self.plot_rolling_sharpe()
        for trace in sharpe_fig.data:
            fig.add_trace(trace, row=2, col=2)

        # 5. Correlation
        corr_fig = self.plot_correlation_heatmap()
        for trace in corr_fig.data:
            fig.add_trace(trace, row=3, col=1)

        # 6. Attribution
        attr_fig = self.plot_attribution()
        for trace in attr_fig.data:
            fig.add_trace(trace, row=3, col=2)

        fig.update_layout(
            title_text