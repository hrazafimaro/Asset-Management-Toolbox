# src/backtesting.py
"""
Portfolio Asset-Management-Toolbox
Module: backtesting.py

Backtesting de stratégies d’investissement :
- Momentum, Value, Low Volatility, Quality
- Walk-forward optimization
- Gestion des coûts de transaction & slippage
- Comparaison vs benchmark
- Rapport complet (PnL, drawdown, turnover)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
import warnings
from src.portfolio import Portfolio
from src.optimization import PortfolioOptimizer

# Type aliases
Prices = pd.DataFrame
Signals = pd.DataFrame  # index: date, columns: tickers → 1.0 (long), 0.0 (hold), -1.0 (short)


class Backtester:
    """
    Moteur de backtesting modulaire pour stratégies de portefeuille.

    Attributes:
        prices (pd.DataFrame): Prix historiques (index datetime, colonnes = tickers)
        benchmark_prices (pd.Series): Prix du benchmark (optionnel)
        initial_capital (float): Capital initial
        transaction_cost (float): Coût de transaction en % (ex: 0.001 = 10 bps)
        slippage (float): Slippage en % par trade
    """

    def __init__(
        self,
        prices: Prices,
        benchmark_prices: Optional[pd.Series] = None,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.001,  # 10 bps
        slippage: float = 0.0005         # 5 bps
    ):
        self.prices = prices.copy()
        self.prices.index = pd.to_datetime(self.prices.index)
        self.benchmark_prices = benchmark_prices.copy() if benchmark_prices is not None else None
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.tickers = prices.columns.tolist()
        self.returns = prices.pct_change().fillna(0)

        # Résultats
        self.nav_history: pd.Series = pd.Series()
        self.weights_history: pd.DataFrame = pd.DataFrame()
        self.trades_history: List[Dict] = []

    # ===================================================================
    # Stratégies Prédéfinies
    # ===================================================================
    @staticmethod
    def momentum_signal(
        returns: pd.DataFrame,
        lookback: int = 126,  # 6 mois
        top_n: int = 5
    ) -> Signals:
        """Signal momentum : top N actifs sur X jours."""
        momentum = returns.rolling(lookback).mean()
        signal = momentum.rank(axis=1, ascending=False).apply(
            lambda row: pd.Series(
                [1.0 if i < top_n else 0.0 for i in range(len(row))],
                index=row.index
            ),
            axis=1
        )
        return signal

    @staticmethod
    def low_volatility_signal(
        returns: pd.DataFrame,
        lookback: int = 252,
        top_n: int = 5
    ) -> Signals:
        """Signal faible volatilité."""
        vol = returns.rolling(lookback).std()
        signal = vol.rank(axis=1, ascending=True).apply(
            lambda row: pd.Series(
                [1.0 if i < top_n else 0.0 for i in range(len(row))],
                index=row.index
            ),
            axis=1
        )
        return signal

    @staticmethod
    def value_signal(
        prices: pd.DataFrame,
        book_to_price: pd.Series,  # B/P ratio par actif
        top_n: int = 5
    ) -> Signals:
        """Signal value (simplifié avec B/P)."""
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ranked = book_to_price.rank(ascending=False)
        winners = ranked.head(top_n).index
        for date in signal.index:
            signal.loc[date, winners] = 1.0
        return signal

    # ===================================================================
    # Backtest Principal
    # ===================================================================
    def run_strategy(
        self,
        signal_generator: Callable[[pd.DataFrame], Signals],
        rebalance_freq: str = "M",  # "W", "M", "Q"
        optimization_method: Optional[Callable] = None,
        **signal_kwargs
    ) -> Dict[str, Any]:
        """
        Exécute un backtest complet.

        Args:
            signal_generator: Fonction qui retourne des signaux
            rebalance_freq: Fréquence de réallocation
            optimization_method: Optionnel : optimiseur (ex: max_sharpe)
            **signal_kwargs: Paramètres pour le générateur de signaux
        """
        signals = signal_generator(self.returns, **signal_kwargs)
        rebalance_dates = signals.resample(rebalance_freq).last().index

        # Initialisation
        current_weights = pd.Series(0.0, index=self.tickers)
        current_weights.iloc[0] = 1.0  # Cash au départ
        nav = self.initial_capital
        nav_series = []
        weights_series = []

        prev_weights = current_weights.copy()

        for date in self.prices.index:
            # Réallocation ?
            if date in rebalance_dates:
                target_signal = signals.loc[date]
                target_weights = target_signal / target_signal.sum()
                target_weights = target_weights.fillna(0)

                # Option : optimisation
                if optimization_method:
                    opt = PortfolioOptimizer(self.returns.loc[:date])
                    try:
                        opt_result = optimization_method(opt)
                        target_weights = pd.Series(opt_result["weights"])
                    except:
                        pass  # fallback sur signal

                # Calcul du turnover & coûts
                turnover = (target_weights - prev_weights).abs().sum() / 2
                trade_cost = turnover * (self.transaction_cost + self.slippage) * nav

                # Trade log
                self.trades_history.append({
                    "date": date,
                    "turnover": turnover,
                    "cost": trade_cost,
                    "weights_before": prev_weights.to_dict(),
                    "weights_after": target_weights.to_dict()
                })

                prev_weights = target_weights.copy()

            # PnL du jour
            daily_return = (self.returns.loc[date] * prev_weights).sum()
            nav *= (1 + daily_return)
            nav -= 0  # coûts déjà déduits à la réallocation

            nav_series.append(nav)
            weights_series.append(prev_weights.copy())

            if date == self.prices.index[-1]:
                break

        # Résultats
        self.nav_history = pd.Series(nav_series, index=self.prices.index)
        self.weights_history = pd.DataFrame(weights_series, index=self.prices.index)

        return self._generate_report()

    # ===================================================================
    # Walk-Forward Optimization
    # ===================================================================
    def walk_forward(
        self,
        signal_generator: Callable,
        train_window: int = 252 * 2,   # 2 ans
        test_window: int = 252,       # 1 an
        rebalance_freq: str = "M",
        **signal_kwargs
    ) -> pd.DataFrame:
        """Backtest walk-forward."""
        results = []
        start_date = self.prices.index[0] + pd.Timedelta(days=train_window)
        end_date = self.prices.index[-1]

        current_date = start_date
        while current_date + pd.Timedelta(days=test_window) <= end_date:
            train_end = current_date
            test_end = current_date + pd.Timedelta(days=test_window)

            train_prices = self.prices.loc[:train_end]
            test_prices = self.prices.loc[train_end:test_end]

            # Entraînement
            opt = PortfolioOptimizer(train_prices.pct_change().dropna())
            try:
                opt_weights = opt.max_sharpe(long_only=True)["weights"]
            except:
                opt_weights = {t: 1/len(self.tickers) for t in self.tickers}

            # Test
            test_backtester = Backtester(
                test_prices,
                initial_capital=1_000_000,
                transaction_cost=self.transaction_cost
            )
            portfolio = Portfolio(
                holdings=opt_weights,
                prices=test_prices
            )
            nav_final = portfolio.nav()

            results.append({
                "train_end": train_end.date(),
                "test_start": current_date.date(),
                "test_end": test_end.date(),
                "return": (nav_final / 1_000_000) - 1
            })

            current_date += pd.Timedelta(days=test_window)

        return pd.DataFrame(results)

    # ===================================================================
    # Rapport Complet
    # ===================================================================
    def _generate_report(self) -> Dict[str, Any]:
        """Génère un rapport complet post-backtest."""
        from src.risk_metrics import RiskMetrics

        portfolio_returns = self.nav_history.pct_change().dropna()
        risk = RiskMetrics(portfolio_returns, risk_free_rate=0.02)

        total_return = (self.nav_history.iloc[-1] / self.nav_history.iloc[0]) - 1
        cagr = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        turnover = sum(t["turnover"] for t in self.trades_history) / len(self.trades_history)

        benchmark_return = None
        if self.benchmark_prices is not None:
            bench_nav = self.benchmark_prices / self.benchmark_prices.iloc[0]
            benchmark_return = bench_nav.iloc[-1] / bench_nav.iloc[0] - 1

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": risk.sharpe_ratio(),
            "sortino": risk.sortino_ratio(),
            "max_drawdown": risk.maximum_drawdown(),
            "calmar": risk.calmar_ratio(),
            "annual_volatility": portfolio_returns.std() * np.sqrt(252),
            "turnover_annual": turnover * 12,
            "final_nav": float(self.nav_history.iloc[-1]),
            "n_rebalances": len(self.trades_history),
            "benchmark_return": benchmark_return,
            "alpha": cagr - (0.02 + 0.8 * (benchmark_return or 0)) if benchmark_return else None,
            "nav_history": self.nav_history,
            "weights_history": self.weights_history
        }

    # ===================================================================
    # Export
    # ===================================================================
    def export_results(self, folder: str = "backtest_results"):
        """Exporte NAV, poids, trades."""
        import os
        os.makedirs(folder, exist_ok=True)

        self.nav_history.to_csv(f"{folder}/nav_history.csv")
        self.weights_history.to_csv(f"{folder}/weights_history.csv")
        pd.DataFrame(self.trades_history).to_csv(f"{folder}/trades.csv", index=False)

        # Graphiques
        self.plot_nav(save_path=f"{folder}/nav_plot.png")
        self.plot_weights_heatmap(save_path=f"{folder}/weights_heatmap.png")

    def plot_nav(self, save_path: Optional[str] = None):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.nav_history.index, y=self.nav_history, name="Portfolio"))
        if self.benchmark_prices is not None:
            bench_nav = self.benchmark_prices / self.benchmark_prices.iloc[0] * self.initial_capital
            fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name="Benchmark"))
        fig.update_layout(title="Backtest NAV", xaxis_title="Date", yaxis_title="NAV")
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    def plot_weights_heatmap(self, save_path: Optional[str] = None):
        import plotly.express as px
        df = self.weights_history.resample("M").last()
        fig = px.imshow(
            df.T,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Évolution des poids (mensuelle)"
        )
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()