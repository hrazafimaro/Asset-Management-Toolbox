# src/portfolio.py
"""
Portfolio Asset-Management-Toolbox
Module: portfolio.py

Classe principale pour la gestion de portefeuille :
- Calcul de NAV
- Réallocation
- Métriques de performance de base
- Export CSV / JSON
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import json
from datetime import datetime

# Type alias
Holdings = Dict[str, float]  # {ticker: weight}
Prices = pd.DataFrame        # index: date, columns: tickers


class Portfolio:
    """
    Représente un portefeuille d'actifs financiers.

    Attributes:
        holdings (Dict[str, float]): Poids cibles {ticker: weight}
        prices (pd.DataFrame): Prix historiques (index datetime, colonnes = tickers)
        cash (float): Montant en cash (optionnel)
        name (str): Nom du portefeuille
        benchmark_prices (pd.DataFrame): Prix du benchmark (optionnel)
    """

    def __init__(
        self,
        holdings: Holdings,
        prices: Prices,
        cash: float = 0.0,
        name: str = "My Portfolio",
        benchmark_prices: Optional[Prices] = None
    ):
        self.name = name
        self.cash = cash
        self._validate_inputs(holdings, prices)
        self.holdings = {k: float(v) for k, v in holdings.items()}
        self.prices = prices.copy()
        self.prices.index = pd.to_datetime(self.prices.index)
        self.benchmark_prices = benchmark_prices.copy() if benchmark_prices is not None else None

        # Calculs internes
        self.returns = self.prices.pct_change().dropna()
        self.nav_history = self._compute_nav_history()

    # ===================================================================
    # Validation
    # ===================================================================
    def _validate_inputs(self, holdings: Holdings, prices: Prices) -> None:
        if not holdings:
            raise ValueError("Holdings cannot be empty.")
        if not all(isinstance(w, (int, float)) for w in holdings.values()):
            raise ValueError("All weights must be numeric.")
        total_weight = sum(abs(w) for w in holdings.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Total weight must sum to 1.0 (current: {total_weight:.6f})")
        if prices.empty:
            raise ValueError("Prices DataFrame cannot be empty.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Prices index must be DatetimeIndex.")

    # ===================================================================
    # NAV & Valeur du portefeuille
    # ===================================================================
    def _compute_nav_history(self) -> pd.Series:
        """Calcule la NAV historique à partir des prix et poids initiaux."""
        weights = pd.Series(self.holdings)
        aligned_prices = self.prices[weights.index]
        portfolio_values = (aligned_prices * weights).sum(axis=1)
        portfolio_values += self.cash
        portfolio_values = portfolio_values.ffill().fillna(self.cash if self.cash > 0 else 100.0)
        return portfolio_values

    def nav(self, date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """NAV à une date donnée (ou dernière date)."""
        if date is None:
            return float(self.nav_history.iloc[-1])
        date = pd.to_datetime(date)
        if date not in self.nav_history.index:
            raise KeyError(f"Date {date.date()} not in NAV history.")
        return float(self.nav_history.loc[date])

    def total_value(self, date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """Alias pour nav()."""
        return self.nav(date)

    # ===================================================================
    # Réallocation
    # ===================================================================
    def rebalance(
        self,
        target_weights: Holdings,
        execution_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> "Portfolio":
        """
        Crée un nouveau portefeuille réalloué à une date donnée.
        """
        execution_date = pd.to_datetime(execution_date) if execution_date else self.prices.index[-1]
        if execution_date not in self.prices.index:
            raise KeyError(f"Execution date {execution_date.date()} not in price data.")

        # Prix à la date de réallocation
        prices_at_date = self.prices.loc[execution_date]
        current_nav = self.nav(execution_date)
        target_nav = current_nav  # On conserve la valeur totale

        # Calcul des nouvelles quantités (approximation en poids)
        new_holdings = {
            ticker: target_weight
            for ticker, target_weight in target_weights.items()
            if ticker in prices_at_date.index
        }

        # Normalisation des poids
        total = sum(new_holdings.values())
        new_holdings = {k: v / total for k, v in new_holdings.items()}

        # Nouveau portefeuille post-réallocation
        future_prices = self.prices.loc[execution_date:]
        new_portfolio = Portfolio(
            holdings=new_holdings,
            prices=future_prices,
            cash=self.cash,
            name=f"{self.name} (rebalanced {execution_date.date()})",
            benchmark_prices=self.benchmark_prices.loc[execution_date:] if self.benchmark_prices is not None else None
        )
        return new_portfolio

    # ===================================================================
    # Métriques de base
    # ===================================================================
    def total_return(self) -> float:
        """Retour total du portefeuille."""
        nav_start = self.nav_history.iloc[0]
        nav_end = self.nav_history.iloc[-1]
        return (nav_end / nav_start) - 1

    def cagr(self) -> float:
        """Taux de croissance annuel composé."""
        nav_start = self.nav_history.iloc[0]
        nav_end = self.nav_history.iloc[-1]
        n_years = (self.nav_history.index[-1] - self.nav_history.index[0]).days / 365.25
        if n_years <= 0:
            return 0.0
        return (nav_end / nav_start) ** (1 / n_years) - 1

    def volatility(self, annualize: bool = True) -> float:
        """Volatilité annualisée des rendements du portefeuille."""
        portfolio_returns = self.nav_history.pct_change().dropna()
        vol = portfolio_returns.std()
        return vol * np.sqrt(252) if annualize else vol

    # ===================================================================
    # Benchmarking
    # ===================================================================
    def tracking_error(self, annualize: bool = True) -> float:
        if self.benchmark_prices is None:
            raise ValueError("Benchmark prices not provided.")
        bench_returns = self.benchmark_prices.pct_change().dropna()
        port_returns = self.nav_history.pct_change().dropna()
        aligned = port_returns.align(bench_returns, join="inner")[0] - port_returns.align(bench_returns, join="inner")[1]
        te = aligned.std()
        return te * np.sqrt(252) if annualize else te

    def beta(self) -> float:
        if self.benchmark_prices is None:
            raise ValueError("Benchmark prices not provided.")
        port_returns = self.nav_history.pct_change().dropna()
        bench_returns = self.benchmark_prices.pct_change().dropna()
        aligned_port, aligned_bench = port_returns.align(bench_returns, join="inner")
        cov = np.cov(aligned_port, aligned_bench)[0, 1]
        var_bench = np.var(aligned_bench)
        return cov / var_bench if var_bench != 0 else 0.0

    # ===================================================================
    # Export
    # ===================================================================
    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Exporte NAV + allocations."""
        df = pd.DataFrame({
            "NAV": self.nav_history,
            **{ticker: self.prices[ticker] * weight for ticker, weight in self.holdings.items()}
        })
        df.to_csv(filepath)

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Exporte métadonnées du portefeuille."""
        data = {
            "name": self.name,
            "cash": self.cash,
            "holdings": self.holdings,
            "nav_start": float(self.nav_history.iloc[0]),
            "nav_end": float(self.nav_history.iloc[-1]),
            "total_return": self.total_return(),
            "cagr": self.cagr(),
            "volatility_annual": self.volatility(),
            "export_date": datetime.now().isoformat()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    # ===================================================================
    # Factory Methods
    # ===================================================================
    @classmethod
    def from_csv(
        cls,
        holdings_csv: Union[str, Path],
        prices_csv: Union[str, Path],
        cash: float = 0.0,
        name: str = "Portfolio from CSV",
        benchmark_csv: Optional[Union[str, Path]] = None
    ) -> "Portfolio":
        """Crée un portefeuille à partir de fichiers CSV."""
        holdings_df = pd.read_csv(holdings_csv, index_col=0, header=None, squeeze=True)
        holdings = holdings_df.to_dict()
        prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
        benchmark = pd.read_csv(benchmark_csv, index_col=0, parse_dates=True) if benchmark_csv else None
        return cls(holdings, prices, cash, name, benchmark)

    # ===================================================================
    # Représentation
    # ===================================================================
    def __repr__(self) -> str:
        return f"<Portfolio '{self.name}' | NAV: {self.nav():,.2f} | Tickers: {len(self.holdings)}>"

    def summary(self) -> Dict[str, float]:
        """Résumé rapide."""
        return {
            "name": self.name,
            "nav": self.nav(),
            "total_return": self.total_return(),
            "cagr": self.cagr(),
            "volatility": self.volatility(),
            "n_assets": len(self.holdings)
        }
    from src.portfolio import Portfolio
from src.risk_metrics import RiskMetrics

portfolio = Portfolio.from_csv("data/holdings.csv", "data/prices.csv")
risk = RiskMetrics.from_portfolio(portfolio.nav_history)
print(f"VaR 95% : {risk.var_historical():.2%}")
