# src/risk_metrics.py
"""
Portfolio Asset-Management-Toolbox
Module: risk_metrics.py

Calcul des métriques de risque avancées :
- Sharpe, Sortino, Calmar
- Value at Risk (VaR) : historique, paramétrique, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Maximum Drawdown, Ulcer Index
- Beta, Tracking Error, Information Ratio
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Dict
from scipy.stats import norm
import warnings

# Type aliases
Returns = pd.Series  # Rendements quotidiens


class RiskMetrics:
    """
    Classe utilitaire pour calculer les métriques de risque sur une série de rendements.

    Parameters:
        returns (pd.Series): Rendements quotidiens (index datetime)
        risk_free_rate (float): Taux sans risque annualisé (défaut: 0.0)
        confidence_level (float): Niveau de confiance pour VaR/CVaR (défaut: 0.95)
    """

    def __init__(
        self,
        returns: Returns,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95
    ):
        self.returns = returns.copy()
        self.risk_free_rate = risk_free_rate / 252  # Conversion annualisé → journalier
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        if self.returns.empty:
            raise ValueError("Returns series cannot be empty.")
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)

    # ===================================================================
    # Métriques de performance ajustée au risque
    # ===================================================================
    def sharpe_ratio(self, annualize: bool = True) -> float:
        """Sharpe Ratio (excès de rendement / volatilité)."""
        excess_returns = self.returns - self.risk_free_rate
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        if std_excess == 0:
            return np.nan
        ratio = mean_excess / std_excess
        return ratio * np.sqrt(252) if annualize else ratio

    def sortino_ratio(self, annualize: bool = True) -> float:
        """Sortino : excès / downside deviation."""
        excess_returns = self.returns - self.risk_free_rate
        mean_excess = excess_returns.mean()
        downside = excess_returns[excess_returns < 0]
        downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
        if downside_dev == 0:
            return np.nan
        ratio = mean_excess / downside_dev
        return ratio * np.sqrt(252) if annualize else ratio

    def calmar_ratio(self) -> float:
        """Calmar : CAGR / Max Drawdown."""
        cagr = self._cagr()
        max_dd = self.maximum_drawdown()
        return cagr / abs(max_dd) if max_dd != 0 else np.nan

    # ===================================================================
    # Drawdown
    # ===================================================================
    def maximum_drawdown(self) -> float:
        """Maximum Drawdown (en %)."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def drawdown_series(self) -> pd.Series:
        """Série complète des drawdowns."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        return (cumulative - rolling_max) / rolling_max

    def ulcer_index(self) -> float:
        """Ulcer Index : racine de la moyenne des drawdowns²."""
        dd = self.drawdown_series()
        return np.sqrt((dd ** 2).mean())

    # ===================================================================
    # Value at Risk (VaR)
    # ===================================================================
    def var_historical(self) -> float:
        """VaR historique (percentile)."""
        return self.returns.quantile(self.alpha)

    def var_parametric(self) -> float:
        """VaR gaussien (moyenne + z-score * sigma)."""
        mean = self.returns.mean()
        std = self.returns.std()
        z = norm.ppf(self.alpha)
        return mean + z * std

    def var_monte_carlo(
        self,
        n_simulations: int = 100_000,
        horizon: int = 1
    ) -> float:
        """VaR par simulation Monte Carlo (log-normal)."""
        mu = self.returns.mean()
        sigma = self.returns.std()
        sim_returns = np.random.normal(mu, sigma, (n_simulations, horizon))
        sim_portfolio = np.exp(np.cumsum(sim_returns, axis=1))[:, -1]
        return np.percentile(sim_portfolio, self.alpha * 100) - 1

    # ===================================================================
    # Conditional VaR (CVaR)
    # ===================================================================
    def cvar_historical(self) -> float:
        """CVaR historique : moyenne des pertes sous VaR."""
        var = self.var_historical()
        tail_losses = self.returns[self.returns <= var]
        return tail_losses.mean() if len(tail_losses) > 0 else np.nan

    def cvar_parametric(self) -> float:
        """CVaR gaussien."""
        z = norm.ppf(self.alpha)
        mean = self.returns.mean()
        std = self.returns.std()
        var = mean + z * std
        cvar = mean - (std * norm.pdf(z) / self.alpha)
        return cvar

    # ===================================================================
    # Benchmarking
    # ===================================================================
    def tracking_error(self, benchmark_returns: Returns, annualize: bool = True) -> float:
        """Tracking Error vs benchmark."""
        aligned = self.returns.align(benchmark_returns, join="inner")
        diff = aligned[0] - aligned[1]
        te = diff.std()
        return te * np.sqrt(252) if annualize else te

    def information_ratio(self, benchmark_returns: Returns, annualize: bool = True) -> float:
        """IR = excès de rendement / tracking error."""
        excess = self.returns - benchmark_returns
        active_return = excess.mean()
        te = self.tracking_error(benchmark_returns, annualize=False)
        if te == 0:
            return np.nan
        ir = active_return / te
        return ir * np.sqrt(252) if annualize else ir

    def beta(self, benchmark_returns: Returns) -> float:
        """Bêta du portefeuille vs benchmark."""
        cov = np.cov(self.returns, benchmark_returns, ddof=0)[0, 1]
        var_bench = np.var(benchmark_returns, ddof=0)
        return cov / var_bench if var_bench != 0 else np.nan

    # ===================================================================
    # Utilitaires internes
    # ===================================================================
    def _cagr(self) -> float:
        """CAGR interne (pour Calmar)."""
        cumulative = (1 + self.returns).cumprod()
        n_periods = len(self.returns)
        if n_periods == 0:
            return 0.0
        n_years = n_periods / 252
        return cumulative.iloc[-1] ** (1 / n_years) - 1

    # ===================================================================
    # Résumé complet
    # ===================================================================
    def summary(self) -> Dict[str, float]:
        """Résumé complet des métriques de risque."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "sharpe_ratio": self.sharpe_ratio(),
                "sortino_ratio": self.sortino_ratio(),
                "calmar_ratio": self.calmar_ratio(),
                "max_drawdown": self.maximum_drawdown(),
                "ulcer_index": self.ulcer_index(),
                "var_historical_95%": self.var_historical(),
                "cvar_historical_95%": self.cvar_historical(),
                "var_parametric_95%": self.var_parametric(),
                "volatility_annual": self.returns.std() * np.sqrt(252),
            }

    # ===================================================================
    # Factory
    # ===================================================================
    @classmethod
    def from_portfolio(
        cls,
        portfolio_nav: pd.Series,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95
    ) -> "RiskMetrics":
        """Crée RiskMetrics à partir de la NAV d’un portefeuille."""
        returns = portfolio_nav.pct_change().dropna()
        return cls(returns, risk_free_rate, confidence_level)risk_metrics.py

