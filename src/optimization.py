# src/optimization.py
"""
Portfolio Asset-Management-Toolbox
Module: optimization.py

Optimisation de portefeuille avancée :
- Mean-Variance (Markowitz)
- Black-Litterman
- Risk Parity (Equal Risk Contribution)
- Contraintes : long-only, secteurs, turnover, max weight
- Efficient Frontier
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
from scipy.optimize import minimize, LinearConstraint, Bounds
import cvxpy as cp
import warnings

# Type aliases
Returns = pd.DataFrame  # index: date, columns: tickers
Weights = np.ndarray     # array of weights


class PortfolioOptimizer:
    """
    Optimiseur de portefeuille multi-stratégies.

    Attributes:
        returns (pd.DataFrame): Rendements historiques
        cov_matrix (pd.DataFrame): Matrice de covariance
        expected_returns (pd.Series): Rendements attendus (optionnel)
    """

    def __init__(
        self,
        returns: Returns,
        expected_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ):
        self.returns = returns.copy()
        self.risk_free_rate = risk_free_rate
        self.tickers = returns.columns.tolist()

        # Calculs internes
        self.cov_matrix = returns.cov() * 252  # Annualisée
        self.expected_returns = (
            expected_returns.copy() * 252
            if expected_returns is not None
            else returns.mean() * 252
        )

        if len(self.tickers) < 2:
            raise ValueError("At least 2 assets required for optimization.")

    # ===================================================================
    # Mean-Variance Optimization (Markowitz)
    # ===================================================================
    def mean_variance(
        self,
        target_return: Optional[float] = None,
        max_risk: Optional[float] = None,
        long_only: bool = True,
        max_weight: float = 1.0,
        sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Optimisation Mean-Variance avec contraintes.

        Args:
            target_return: Rendement cible annualisé
            max_risk: Volatilité max annualisée
            long_only: True → pas de short
            max_weight: Poids max par actif
            sector_constraints: {"Tech": (0.1, 0.4), ...}

        Returns:
            dict avec weights, return, risk, sharpe
        """
        n = len(self.tickers)
        w = cp.Variable(n)
        ret = self.expected_returns.values @ w
        risk = cp.quad_form(w, self.cov_matrix.values)

        objective = cp.Minimize(risk)
        constraints = [cp.sum(w) == 1]

        if target_return is not None:
            constraints.append(ret >= target_return)
        if max_risk is not None:
            constraints.append(risk <= max_risk**2)
        if long_only:
            constraints.append(w >= 0)
        else:
            constraints.append(w >= -1)
        constraints.append(w <= max_weight)

        # Contraintes sectorielles
        if sector_constraints:
            for sector, (min_w, max_w) in sector_constraints.items():
                idx = [i for i, t in enumerate(self.tickers) if sector in t]
                if idx:
                    constraints.append(cp.sum(w[idx]) >= min_w)
                    constraints.append(cp.sum(w[idx]) <= max_w)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {prob.status}")

        weights = pd.Series(w.value, index=self.tickers)
        weights = weights.round(6)
        weights /= weights.sum()  # Renormalisation

        portfolio_return = float(ret.value)
        portfolio_risk = np.sqrt(float(risk.value))

        return {
            "weights": weights.to_dict(),
            "return": portfolio_return,
            "risk": portfolio_risk,
            "sharpe": (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else np.nan,
            "status": prob.status
        }

    # ===================================================================
    # Efficient Frontier
    # ===================================================================
    def efficient_frontier(
        self,
        n_points: int = 50,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> pd.DataFrame:
        """Calcule la frontière efficiente."""
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        target_returns = np.linspace(min_ret * 0.8, max_ret * 1.2, n_points)

        frontier = []
        for tr in target Returns:
            try:
                res = self.mean_variance(
                    target_return=tr,
                    long_only=long_only,
                    max_weight=max_weight
                )
                frontier.append({
                    "return": res["return"],
                    "risk": res["risk"],
                    "sharpe": res["sharpe"],
                    "weights": res["weights"]
                })
            except:
                continue
        return pd.DataFrame(frontier).dropna()

    # ===================================================================
    # Maximum Sharpe Portfolio
    # ===================================================================
    def max_sharpe(
        self,
        long_only: bool = True,
        max_weight: float = 0.3
    ) -> Dict[str, Any]:
        """Portefeuille à Sharpe maximal."""
        def neg_sharpe(w):
            w = np.array(w)
            ret = w @ self.expected_returns.values
            risk = np.sqrt(w @ self.cov_matrix.values @ w)
            return - (ret - self.risk_free_rate) / risk if risk > 0 else 1e6

        bounds = [(0, max_weight) if long_only else (-1, max_weight)] * len(self.tickers)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        init = np.ones(len(self.tickers)) / len(self.tickers)

        res = minimize(
            neg_sharpe, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not res.success:
            raise ValueError("Max Sharpe optimization failed.")

        weights = pd.Series(res.x, index=self.tickers)
        weights /= weights.sum()
        weights = weights.round(6)

        port_ret = float(weights @ self.expected_returns)
        port_risk = np.sqrt(weights @ self.cov_matrix @ weights)

        return {
            "weights": weights.to_dict(),
            "return": port_ret,
            "risk": port_risk,
            "sharpe": (port_ret - self.risk_free_rate) / port_risk
        }

    # ===================================================================
    # Risk Parity (Equal Risk Contribution)
    # ===================================================================
    def risk_parity(self) -> Dict[str, float]:
        """Risk Parity : chaque actif contribue également au risque total."""
        n = len(self.tickers)
        w = cp.Variable(n)
        risk_contrib = cp.multiply(w, self.cov_matrix.values @ w)
        objective = cp.Maximize(cp.geo_mean(risk_contrib + 1e-10))
        constraints = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if prob.status not in ["optimal"]:
            raise ValueError("Risk Parity failed.")

        weights = pd.Series(w.value, index=self.tickers)
        weights /= weights.sum()
        weights = weights.round(6)

        return {"weights": weights.to_dict(), "method": "risk_parity"}

    # ===================================================================
    # Black-Litterman Model
    # ===================================================================
    def black_litterman(
        self,
        views: Dict[str, float],           # {"AAPL": 0.15, "GOOG": 0.10}
        pick_matrix: List[List[int]],      # [[1, 0], [0, 1]]
        confidence: List[float],           # [0.8, 0.6]
        tau: float = 0.025
    ) -> pd.Series:
        """
        Black-Litterman : combine vues d'investisseur et équilibre de marché.

        Args:
            views: Rendements attendus par l'investisseur
            pick_matrix: Matrice P (1 si l'actif est dans la vue)
            confidence: Confiance dans chaque vue (0-1)
        """
        pi = self.expected_returns.values  # Équilibre de marché
        Sigma = self.cov_matrix.values
        P = np.array(pick_matrix)
        Q = np.array(list(views.values()))
        Omega = np.diag(1 / np.array(confidence))

        tau_Sigma = tau * Sigma
        inv = np.linalg.inv(P @ tau_Sigma @ P.T + Omega)
        bl_return = pi + tau_Sigma @ P.T @ inv @ (Q - P @ pi)

        return pd.Series(bl_return, index=self.tickers) * 252

    # ===================================================================
    # Minimum Variance (Global Minimum Variance)
    # ===================================================================
    def min_variance(self, long_only: bool = True) -> Dict[str, float]:
        """Portefeuille de variance minimale."""
        n = len(self.tickers)
        w = cp.Variable(n)
        risk = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(risk)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = pd.Series(w.value, index=self.tickers)
        weights /= weights.sum()
        weights = weights.round(6)

        port_risk = np.sqrt(float(risk.value))
        port_ret = float(weights @ self.expected_returns)

        return {
            "weights": weights.to_dict(),
            "return": port_ret,
            "risk": port_risk,
            "method": "min_variance"
        }

    # ===================================================================
    # Résumé
    # ===================================================================
    def summary(self) -> Dict[str, Any]:
        """Résumé des optimisations clés."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                "n_assets": len(self.tickers),
                "expected_returns": self.expected_returns.round(4).to_dict(),
                "volatility": np.sqrt(np.diag(self.cov_matrix)).round(4).tolist(),
                "correlation": self.returns.corr().round(3).to_dict()
            }