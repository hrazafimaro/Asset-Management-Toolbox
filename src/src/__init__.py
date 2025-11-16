# src/__init__.py
"""
Portfolio Asset-Management-Toolbox

Un toolbox open-source Python dédié aux Asset Managers :
- Gestion de portefeuille (Portfolio)
- Métriques de risque (RiskMetrics)
- Optimisation (PortfolioOptimizer)
- Backtesting (Backtester)
- Visualisation interactive (PortfolioVisualizer)

Importez simplement :

>>> from src import Portfolio, RiskMetrics, PortfolioOptimizer, Backtester, PortfolioVisualizer
"""

# ----------------------------------------------------------------------
# Imports explicites – tout est disponible au niveau du package
# ----------------------------------------------------------------------
from .portfolio import Portfolio
from .risk_metrics import RiskMetrics
from .optimization import PortfolioOptimizer
from .backtesting import Backtester
from .visualization import PortfolioVisualizer

# ----------------------------------------------------------------------
# Version du package (PEP 396)
# ----------------------------------------------------------------------
__version__ = "1.0.0"

# ----------------------------------------------------------------------
# Métadonnées utiles
# ----------------------------------------------------------------------
__author__ = "Aubin Razafimaro (@AubinRazafimaro)"
__email__ = "contact@aubinrazafimaro.com"
__description__ = "Outil complet d'Asset Management : portefeuille, risque, optimisation, backtesting & visualisation."
__url__ = "https://github.com/AubinRazafimaro/Portfolio-Asset-Management-Toolbox"

# ----------------------------------------------------------------------
# Export public (pour `from src import *`)
# ----------------------------------------------------------------------
__all__ = [
    "Portfolio",
    "RiskMetrics",
    "PortfolioOptimizer",
    "Backtester",
    "PortfolioVisualizer",
]