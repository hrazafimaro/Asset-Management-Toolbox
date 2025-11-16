from src.portfolio import Portfolio
from src.optimization import PortfolioOptimizer
from src.visualization import PortfolioVisualizer
import pandas as pd

# Données
prices = pd.DataFrame({
    "AAPL": [100, 105, 110, 115, 120],
    "GOOG": [50, 52, 55, 53, 58],
    "MSFT": [200, 205, 195, 210, 220]
}, index=pd.date_range("2025-01-01", periods=5))

holdings = {"AAPL": 0.4, "GOOG": 0.3, "MSFT": 0.3}
portfolio = Portfolio(holdings, prices, name="Tech Portfolio")

# Optimisation
opt = PortfolioOptimizer(portfolio.returns)
visualizer = PortfolioVisualizer(portfolio)

# Graphiques
visualizer.plot_nav_vs_benchmark().show()
visualizer.plot_efficient_frontier(opt).show()
visualizer.dashboard().show()

# Export
visualizer.plot_nav_vs_benchmark().write_image("reports/nav.png")