import pandas as pd
from src.optimization import PortfolioOptimizer

# Données
returns = pd.DataFrame({
    "AAPL": [0.001, 0.002, -0.001, 0.003],
    "GOOG": [0.0005, 0.0015, 0.002, -0.001],
    "MSFT": [0.0012, -0.0008, 0.0025, 0.001]
}, index=pd.date_range("2025-01-01", periods=4))

opt = PortfolioOptimizer(returns, risk_free_rate=0.02)

# Max Sharpe
result = opt.max_sharpe(long_only=True)
print("Max Sharpe weights:", result["weights"])

# Frontière
frontier = opt.efficient_frontier(n_points=10)
print(frontier[["return", "risk", "sharpe"]])