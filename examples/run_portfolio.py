from src.portfolio import Portfolio
import pandas as pd

# Données fictives
prices = pd.DataFrame({
    "AAPL": [100, 105, 110, 107, 115],
    "GOOG": [50, 52, 55, 53, 58]
}, index=pd.date_range("2025-01-01", periods=5))

holdings = {"AAPL": 0.6, "GOOG": 0.4}

portfolio = Portfolio(holdings, prices, cash=1000, name="Tech Growth")
print(portfolio)
print(f"NAV: {portfolio.nav():.2f}")
print(f"CAGR: {portfolio.cagr():.2%}")