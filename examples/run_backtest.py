import pandas as pd
from src.backtesting import Backtester

# Données fictives
prices = pd.DataFrame({
    "AAPL": [100, 105, 110, 107, 115, 120],
    "GOOG": [50, 48, 52, 55, 53, 58],
    "MSFT": [200, 205, 195, 210, 215, 220]
}, index=pd.date_range("2025-01-01", periods=6))

backtester = Backtester(prices, initial_capital=1_000_000)
report = backtester.run_strategy(
    signal_generator=Backtester.momentum_signal,
    lookback=3,
    top_n=2,
    rebalance_freq="M"
)

print(f"CAGR: {report['cagr']:.2%}")
print(f"Sharpe: {report['sharpe']:.2f}")
backtester.export_results("results_momentum")