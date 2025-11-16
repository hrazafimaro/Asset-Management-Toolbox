import pandas as pd
from src.risk_metrics import RiskMetrics

# Données fictives
returns = pd.Series([
    0.01, -0.02, 0.03, 0.015, -0.01, 0.02, -0.03, 0.025
], index=pd.date_range("2025-01-01", periods=8))

risk = RiskMetrics(returns, risk_free_rate=0.02, confidence_level=0.95)
print(risk.summary())