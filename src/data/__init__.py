"""Data module for multi-signal CEEMDAN-ARMA-LSTM trading system."""

# Re-export main classes
__all__ = [
    "DataIngestionModule",
    "YFinanceFetcher",
    "CEEMDAN",
    "ADFClassifier",
    "VARModel",
    "MultivariateLSTMForTrading",
    "ForecastEnsemble",
]
