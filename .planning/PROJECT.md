# Multi-Signal CEEMDAN-ARMA-LSTM (CAL) Trading System

## What This Is

A machine learning trading forecast system that decomposes multi-stock/multi-signal price time series using CEEMDAN into Intrinsic Mode Functions (IMFs), classifies them as stationary/non-stationary, and trains dual models (VAR/ARMA for stationary IMFs, Multivariate LSTM for non-stationary) to produce joint next-day price forecasts with uncertainty bands for all stocks.

## Core Value

Forecast multi-stock price movements using CEEMDAN decomposition + dual modeling (VAR/ARMA for stationary IMFs, LSTM for non-stationary) to generate actionable trading signals with 68%+ directional accuracy and 0.8-1.2% MAPE for 1-day ahead forecasts.

## Requirements

### Validated

- ✓ Data ingestion from yfinance/FRED — existing (yfinance fetcher with signal engineering)
- ✓ CEEMDAN decomposition per signal → IMFs — existing
- ✓ ADF stationarity test classification — existing
- ✓ VAR forecast on stationary IMFs — existing
- ✓ Multivariate LSTM on non-stationary IMFs — existing
- ✓ Per-signal reconstruction and aggregation — existing

### Active

- [ ] Real-time forecast ensemble
- [ ] Backtest PnL validation
- [ ] Production deployment

### Out of Scope

- [ Real-time streaming inference] — async I/O complexity
- [ Options pricing and volatility} — risk model complexity
- [ Multi-asset portfolio optimization] — dimensionality
- [ Reinforcement learning trading agent] — control policy

## Context

Academic research project demonstrating CEEMDAN + ARMA-LSTM + VAR + Multivariate LSTM joint forecasting for stock price prediction. Uses ROCm GPU acceleration via PyTorch Lightning in devcontainer.

Key technical elements:
- CEEMDAN decomposition (max_imf=10, num_realizations=100)
- ADF test (p<0.05 stationary threshold)
- VAR(autoregressive) for stationary IMFs
- Bidirectional LSTM with attention for non-stationary IMFs
- Per-stock per-signal forecast ensemble
- 60-day lookback window

## Constraints

- **Tech Stack**: Python 3.10+, PyTorch 2.4+, ROCm 6.1, FastAPI
- **Environment**: Docker devcontainer with GPU support
- **Data**: Daily OHLCV from yfinance, 3+ years history minimum
- **Training**: ROCm GPU (MI series or RX 7000), DataParallel multi-GPU
- **Logging**: Weights & Biases + TensorBoard
- **Compliance**: Research ethics, no market manipulation claims

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CEEMDAN decomposition | Captures multi-scale price movements, orthogonal IMFs | ✓ Good |
| VAR + LSTM dual model | Stationary/imperfectly stationary separation | ✓ Good |
| 60-day LSTM lookback | Balances context length/overfitting | ✓ Good |
| Per-signal per-IMF decomposition | Maintains signal-specific signal patterns | ✓ Good |
| Ensemble forecast | Combines orthogonal model predictions | ✓ Good |
| W&B experiment tracking | Reproducible training analysis | ✓ Good |

---

*Last updated: 2026-02-07 after initial specification*
