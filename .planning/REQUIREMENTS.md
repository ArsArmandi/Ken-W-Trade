# Requirements: Multi-Signal CEEMDAN-ARMA-LSTM

**Defined:** 2026-02-07
**Core Value:** Forecast multi-stock price movements using CEEMDAN + ARMA-LSTM + VAR

## v1 Requirements

### Data Ingestion

- [ ] **AUTH-01**: User can fetch N stocks × M signals (AAPL, MSFT, TSLA, SPY) with signals (Close, Volume, RSI, Vol) from yfinance
- [ ] **AUTH-02**: Signal engineering (pct change, rolling std, RSI, MACD) produces 1000+ daily timesteps per signal
- [ ] **AUTH-03**: Forward fill handles <5% gaps, drops <10% of total
- [ ] **AUTH-04**: Output DataFrame structured [Date, stock, signal, value] with 30+ columns

### Decomposition

- [ ] **CONT-01**: CEEMDAN decomposes each signal into ≤10 IMFs (max_imf=10, num_realizations=100)
- [ ] **CONT-02**: Reconstruction error <1% (sum IMFs + residue ≈ original)
- [ ] **CONT-03**: Orthogonality verified (correlation <0.2 for adjacent IMFs)
- [ ] **CONT-04**: Per-signal-IMF panel stored (N×M×K shape, HDF5)

### Classification

- [ ] **SPLIT-01**: ADF test classifies each IMF as stationary/non-stationary (p<0.05 stationary)
- [ ] **SPLIT-02**: Edge cases (<50 points) correctly routed to LSTM
- [ ] **SPLIT-03**: Benjamini-Hochberg FDR applied (20+ tests)

### Training

- [ ] **TRAIN-01**: VAR forecast MAE <0.8 for stationary IMFs
- [ ] **TRAIN-02**: LSTM forecast MAE <1.5 for non-stationary IMFs
- [ ] **TRAIN-03**: Directional accuracy >65% (sign of change correct)
- [ ] **TRAIN-04**: Granger causality p-value <0.05 (confirms predictive power)

### Ensembling

- [ ] **ENSEMBLE-01**: Forecast ensemble RMSE <1.5 (all signals)
- [ ] **ENSEMBLE-02**: Backtest PnL beats SPY buy-hold (+10% annualized)
- [ ] **ENSEMBLE-03**: Max drawdown <15% (vs 20% for buy-hold)
- [ ] **ENSEMBLE-04**: Calmar ratio >0.7 (return/dividend)

## v2 Requirements

### Forecast Enhancement

- [ ] **FORF-01**: Real-time forecast ensemble (streaming)
- [ ] **FORF-02**: Confidence intervals (5th/95th quantiles)

### Production

- [ ] **PROD-01**: Fast inference (≤100ms per forecast)
- [ ] **PROD-02**: Production deployment (REST API)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time streaming | Async complexity |
| Options pricing | Risk model |
| Portfolio optimization | Dimensionality |
| RL trading agent | Control policy |
| Live trading system | Ethics/verification |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUTH-01 | 1 | Pending |
| AUTH-02 | 1 | Pending |
| AUTH-03 | 1 | Pending |
| AUTH-04 | 1 | Pending |
| CONT-01 | 2 | Pending |
| CONT-02 | 2 | Pending |
| CONT-03 | 2 | Pending |
| CONT-04 | 2 | Pending |
| SPLIT-01 | 3 | Pending |
| SPLIT-02 | 3 | Pending |
| SPLIT-03 | 3 | Pending |
| TRAIN-01 | 4 | Pending |
| TRAIN-02 | 4 | Pending |
| TRAIN-03 | 4 | Pending |
| TRAIN-04 | 4 | Pending |
| ENSEMBLE-01 | 5 | Pending |
| ENSEMBLE-02 | 5 | Pending |
| ENSEMBLE-03 | 5 | Pending |
| ENSEMBLE-04 | 5 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

---

*Requirements defined: 2026-02-07*
*Last updated: 2026-02-07 after initial definition*
