## Proposed Roadmap

**5 phases** | **20 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|-----------------|
| 1 | Data Ingestion | Fetch and structure multi-stock signal data | AUTH-01 through AUTH-04 | 95%+ fetch success, <5% missing, 1000+ timesteps |
| 2 | CEEMDAN Decomposition | Extract IMFs with validation metrics | CONT-01 through CONT-04 | 98%+ reconstruction R², <0.2 orthogonality |
| 3 | Stationarity Classification | Classify IMFs as stationary/non-stationary | SPLIT-01 through SPLIT-03 | 90%+ ADF accuracy, 20+ tests applied |
| 4 | Model Training | Achieve <1.5 MAE, >65% directional | TRAIN-01 through TRAIN-04 | VAR MAE <0.8, LSTM MAE <1.5, dir acc >65% |
| 5 | Ensembling & Validation | <1.5 RMSE, >10% beat buy-hold | ENSEMBLE-01 through ENSEMBLE-04 | RMSE <1.5, PnL >10%, drawdown <15% |

### Phase Details

**Phase 1: Data Ingestion**

Goal: Fetch and structure multi-stock signal data

Requirements: AUTH-01 through AUTH-04

Success criteria:
1. Single yfinance call fetches N stocks × M signals (AAPL, MSFT, TSLA, SPY; Close, Volume, RSI, Vol)
2. Signal engineering yields 1000+ daily timesteps per signal
3. Forward fill handles <5% gaps, drops <10%
4. Output DataFrame with 30+ columns, 1000+ rows
5. Validation passes: collinearity <0.95, missing <5%, timesteps ≥1000

**Phase 2: CEEMDAN Decomposition**

Goal: Extract IMFs with validation

Requirements: CONT-01 through CONT-04

Success criteria:
1. Each signal decomposes into ≤10 IMFs (max_imf=10, num_realizations=100)
2. Reconstruction error <1% (sum IMFs + residue ≈ original)
3. Orthogonality verified (correlation <0.2 for adjacent IMFs)
4. Per-signal-IMF panel stored (N×M×K shape, HDF5)

**Phase 3: Stationarity Classification**

Goal: Classify IMFs via ADF test

Requirements: SPLIT-01 through SPLIT-03

Success criteria:
1. 90%+ classification accuracy vs ground truth
2. Edge cases (<50 points) correctly routed
3. Benjamini-Hochberg FDR applied (20+ tests)

**Phase 4: Model Training**

Goal: Achieve <1.5 MAE, >65% directional

Requirements: TRAIN-01 through TRAIN-04

Success criteria:
1. VAR forecast MAE <0.8 for stationary IMFs
2. LSTM forecast MAE <1.5 for non-stationary IMFs
3. Directional accuracy >65% (sign of change correct)
4. Granger causality p-value <0.05 (confirms predictive power)

**Phase 5: Ensembling & Validation**

Goal: <1.5 RMSE, >10% beat buy-hold

Requirements: ENSEMBLE-01 through ENSEMBLE-04

Success criteria:
1. Forecast ensemble RMSE <1.5 (all signals)
2. Backtest PnL >10% annualized (beat SPY)
3. Max drawdown <15% (vs 20% for buy-hold)
4. Calmar ratio >0.7 (return/dividend)

---

## Roadmap Created ✓
