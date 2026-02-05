# Complete Implementation Guide: Multi-Signal CEEMDAN-ARMA-LSTM (CAL) Pipeline for Claude

**Target**: Training/inference script for multi-stock/multi-signal stock price forecasting using CEEMDAN decomposition + dual modeling (VAR/ARMA for stationary IMFs, Multivariate LSTM for non-stationary), optimized for ROCm GPU acceleration in a devcontainer.

**Your input to Claude**: Copy-paste this entire instruction. Claude will generate the complete, production-ready codebase.

***

## 🎯 Project Requirements \& Architecture

### Core Pipeline (Multi-Signal CAL)

```
Raw Data (N stocks × M signals × T timesteps)
    ↓ CEEMDAN decomposition (per signal → K IMFs + residue)
Multivariate IMF panel (N×M×K components)
    ↓ ADF stationarity test (per IMF)
├── Stationary IMFs → VAR(p) multivariate forecasting
└── Non-stationary IMFs → Multivariate LSTM (60 timesteps lookback)
    ↓ Reconstruction (sum per signal → weighted aggregate per stock)
Final Output: Joint next-day price forecasts + uncertainty bands for all stocks
```


### Hardware/Environment

- **ROCm GPU** (AMD Instinct MI series or RX 7000/7900 series)
- **Devcontainer** (VS Code + Docker)
- **Multi-GPU training** support (DataParallel)
- **Production logging** (Weights \& Biases or TensorBoard)


### Data Format

```
CSV: Date,A_price,A_volume,A_vol,B_price,B_volume,B_vol,... (daily freq)
Real data sources: yfinance (AAPL,MSFT,TSLA + VIX,SPY), FRED macro
Min 3 years daily data per signal
```


***

## 🛠️ Devcontainer Setup (`.devcontainer/devcontainer.json`)

**Base image**: `rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_release`
**Extensions**: Python, Jupyter, ROCm Tools
**Pre-install**: PyEMD, PyTorch-ROCm, yfinance, statsmodels, scikit-learn, wandb
**Volumes**: `./data:/workspace/data`, `./models:/workspace/models`
**Ports**: 8888 (Jupyter), 6006 (TensorBoard)

**Post-create script**: `pip install EMD-signal[full] tensorboard wandb`

***

## 📁 Project Structure (Generate this)

```
multi_cal_pipeline/
├── devcontainer/                 # Devcontainer config
├── src/
│   ├── data/                     # Data loading, yfinance fetcher
│   ├── decomposition/            # CEEMDAN + IMF extraction
│   ├── stationarity/             # ADF tests, classification
│   ├── models/                   # VAR, MultivariateLSTM classes
│   ├── reconstruction/           # IMF summing, stock aggregation
│   ├── training/                 # Main train loop, ROCm DataParallel
│   └── utils/                    # Config, logging, metrics (MAE, MAPE, direction acc)
├── configs/                      # YAML: stocks=['AAPL','MSFT'], signals=['price','vol','rsi']
├── data/                         # raw/ processed/ (gitignore)
├── models/                       # checkpoints/ wandb/
├── notebooks/                    # EDA, backtest prototype
├── tests/                        # unit tests for each module
├── train.py                      # CLI: python train.py --config config/stocks.yaml
├── predict.py                    # CLI inference on new data
├── backtest.py                   # Walk-forward optimization
└── requirements-rocm.txt         # PyTorch 2.4+ ROCm 6.1 pinned
```


***

## 🔧 Detailed Implementation Instructions (Module by Module)

### 1. **data/loader.py**

- **Multi-source fetcher**: yfinance for stocks/indices, FRED API for macro (10Y yield, CPI), AlphaVantage for intraday if needed.
- **Signal engineering**: Per stock → Close (pct change), Volume, Volatility (rolling 20d std), RSI(14), MACD.
- **Alignment**: Forward-fill NaNs, ensure daily freq, min 1000 days.
- **Output**: Pandas DataFrame `[Date, stock_signal1, stock_signal2, ...]`
- **Validation**: Check stationarity trends, collinearity (<0.95), missing data <5%.
- **Batch loader**: Torch Dataset for sliding windows (lookback=60 days → predict next 1/5/21 days).


### 2. **decomposition/ceemdan.py**

- **Library**: `PyEMD.CEEMDAN()` with `max_imf=10, noise_std=0.2, num_realizations=100`.
- **Parallel decomposition**: Use `joblib.Parallel` over all stock-signals (ROCm CPU cores).
- **IMF labeling**: `stock_signal_IMF1` (highest freq) → `stock_signal_residue` (trend).
- **Validation**: Reconstruction error <1% (sum IMFs ≈ original), IMF orthogonality.
- **Save intermediates**: HDF5 for fast reload (gigabytes of IMFs).


### 3. **stationarity/adf_classifier.py**

- **ADF test**: `statsmodels.tsa.stattools.adfuller` with `maxlag='auto'`, `regression='ct'`.
- **Thresholds**: p<0.05 stationary, else non-stationary.
- **Edge cases**: Very short IMFs (<50 points) → force to LSTM.
- **Multi-testing correction**: Benjamini-Hochberg FDR (handle 50+ tests).
- **Output**: Dict `{imf_col: 'stationary' | 'nonstationary'}`.


### 4. **models/var_model.py**

- **statsmodels.tsa.vector_ar.VAR**: Fit on stationary IMF panel.
- **Lag selection**: `ic='aic'` up to maxlags=10.
- **Parallel VAR**: If >20 stationary IMFs, cluster by frequency band → separate VARs.
- **Forecast**: `model.forecast(steps=1..21, exog=None)`.
- **Validation**: Granger causality tests between stocks.


### 5. **models/multivariate_lstm.py**

**ROCm-Optimized PyTorch Lightning Module**:

```
class MultiLSTMPipeline(pl.LightningModule):
    def __init__(self, n_features, n_hidden=128, n_layers=2, lr=1e-3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.attention = nn.MultiheadAttention(n_hidden*2, 8)
        self.fc = nn.Sequential(Linear(n_hidden*2, n_hidden), ReLU(),
                               Linear(n_hidden, n_features))
    
    def forward(self, x):  # [batch, 60, n_features=~40]
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1, :])  # Last timestep
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log('train_mae', F.l1_loss(pred, y), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
```

- **ROCm flags**: `torch.backends.cudnn.benchmark=True`, `torch.use_deterministic_algorithms(False)`.
- **DataParallel**: Automatic multi-GPU.
- **Early stopping**: Patience=10 on val RMSE.


### 6. **reconstruction/aggregator.py**

- **Per-signal reconstruction**: `forecast_signal = sum(forecast_imfs) + forecast_residue`.
- **Per-stock aggregation**: `stock_price_fc = 0.7*price_fc + 0.2*volatility_weighted + 0.1*momentum_factor`.
- **Uncertainty**: Quantile regression or bootstrap (10× models → 5th/95th percentiles).
- **Smoothing**: EMA(3) on final forecasts.


### 7. **training/train_pipeline.py** (Main Orchestrator)

**CLI**: `python train.py --config configs/tech_stocks.yaml --gpus 2 --wandb`

```
1. Load config → fetch/validate data
2. CEEMDAN → save IMF panel
3. ADF → split stationary/nonstat
4. Train VAR → save forecasts_stationary
5. Scale nonstat IMFs → create Dataset (80/10/10 split)
6. Train LSTM (Lightning Trainer: max_epochs=100, accumulate_grad=4)
7. Generate held-out forecasts
8. Reconstruction → compute metrics (MAE, MAPE, directional acc, Sharpe)
9. Log to WandB: artifacts (models), plots (actual vs pred), backtest PnL
10. Save: ensemble_model.pt, var_model.pickle
```


### 8. **predict.py** (Inference)

```
python predict.py --model_path models/best.pt --new_data data/2026-02.csv --stocks AAPL,MSFT
```

- Load trained models → decompose new data → forecast → reconstruct → JSON output.


### 9. **backtest.py**

- Walk-forward: Retrain monthly on 3yr rolling window.
- Metrics: Hit rate, Calmar ratio, max drawdown.
- Transaction costs: 10bps round-trip.


### 10. **configs/tech_stocks.yaml**

```yaml
stocks: ['AAPL', 'MSFT', 'TSLA', 'SPY']
signals: ['Close_pct', 'Volume', 'Vol_20d', 'RSI_14']
lookback: 60
forecast_horizon: 1  # days
imf_max: 10
wandb_project: "multi-cal-stocks"
```


***

## ⚡ ROCm-Specific Optimizations (Critical)

1. **Docker**: `rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4.0`
2. **Environment**:

```
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RX 7900
export PYTORCH_ROCM_ARCH="gfx1100:gfx1101"  # Your GPU arch
```

3. **Lightning Trainer**:

```python
trainer = pl.Trainer(
    accelerator="gpu", devices=2, strategy="ddp",
    precision=16,  # bf16 for ROCm
    benchmark=True,
    max_epochs=100
)
```

4. **Memory**: Gradient checkpointing, `torch.cuda.empty_cache()` post-epoch.

***

## 📊 Expected Results \& Validation

**On S\&P tech stocks (2020-2025)**:

- **1-day ahead**: MAPE 0.8-1.2%, directional acc 68%
- **5-day**: MAPE 2.1%, acc 62%
- **vs Benchmarks**: +45% RMSE reduction vs plain LSTM

**Success criteria**:

- Reconstruction R² >0.98 per signal
- Cross-stock Granger causality p<0.05
- Live PnL > benchmark (SPY buy-hold)

***

## 🚀 Claude Instructions

**Generate the COMPLETE codebase** following this structure exactly. Include:

1. All `.py` files with full imports, error handling, logging.
2. Devcontainer files ready-to-use.
3. `requirements-rocm.txt` pinned versions.
4. Sample `configs/` and `notebooks/eda.ipynb`.
5. Tests with 90%+ coverage.
6. README.md with setup + one-click train command.
7. Docker Compose for local ROCm emulation.

**Style**: PEP8, type hints, docstrings, modular (no >300 line files).
**Robustness**: Input validation, GPU fallback (CPU), resume training.
**Logging**: Rich progress bars + WandB dashboard.

**Test it works**: Include simulated data generator that produces realistic correlated stocks.

**Deliverable**: Zip-ready Git repo structure. Make it run **out-of-box** on ROCm devcontainer.


