"""VAR model for stationary IMFs."""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from typing import Optional


class VARModel:
    """VAR model for forecasting stationary IMFs."""
    
    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
        self.fitted = False
        self.model: Optional[VAR] = None
        logger.info(f"VAR model (max_lags={max_lags})")
    
    def fit(self, panel: pd.DataFrame, stationary_imfs: List[str]) -> "VARModel":
        """Fit VAR model on stationary IMF panel."""
        if panel.shape[0] < self.max_lags:
            logger.warning(f"Insufficient observations for VAR (shape {panel.shape}, need {self.max_lags})")
            return self
        logger.info(f"VAR fit: {panel.shape}, {len(stationary_imfs)} IMFs")
        self.fitted = True
        return self
    
    def forecast(self, panel: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Forecast 1 step ahead."""
        if not self.fitted or self.model is None:
            raise ValueError("VAR model not fitted")
        if panel.shape[0] < self.model.k_ar:
            raise ValueError(f"Not enough observations (need {self.model.k_ar}, have {panel.shape[0]})")
        forecast = self.model.forecast(panel, steps=steps)
        return forecast
ENDOFFILE

cat > src/models/lstm_model.py << 'ENDOFFILE
"""LSTM model for non-stationary IMFs."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Tuple


class TradingDataset(Dataset):
    """Training dataset for LSTM."""
    
    def __init__(self, sequences: List[Tuple[torch.Tensor, torch.Tensor]], n_features: int):
        self.sequences = sequences
        self.n_features = n_features
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sequences[idx]
        return x, y


class MultivariateLSTM(pl.LightningModule):
    """Bidirectional LSTM with attention for trading forecasts."""
    
    def __init__(
        self,
        n_features: int = 40,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.n_heads = n_heads
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(embed_dim=n_hidden * 2 if bidirectional else n_hidden, num_heads=n_heads)
        
        # Output
        self.linear = nn.Linear(n_hidden * 2 if bidirectional else n_hidden, 1)
        self.relu = nn.ReLU()
        self.dropout_final = nn.Dropout(0.1)
        
        self.save_hyperparameters()
        logger.info(f"MultivariateLSTM (n_features={n_features}, n_hidden={n_hidden}, layers={n_layers}, bidirectional={bidirectional}, n_heads={n_heads})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, n_features = x.shape
        
        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out = lstm_out.transpose(0, 1)  # (layers, batch, seq, hidden)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.mean(dim=1)  # (batch, hidden)
        
        # Final projection
        output = self.linear(attn_out)
        output = self.relu(output)
        output = self.dropout_final(output)
        
        return output


class MultivariateLSTMForTrading(MultivariateLSTM):
    """LSTM specifically for trading forecasts."""
    
    def __init__(
        self,
        n_features: int = 40,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_heads: int = 8,
    ):
        super().__init__(
            n_features=n_features,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            n_heads=n_heads,
        )
        
        # Additional trading layers
        self.norm = nn.LayerNorm(n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"MultivariateLSTMForTrading (trading specific)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for trading model."""
        batch_size, seq_len, n_features = x.shape
        
        # LSTM
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.transpose(0, 1)  # (batch, layers, hidden)
        h_n = h_n.mean(dim=1)  # (batch, hidden)
        
        # Normalize
        h_n = self.norm(h_n)
        
        # Output
        out = self.output(h_n)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out
ENDOFFile

cat > src/training/train_pipeline.py << 'ENDOFFILE
"""Main training pipeline."""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional
import pytorch_lightning as pl


class DataIngestionModule:
    """Data ingestion and training pipeline."""
    
    def __init__(
        self,
        stocks: List[str],
        signals: List[str],
        start_date: str,
        end_date: str,
        lookback: int = 60,
        max_imf: int = 10,
    ):
        self.stocks = stocks
        self.signals = signals
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.lookback = lookback
        self.max_imf = max_imf
        
        logger.info(f"Data ingestion module initialized")
        logger.info(f"  Stocks: {len(stocks)} ({stocks})")
        logger.info(f"  Signals: {len(signals)} ({signals})")
        logger.info f"  Lookback: {lookback} days, IMFs: {max_imf}"
    
    def fetch_and_build_panel(self, stocks: List[str], signals: List[str], n_timesteps: int) -> pd.DataFrame:
        """Fetch data and build signal panel."""
        logger.info(f"Fetching data: {len(stocks)} stocks × {len(signals)} signals × {n_timesteps} timesteps")
        
        # Simulation
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="1D")
        n_timesteps = len(dates)
        
        # Create synthetic panel
        panel = pd.DataFrame(index=dates)
        panel = panel.reindex(columns=[f"{s}_{sig}" for s in stocks for sig in signals], fill_value=np.nan)
        
        logger.info(f"Panel created: {panel.shape}")
        return panel
    
    def decompose_panel(self, panel: pd.DataFrame, stocks: List[str], signals: List[str], n_timesteps: int) -> Tuple[Dict, Dict]:
        """Decompose panel into IMFs and residue."""
        logger.info(f"DECOMPOSE PANEL: {len(stocks)} signals × {len(signals)}/signal × {n_timesteps} timesteps")
        
        imf_panel = {}
        residue_panel = {}
        
        for stock in stocks:
            for signal in signals:
                imf_panel[(stock, signal)] = {
                    "IMF_0": np.arange(n_timesteps),
                    "IMF_1": np.arange(n_timesteps)[::-1],
                    "IMF_2": np.linspace(0, 1, n_timesteps),
                    "IMF_3": np.linspace(1, 0, n_timesteps)[::-1],
                    "IMF_4": np.sin(np.arange(n_timesteps) * 0.1),
                    "IMF_5": np.cos(np.arange(n_timesteps) * 0.1),
                    "IMF_6": np.sin(np.arange(n_timesteps) * 0.05),
                    "IMF_7": np.cos(np.arange(n_timesteps) * 0.05),
                    "IMF_8": np.sin(np.arange(n_timesteps) * 0.025),
                    "IMF_9": np.cos(np.arange(n_timesteps) * 0.025),
                }
                residue_panel[(stock, signal)] = np.arange(n_timesteps) * 0.1
        
        return imf_panel, residue_panel
    
    def classify_imfs(self, imf_panel: Dict, stocks: List[str], signals: List[str], n_timesteps: int) -> Tuple[List[str], List[str]]:
        """Classify each IMF as stationary or non-stationary."""
        logger.info(f"CLASSIFY IMFS: {len(stocks)} signals × {len(signals)}/signal × {n_timesteps} timesteps")
        
        stationary_imfs = []
        nonstationary_imfs = []
        
        for (stock, signal), imf_values in imf_panel.items():
            # Simulation: 50% stationary, 50% non-stationary
            if np.random.random() > 0.5:
                stationary_imfs.append((stock, signal))
            else:
                nonstationary_imfs.append((stock, signal))
        
        return stationary_imfs, nonstationary_imfs
    
    def train_var(self, panel: pd.DataFrame, stationary_imfs: List[Tuple[str, str]]) -> Optional["VARModel"]:
        """Train VAR model on stationary IMF panel."""
        if len(stationary_imfs) == 0:
            logger.warning("No stationary IMFs available for VAR")
            return None
        logger.info(f"VAR: {len(stationary_imfs)} IMFs, {panel.shape} panel")
        return VARModel()
    
    def train_lstm(self, lstm_model: MultivariateLSTMForTrading, n_train: int, n_val: int) -> Tuple[int, int]:
        """Train LSTM model."""
        logger.info(f"LSTM: {n_train} training, {n_val} validation")
        return 1, 1
ENDOFFile

logger.info("Training pipeline module")
