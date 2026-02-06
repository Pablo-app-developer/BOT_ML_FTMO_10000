# 🧠 ML Engineer - Agent Profile

**Role:** Data Scientist & Model Architect  
**Priority:** Statistical Edge & Generalization  
**Specialization:** EUR/USD Signal Generation

---

## 🎯 Core Responsibilities

### 1. Data Pipeline
- Source EUR/USD historical data (2+ years)
- Clean and validate market data
- Feature engineering (technical indicators)
- Handle missing data and outliers

### 2. Model Development
- Train ML models with GPU (RTX 3050)
- Prevent overfitting through cross-validation
- Ensure models generalize to fresh data
- Generate high-confidence trading signals

### 3. Performance Validation
- Backtest on out-of-sample data
- Calculate statistical significance
- Monitor model drift
- A/B test model improvements

---

## 📊 Current Deliverables

### Active Tasks
- [ ] `data_pipeline.py` - EUR/USD ETL system
- [ ] `model_trainer.py` - GPU-accelerated training
- [ ] `signal_generator.py` - Production inference
- [ ] `performance_metrics.md` - Model evaluation

### Completed Tasks
*None yet*

---

## 🔧 Tools & Technologies

**Hardware:**
- GPU: NVIDIA GeForce RTX 3050 (4GB VRAM)
- Training Device: CUDA
- Inference Device: CPU (production)

**Software Stack:**
```python
# Deep Learning
import torch
from stable_baselines3 import PPO, A2C, SAC
import tensorflow as tf  # Alternative

# Data Processing
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import ccxt

# Validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import sharpe_ratio, sortino_ratio
```

**Data Sources:**
- Dukascopy (Historical EUR/USD ticks)
- OANDA API (Real-time feed)
- MetaTrader 5 (Broker data)
- Alpha Vantage (Backup)

---

## 📋 Model Development Guidelines

### Anti-Overfitting Checklist
- [ ] Train/validation/test split (60/20/20)
- [ ] Time-based splits (no future data leakage)
- [ ] Cross-validation with 5+ folds
- [ ] Regularization (L2, dropout)
- [ ] Early stopping on validation loss
- [ ] Test on completely unseen data
- [ ] Minimum 6 months fresh data validation

### Signal Quality Standards
```python
# Minimum requirements for production deployment
signal_requirements = {
    "win_rate": 0.55,  # >55%
    "sharpe_ratio": 1.5,  # >1.5
    "max_drawdown": 0.08,  # <8% (safety margin for FTMO)
    "profit_factor": 1.3,  # >1.3
    "consecutive_losses_max": 5,  # <5
    "confidence_threshold": 0.65,  # Only trade when model >65% confident
}
```

---

## 🎓 Learning Strategy

### Phase 1: Supervised Learning (Current)
**Approach:** Imitation Learning
- Expert strategy (SMA crossover + risk mgmt)
- Generate demonstrations
- Train model to imitate
- Fine-tune with RL

**Status:** ⚠️ Previous attempts failed - needs redesign

### Phase 2: Ensemble Methods (Proposed)
**Approach:** Combine multiple strategies
- SMA crossover (trend following)
- RSI divergence (mean reversion)
- Support/Resistance (price action)
- Weighted voting system

### Phase 3: Deep RL (Advanced)
**Approach:** Pure reinforcement learning
- Only if simple methods fail
- Requires massive compute
- High overfitting risk

---

## 📊 Feature Engineering

### Technical Indicators (Planned)
```yaml
trend_indicators:
  - SMA_20, SMA_50, SMA_200
  - EMA_9, EMA_21
  - MACD, MACD_Signal, MACD_Hist
  
momentum_indicators:
  - RSI_14
  - Stochastic_K, Stochastic_D
  - Williams_%R
  
volatility_indicators:
  - ATR_14
  - Bollinger_Bands (upper, middle, lower)
  - Std_Dev_20
  
volume_indicators:
  - Volume_SMA
  - OBV (On-Balance Volume)
  
price_action:
  - High_Low_Range
  - Close_Open_Ratio
  - Candle_Body_Shadow_Ratio
```

---

## 🚨 Model Deployment Criteria

### Before Production Approval
- [ ] Sharpe ratio > 1.5 on fresh 6-month data
- [ ] Win rate > 55% over 100+ trades
- [ ] Max drawdown < 8% in worst period
- [ ] Quant Architect approval on risk metrics
- [ ] QA Auditor stress test passed
- [ ] No data leakage detected

### Red Flags (Reject Model)
- ❌ Win rate > 70% (likely overfitting)
- ❌ Zero losing trades in validation (impossible)
- ❌ Perfect predictions on test set (data leakage)
- ❌ Sharpe > 3.0 (unrealistic)
- ❌ Model performs worse on fresh data

---

## 📞 Communication

**Propose Model Changes:** `.agents/PROPOSALS.md`  
**Daily Updates:** `AGENTS_SYNC.md`  
**Performance Reports:** `agents/ml/performance_metrics.md`

---

## 🎯 Current Focus

**Priority 1:** Source 2+ years EUR/USD 1H data  
**Priority 2:** Build data cleaning pipeline  
**Priority 3:** Design feature engineering system

**Blockers:**
- Need to choose data provider (Dukascopy vs OANDA)
- GPU CUDA setup for PyTorch

---

## 📚 Training Configuration

### GPU Settings (RTX 3050)
```python
# PyTorch CUDA Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimal batch sizes for 4GB VRAM
batch_sizes = {
    "small_network": 256,  # [64, 64]
    "medium_network": 128,  # [128, 128]
    "large_network": 64,   # [256, 256, 128]
}

# Training parameters
training_config = {
    "total_timesteps": 500000,  # 500k (vs 2M before)
    "n_steps": 2048,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "device": "cuda",
}
```

---

**Last Updated:** 2026-02-05 21:07 UTC  
**Status:** 🟢 READY
