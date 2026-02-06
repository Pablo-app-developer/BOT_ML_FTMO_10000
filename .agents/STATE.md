# 📊 SYSTEM STATE

**Last Updated:** 2026-02-05 21:07 UTC  
**Status:** 🟢 OPERATIONAL  
**Active Phase:** Phase 1 - Infrastructure Setup

---

## 🖥️ Hardware Configuration

### Compute Resources
```yaml
cpu:
  type: "x86_64"
  cores: 12
  usage_limit: 80%
  
gpu:
  model: "NVIDIA GeForce RTX 3050"
  memory: "4GB GDDR6"
  cuda_version: "11.8+"
  purpose: "ML Training Only"
  
memory:
  total_ram: "8GB"
  usage_limit: "6GB"
  swap: "Enabled"
  
storage:
  type: "SSD"
  available: "500GB+"
  data_path: "data/"
  models_path: "models/"
```

### Network
```yaml
internet:
  connection: "Home Broadband"
  backup: "Mobile Hotspot (todo)"
  latency_target: "<100ms to broker"
```

---

## 💼 FTMO Account Parameters

```yaml
account:
  initial_balance: 10000  # USD
  current_balance: 10000  # USD (not live yet)
  equity: 10000
  
risk_limits:
  max_daily_loss_pct: 0.05  # 5%
  max_daily_loss_usd: 500
  max_total_loss_pct: 0.10  # 10%
  max_total_loss_usd: 1000
  profit_target_pct: 0.10  # 10%
  profit_target_usd: 1000
  
current_risk:
  daily_drawdown: 0.0
  total_drawdown: 0.0
  daily_pnl: 0.0
  total_pnl: 0.0
```

---

## 📈 Trading Configuration

```yaml
instruments:
  primary: "EUR/USD"
  secondary: []  # May add GBP/USD, USD/JPY later
  
timeframes:
  analysis: "1H"  # For signals
  execution: "15M"  # For entries
  risk_check: "1M"  # For monitoring
  
trading_hours:
  start: "00:00 UTC"  # 24/5
  end: "23:59 UTC"
  weekend_trading: false
  
position_sizing:
  method: "risk_based"
  risk_per_trade_pct: 0.01  # 1% of account
  max_position_size_lots: 0.5
  min_position_size_lots: 0.01
```

---

## 🤖 ML Model Status

### Current Model: V3 Imitation Learning

```yaml
model:
  version: "v3_imitation"
  path: "models/FTMO_IMITATION/ftmo_bot.zip"
  architecture: "PPO"
  network: "[64, 64]"
  trained_on: "BTC/ETH/SOL 15m data"
  training_steps: 301000
  device_trained: "cpu"
  
validation_results:
  btc_fresh_60d:
    pass_rate: 0.0
    avg_profit_pct: 0.0
  eth_fresh_60d:
    pass_rate: 0.0
    avg_profit_pct: -9.52
  sol_fresh_60d:
    pass_rate: 0.0
    avg_profit_pct: -0.06
  overall:
    pass_rate: 0.0
    avg_profit_pct: -3.19
    
status: "⚠️ NOT READY FOR PRODUCTION"
notes: "Model failed validation - needs complete redesign for EUR/USD"
```

---

## 📊 Data Status

### Available Data

```yaml
historical_data:
  btc_binance_15m:
    rows: 17081
    period: "6 months"
    source: "Binance via CCXT"
    path: "data/btc_binance_15m.csv"
    
  eth_validation_60d:
    rows: 5561
    period: "60 days"
    source: "Binance via CCXT"
    path: "data/eth_validation_60d.csv"
    
  sol_validation_60d:
    rows: 5561
    period: "60 days"
    source: "Binance via CCXT"
    path: "data/sol_validation_60d.csv"
    
  eurusd:
    status: "⚠️ NEEDED"
    required_period: "2+ years"
    timeframe: "1H"
    source: "TBD (OANDA, MT5, or Dukascopy)"
```

---

## 👥 Agent Status

```yaml
agents:
  quant_architect:
    status: "🟢 READY"
    workspace: "agents/quant/"
    current_task: "Define FTMO risk parameters"
    
  ml_engineer:
    status: "🟢 READY"
    workspace: "agents/ml/"
    current_task: "Design EUR/USD data pipeline"
    
  python_developer:
    status: "🟢 READY"
    workspace: "agents/dev/"
    current_task: "Create broker connector skeleton"
    
  qa_auditor:
    status: "🟢 READY"
    workspace: "agents/qa/"
    current_task: "Define stress test scenarios"
```

---

## 📋 Active Proposals

```yaml
proposals:
  count: 0
  pending_review: 0
  approved: 0
  rejected: 0
```

*(See `.agents/PROPOSALS.md` for details)*

---

## 🎯 Current Objectives

### Immediate (This Session)
- [x] Create agent infrastructure
- [x] Initialize AGENT_MANAGER.md
- [x] Create STATE.md (this file)
- [ ] Create PROPOSALS.md template
- [ ] Create DECISIONS.md template
- [ ] Initialize agent workspaces

### Short-Term (Next 24-48h)
- [ ] Quant: FTMO risk parameter file
- [ ] ML: EUR/USD data sourcing plan
- [ ] Dev: Broker API research
- [ ] QA: Initial stress test suite

### Medium-Term (Week 1)
- [ ] Complete risk framework
- [ ] EUR/USD historical data downloaded
- [ ] Initial model training with GPU
- [ ] Backtesting infrastructure

---

## 🔐 Security

```yaml
api_keys:
  broker: "NOT_SET"
  data_provider: "NOT_SET"
  telegram_bot: "NOT_SET"
  
credentials_storage:
  method: "Environment variables"
  file: ".env (gitignored)"
  
api_rate_limits:
  broker: "TBD"
  data_provider: "Binance: 1200/min"
```

---

## 📞 Communication Files

```yaml
files:
  agent_manager: ".agents/AGENT_MANAGER.md"
  state: ".agents/STATE.md" # This file
  proposals: ".agents/PROPOSALS.md"
  decisions: ".agents/DECISIONS.md"
  sync: "AGENTS_SYNC.md"
```

---

## 🚨 Alerts & Monitoring

```yaml
monitoring:
  system_health:
    cpu_threshold: 75%
    ram_threshold: 5GB
    disk_threshold: 50GB
    check_interval: "60s"
    
  trading_health:
    daily_dd_warning: 4.0%  # Warn before 5% limit
    total_dd_warning: 8.0%  # Warn before 10% limit
    position_count_max: 3
    
  notifications:
    console: true
    file_log: true
    telegram: false  # Future
    email: false  # Future
```

---

## 🔄 Last System Changes

```yaml
changes:
  - timestamp: "2026-02-05 21:07"
    type: "INITIALIZATION"
    description: "Multi-agent system initialized"
    author: "Mission Control"
```

---

## 📝 Notes

- GPU (RTX 3050) will significantly speed up training vs previous CPU-only runs
- Need to configure CUDA and verify PyTorch GPU support
- V3 model failed validation - starting fresh with EUR/USD focus
- Priority: Get quality EUR/USD data before any model training

---

**Status Check Frequency:** Every agent action  
**Full Review:** Daily at 00:00 UTC  
**Emergency Updates:** As needed via AGENTS_SYNC.md
