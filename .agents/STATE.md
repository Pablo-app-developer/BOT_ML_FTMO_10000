# 📊 SYSTEM STATE

**Last Updated:** 2026-02-05 23:15 UTC  
**Status:** 🔄 MIGRATING TO DOCKER  
**Active Phase:** Infrastructure V2 (Dockerization)

---

## 🏗️ Infrastructure

### Local (Training/Dev)
- **GPU:** NVIDIA GeForce RTX 3050 (Active)
- **Role:** Data Science, Training, Development

### VPS (Production)
- **Type:** Linux Server (Ubuntu/Debian)
- **Container:** Docker
- **Role:** 24/7 Execution (Inference Only)
- **Resources:** Assumed Low (1-2 vCPU, 2-4GB RAM)

---

## 💼 Account Parameters (Crypto Focus)

**ULTIMATE GOAL: PASS FTMO CHALLENGE ($10k)**
*   **Vehicle:** Crypto CFDs (BTCUSD, ETHUSD)
*   **Why:** Linux/Automation stability vs Windows/MT5 complexity
*   **Strategy:** High volatility capture with strict daily drawdown guards

```yaml
account:
  initial_balance: 10000  # USD
  type: "FTMO Aggressive / Crypto"
  currency: "USD"

risk_limits:
  max_daily_loss_pct: 0.05
  max_total_loss_pct: 0.10
  profit_target_pct: 0.10  # Phase 1 Target
  profit_target_phase2: 0.05 # Phase 2 Target

  
assets:
  primary: ["BTC/USD", "ETH/USD"]
  secondary: ["SOL/USD", "BNB/USD"]
  timeframe_execution: "15m" or "1h"
```

---

## 🤖 Data Pipeline Status

### Data Sources
- **Exchange:** Binance (via CCXT) because of high quality data.
- **Normalization:** Must convert Binance/Bybit data to match Prop Firm feed characteristics (spread simulation).

```yaml
historical_data:
  btc_15m: "AVAILABLE (Local)"
  eth_15m: "AVAILABLE (Local)"
  sol_15m: "AVAILABLE (Local)"
```

---

## 👥 Agent Tasks (Pivot)

```yaml
agents:
  quant_architect:
    status: "🔄 UPDATING"
    task: "Adapt risk parameters for Crypto volatility (wider stops, smaller position size)"
    
  ml_engineer:
    status: "🟢 READY"
    task: "Train V4 model on Crypto Data (already present) using Local GPU"
    
  python_developer:
    status: "🔥 BUSY"
    task: "Create Dockerfile, docker-compose.yml, and setup CCXT Async implementation"
    
  qa_auditor:
    status: "🟢 READY"
    task: "Verify container build and restart policies"
```

---

## 🔐 Credentials Strategy

- **Local:** `.env` file.
- **Docker:** `env_file` directive in docker-compose or environment variables injected via CI/CD.

