# 🎯 AGENT MANAGER - Mission Control

**Status:** 🟢 ACTIVE  
**Last Updated:** 2026-02-05 23:15 UTC  
**System Version:** 2.0.0 (Crypto/Docker Pivot)

---

## 🎮 Mission Control Overview

Central coordination for the FTMO Crypto Bot (Dockerized).

### Active Agents

| Agent | Status | Workspace | Primary Focus |
|-------|--------|-----------|---------------|
| 📐 **Quant Architect** | 🟢 READY | `agents/quant/` | Crypto risk mgmt, Volatility filters |
| 🧠 **ML Engineer** | 🟢 READY | `agents/ml/` | Local GPU Training -> VPS Inference |
| 💻 **Python Developer** | 🟢 READY | `agents/dev/` | Docker, CCXT, Linux resilience |
| 🔍 **QA Auditor** | 🟢 READY | `agents/qa/` | Container health, Connectivity tests |

---

## 📋 System Architecture

**Development & Training (HOME):**
- Hardware: Laptop + NVIDIA RTX 3050
- Task: Heavy ML training, Data processing, Backtesting
- OS: Windows 

**Production Execution (VPS):**
- Environment: Docker Containers (Linux)
- Task: Live Trading, Inference, Lightweight Execution
- Broker: **Crypto Exchange (via CCXT)** - Targeting FTMO Crypto-friendly evaluation or direct crypto prop firms.

**Target:**
- FTMO Challenge (Crypto pairs) or Crypto Prop Firm
- Instrument: **BTC/USD, ETH/USD, SOL/USD**
- Execution: 24/7

---

## 🔄 Workflow Protocol (Updated)

1. **Training (Local):** ML Agent trains models with GPU.
2. **Packaging:** Dev Agent wraps model + code into Docker image.
3. **Deployment:** Push to VPS / Pull & Run.
4. **Monitoring:** QA Agent monitors container health logs.

---

## 🚨 Critical Rules

### Rule #1: Linux Native
- No Windows dependencies (win32api, etc.)
- Everything must run in a headless Debian/Alpine container.
- Use `ccxt` for ALL execution (no compiled MT5 bridges).

### Rule #2: Resource Efficiency (VPS)
- Production container must use < 1GB RAM if possible.
- CPU inference only (models must be optimized).

### Rule #3: Crypto Volatility Risk
- **Quant Veto:** Strategy must handle 10%+ flash crashes.
- Stop-losses are mandatory and must be on-exchange (if possible) or strictly managed via WebSocket.

---

## 🔧 System Configuration

### Production (Docker)
```yaml
base_image: "python:3.10-slim"
orchestration: "docker-compose"
restart_policy: "always"
logging: "json-file"
```

### Connectivity
- **Exchange:** Binance / Bybit / FTMO (via Demo/API if avail)
- **Protocol:** HTTP REST + WebSockets (asyncio)

---

**Next Review:** After Docker setup  
**Emergency Contact:** Check AGENTS_SYNC.md
