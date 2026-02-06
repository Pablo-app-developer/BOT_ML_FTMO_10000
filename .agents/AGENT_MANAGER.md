# 🎯 AGENT MANAGER - Mission Control

**Status:** 🟢 ACTIVE  
**Last Updated:** 2026-02-05 21:07 UTC  
**System Version:** 1.0.0

---

## 🎮 Mission Control Overview

This is the central coordination system for the 4-agent FTMO trading bot development team.

### Active Agents

| Agent | Status | Workspace | Primary Focus |
|-------|--------|-----------|---------------|
| 📐 **Quant Architect** | 🟢 READY | `agents/quant/` | Risk management, FTMO compliance |
| 🧠 **ML Engineer** | 🟢 READY | `agents/ml/` | Model training, signal generation |
| 💻 **Python Developer** | 🟢 READY | `agents/dev/` | Code architecture, execution |
| 🔍 **QA Auditor** | 🟢 READY | `agents/qa/` | Testing, validation, failure modes |

---

## 📋 Current System State

**Hardware:**
- GPU: NVIDIA GeForce RTX 3050 (for ML training)
- Server: Home laptop with SSD

**Target:**
- FTMO Challenge: $10,000 account
- Instrument: EUR/USD (primary)
- Max Daily Loss: 5% ($500)
- Max Total Loss: 10% ($1,000)
- Profit Target: 10% ($1,000)

---

## 🔄 Workflow Protocol

### Change Request Process

```
1. 📝 PROPOSE
   → Agent writes to .agents/PROPOSALS.md
   → Includes: What, Why, Impact, Risk Assessment
   → Status: PENDING_REVIEW

2. 💻 TECHNICAL_REVIEW (Python Developer)
   → Validates feasibility
   → Checks architecture impact
   → Updates proposal: TECH_APPROVED or TECH_REJECTED

3. 📐 RISK_FILTER (Quant Architect)
   → Verifies FTMO compliance
   → Calculates risk impact
   → Updates proposal: RISK_APPROVED or RISK_REJECTED

4. 🔍 QA_VALIDATION (Auditor)
   → Tests edge cases
   → Runs stress tests
   → Updates proposal: QA_APPROVED or QA_REJECTED

5. ✅ MERGE or ❌ REJECT
   → If all approve: Execute change, log to DECISIONS.md
   → If any reject: Document reason, return to proposer
```

---

## 🚨 Critical Rules

### Rule #1: FTMO Compliance is NON-NEGOTIABLE
- Quant Architect has **VETO POWER** on any risk-related decision
- All trades MUST pass `ftmo_compliance_checker.py` before execution
- Real-time monitoring of daily and total drawdown

### Rule #2: No Concurrent File Writes
- Only ONE agent modifies a file at a time
- Use PROPOSALS.md as a locking mechanism
- Changes are queued with timestamps

### Rule #3: Evidence-Based Decisions
- All model improvements must show statistical edge on fresh data
- Backtesting is required before any strategy deployment
- QA Auditor must approve all changes to production code

### Rule #4: Resource Efficiency
- Optimize for home server constraints
- GPU (RTX 3050) reserved for ML training only
- Production inference runs on CPU

---

## 📊 Communication Channels

### Synchronous Communication
**File:** `.agents/PROPOSALS.md`
- Active change requests
- Review statuses
- Discussion threads

### State Management
**File:** `.agents/STATE.md`
- Current system configuration
- Active parameters
- Performance metrics

### Audit Trail
**File:** `.agents/DECISIONS.md`
- All approved changes
- Rejection reasons
- Historical decisions

### Agent Sync
**File:** `AGENTS_SYNC.md` (existing)
- Real-time coordination
- Task assignments
- Progress updates

---

## 🎯 Current Priorities

### Phase 1: Infrastructure ✅
- [x] Create agent workspace structure
- [x] Initialize communication files
- [x] Define workflow protocols

### Phase 2: Risk Framework 🔄
- [ ] Quant: Define FTMO risk parameters (YAML)
- [ ] Dev: Implement position sizing calculator
- [ ] Dev: Build real-time compliance checker
- [ ] QA: Test with edge cases (spread widening, slippage)

### Phase 3: Data Pipeline 📋
- [ ] ML: Design EUR/USD data pipeline
- [ ] ML: Historical data cleaning (remove outliers)
- [ ] Dev: Integrate with CCXT/broker API
- [ ] QA: Validate data integrity

### Phase 4: Model Development 📋
- [ ] ML: Train model with GPU acceleration
- [ ] ML: Cross-validation on fresh data
- [ ] Quant: Risk-adjusted performance metrics
- [ ] QA: Stress test model predictions

### Phase 5: Execution System 📋
- [ ] Dev: Broker API connector
- [ ] Dev: Order execution engine
- [ ] Quant: Position sizing integration
- [ ] QA: Failure mode testing

---

## 🔧 System Configuration

### GPU Training (RTX 3050)
```yaml
gpu:
  device: "cuda"
  model: "NVIDIA GeForce RTX 3050"
  memory: "4GB GDDR6"
  usage: "ML model training only"
  inference: "CPU (production)"
```

### Resource Limits
```yaml
limits:
  max_cpu_percent: 80
  max_ram_gb: 6
  alert_threshold_cpu: 75
  alert_threshold_ram: 5
```

---

## 📞 Agent Contact Protocol

### Emergency Halt
**Trigger:** Any agent can issue EMERGENCY_HALT  
**Conditions:**
- FTMO rule violation imminent
- Critical bug detected
- Data corruption
- Broker API failure

**Action:**
1. Write to `.agents/STATE.md` with `STATUS: HALTED`
2. Stop all trading activity
3. Notify all agents via AGENTS_SYNC.md
4. Convene emergency review

---

## 📈 Success Metrics

### System Health
- [ ] Zero conflicting file writes
- [ ] All agents responsive within 30s
- [ ] Workflow protocol followed 100%
- [ ] Complete audit trail maintained

### Trading Performance
- [ ] Pass FTMO daily drawdown check (< 5%)
- [ ] Pass FTMO total drawdown check (< 10%)
- [ ] Positive Sharpe ratio > 1.5
- [ ] Win rate > 55% on fresh data

### Code Quality
- [ ] Test coverage > 80%
- [ ] No critical bugs in QA review
- [ ] Resource usage within limits
- [ ] GitHub repo up to date

---

## 🔐 Access Control

### Write Permissions
- **AGENT_MANAGER.md:** Mission Control only (manual updates)
- **STATE.md:** All agents (read), Authorized agent (write with lock)
- **PROPOSALS.md:** All agents (append only)
- **DECISIONS.md:** Mission Control only
- **Agent workspaces:** Respective agent only

---

## 📝 Version History

### v1.0.0 (2026-02-05)
- Initial system setup
- 4 agents initialized
- Workflow protocol established
- GPU configuration added

---

**Next Review:** After Phase 2 completion  
**Emergency Contact:** Check AGENTS_SYNC.md for urgent updates
