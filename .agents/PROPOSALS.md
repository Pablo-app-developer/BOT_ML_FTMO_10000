# 📝 PROPOSALS - Change Request Queue

**Last Updated:** 2026-02-05 21:07 UTC  
**Active Proposals:** 0  
**Status:** 🟢 READY

---

## 📋 How to Submit a Proposal

```markdown
## [PROP-{ID}] {Short Title}

**Submitted by:** {Agent Name}  
**Date:** {YYYY-MM-DD HH:MM UTC}  
**Status:** PENDING_REVIEW  
**Priority:** LOW | MEDIUM | HIGH | CRITICAL

### What
{Clear description of the proposed change}

### Why
{Justification - what problem does this solve?}

### Impact
{Which systems/files will be affected?}

### Risk Assessment
{Potential risks, FTMO compliance considerations}

### Implementation Plan
{Step-by-step how this will be done}

---

### Review Status

#### Technical Review (Python Developer)
**Status:** PENDING | APPROVED | REJECTED  
**Reviewer:** {Name}  
**Date:** {Date}  
**Notes:** {Feasibility, architecture impact, resource requirements}

#### Risk Filter (Quant Architect)
**Status:** PENDING | APPROVED | REJECTED  
**Reviewer:** {Name}  
**Date:** {Date}  
**Notes:** {FTMO compliance, risk impact, position sizing effects}

#### QA Validation (Auditor)
**Status:** PENDING | APPROVED | REJECTED  
**Reviewer:** {Name}  
**Date:** {Date}  
**Notes:** {Edge cases, stress test results, failure modes}

---

### Final Decision
**Status:** APPROVED | REJECTED | DEFERRED  
**Decided by:** Mission Control  
**Date:** {Date}  
**Action:** {Next steps or rejection reason}

---
```

---

## 🔴 Active Proposals

*No active proposals at this time.*

---

## 🟢 Approved Proposals

*No approved proposals yet.*

---

## ❌ Rejected Proposals

*No rejected proposals yet.*

---

## 📊 Proposal Statistics

```yaml
total_submitted: 0
approved: 0
rejected: 0
deferred: 0
avg_approval_time: "N/A"
```

---

## 📌 Proposal ID Counter

**Next ID:** PROP-001

*(Auto-increment for each new proposal)*

---

## 🚨 Emergency Fast-Track

**Conditions for bypassing standard review:**
1. FTMO rule violation imminent
2. Critical security issue
3. Data corruption detected
4. Broker API failure

**Process:**
1. Mark proposal as `CRITICAL`
2. Notify all agents via AGENTS_SYNC.md
3. Quant Architect must still approve if risk-related
4. Log to DECISIONS.md immediately after action

---

## 💡 Proposal Templates

### Risk Parameter Change
```markdown
## [PROP-XXX] Update Stop-Loss Level

**Submitted by:** Quant Architect  
**Priority:** HIGH

### What
Change stop-loss from 2% to 1.5% to reduce drawdown risk.

### Why
Current 2% allows too many losses to accumulate near daily limit.

### Impact
- `agents/quant/risk_parameters.yaml`
- All position sizing calculations

### Risk Assessment
**POSITIVE:** Reduces daily drawdown risk by 25%  
**NEGATIVE:** May increase stopped-out trades

### Implementation Plan
1. Update `risk_parameters.yaml`
2. Dev: Modify position sizing calculator
3. QA: Backtest with new parameters
```

### Model Improvement
```markdown
## [PROP-XXX] New Feature: ATR-Based Position Sizing

**Submitted by:** ML Engineer  
**Priority:** MEDIUM

### What
Add ATR (Average True Range) to adjust position size dynamically.

### Why
Fixed position sizing doesn't account for volatility changes.

### Impact
- `agents/ml/signal_generator.py`
- `agents/quant/position_sizing.py`

### Risk Assessment
**POSITIVE:** Better risk-reward in volatile markets  
**NEGATIVE:** Complexity increase, needs testing

### Implementation Plan
1. ML: Calculate ATR in data pipeline
2. Quant: Define ATR-to-lot-size formula
3. Dev: Integrate into execution engine
4. QA: Validate with historical volatility spikes
```

---

**Review Schedule:** Check every 4 hours or on-demand  
**SLA:** All proposals reviewed within 24 hours
