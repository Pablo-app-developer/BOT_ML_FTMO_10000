# 💻 Python Developer - Agent Profile

**Role:** Software Architect & Execution Engineer  
**Priority:** Code Quality & System Reliability  
**Specialization:** API Integration & Error Handling

---

## 🎯 Core Responsibilities

### 1. Code Architecture
- Maintain clean, modular codebase
- Implement design patterns
- Ensure type safety and documentation
- Code reviews for all agents

### 2. Broker Integration
- MT5 or broker API connection
- Order execution logic
- Handle API rate limits
- Latency optimization

### 3. System Efficiency
- Optimize for home server (laptop + SSD)
- CPU/RAM monitoring
- Efficient data storage
- Error recovery mechanisms

---

## 📊 Current Deliverables

### Active Tasks
- [ ] `broker_connector.py` - MT5/broker API wrapper
- [ ] `execution_engine.py` - Order placement logic
- [ ] `error_handler.py` - Robust error recovery
- [ ] `monitoring.py` - System health checks

### Completed Tasks
*None yet*

---

## 🔧 Tech Stack

### Core Libraries
```python
# Broker APIs
import MetaTrader5 as mt5  # Option 1
from oandapyV20 import API  # Option 2
import ibapi  # Option 3: Interactive Brokers

# Async Processing
import asyncio
import aiohttp

# Monitoring
import psutil  # CPU/RAM
import watchdog  # File system events
from apscheduler.schedulers.background import BackgroundScheduler

# Logging
import logging
from logging.handlers import RotatingFileHandler

# Config
import yaml
from pydantic import BaseModel  # Type validation
from dotenv import load_dotenv
```

### Architecture Patterns
- **Repository Pattern:** Data access layer
- **Strategy Pattern:** Trading strategies
- **Observer Pattern:** Event-driven monitoring
- **Factory Pattern:** Order creation
- **Singleton:** Broker connection

---

## 📋 Code Quality Standards

### Python Style
```python
# PEP 8 compliant
# Type hints mandatory
from typing import Optional, List, Dict, Tuple

def calculate_position_size(
    balance: float,
    risk_pct: float,
    stop_loss_pips: int,
    pip_value: float
) -> float:
    """
    Calculate position size in lots.
    
    Args:
        balance: Account balance in USD
        risk_pct: Risk percentage (0.01 = 1%)
        stop_loss_pips: Distance to stop-loss in pips
        pip_value: Value of 1 pip in USD
    
    Returns:
        Position size in standard lots
        
    Raises:
        ValueError: If balance <= 0 or risk_pct > 0.10
    """
    if balance <= 0:
        raise ValueError("Balance must be positive")
    if risk_pct > 0.10:
        raise ValueError("Risk cannot exceed 10%")
    
    risk_amount = balance * risk_pct
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)
```

### Testing Requirements
```python
# Unit tests with pytest
import pytest

def test_position_size_calculation():
    """Test basic position sizing"""
    balance = 10000
    risk = 0.01  # 1%
    sl_pips = 20
    pip_value = 10  # 1 standard lot EUR/USD
    
    size = calculate_position_size(balance, risk, sl_pips, pip_value)
    assert size == 0.50, "Should calculate 0.5 lots"

def test_position_size_validation():
    """Test input validation"""
    with pytest.raises(ValueError):
        calculate_position_size(-1000, 0.01, 20, 10)
```

---

## 🏗️ System Architecture

### Broker Connector
```python
class BrokerConnector:
    """
    Abstract base class for broker connections
    """
    def __init__(self, credentials: Dict):
        self.credentials = credentials
        self.connected = False
        
    async def connect(self) -> bool:
        """Establish connection to broker"""
        raise NotImplementedError
        
    async def disconnect(self) -> None:
        """Close broker connection"""
        raise NotImplementedError
        
    async def get_account_info(self) -> Dict:
        """Fetch account balance, equity, margin"""
        raise NotImplementedError
        
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        lots: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> str:
        """
        Place market order
        
        Returns:
            order_id: Broker's order ID
        """
        raise NotImplementedError
```

### Error Handler
```python
class ErrorHandler:
    """
    Centralized error handling and recovery
    """
    def __init__(self):
        self.setup_logging()
        
    @staticmethod
    def handle_connection_error(error: Exception) -> bool:
        """
        Handle broker disconnection
        
        Returns:
            True if reconnected, False if failed
        """
        logging.error(f"Connection lost: {error}")
        
        # Retry logic
        for attempt in range(3):
            try:
                # Attempt reconnection
                broker.connect()
                logging.info("Reconnected successfully")
                return True
            except Exception as e:
                logging.warning(f"Retry {attempt+1}/3 failed")
                time.sleep(5)
        
        # Failed - trigger emergency halt
        logging.critical("Cannot reconnect - HALTING SYSTEM")
        send_alert("EMERGENCY: Broker connection lost")
        return False
```

---

## 🚨 Error Handling Strategy

### Network Errors
**Triggers:**
- Broker API timeout
- Internet disconnection
- DNS resolution failure

**Response:**
1. Log error with timestamp
2. Retry 3 times (5s delay)
3. If failed: Close all positions
4. Alert user
5. Halt trading

### API Rate Limits
**Triggers:**
- 429 Too Many Requests
- Rate limit exceeded

**Response:**
1. Implement exponential backoff
2. Queue pending requests
3. Respect broker limits (e.g., 10 req/sec)

### Invalid Orders
**Triggers:**
- Insufficient margin
- Invalid lot size
- Market closed

**Response:**
1. Log rejection reason
2. Do NOT retry automatically
3. Alert Quant Architect
4. Skip this signal

---

## 📊 System Monitoring

### Resource Tracking
```python
import psutil

def monitor_system():
    """Check CPU/RAM every 60 seconds"""
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_percent = psutil.virtual_memory().percent
    
    # Alert if thresholds exceeded
    if cpu_percent > 75:
        logging.warning(f"High CPU: {cpu_percent}%")
    if ram_percent > 75:
        logging.warning(f"High RAM: {ram_percent}%")
    
    # Log to STATE.md
    update_state_file({
        "cpu_usage": cpu_percent,
        "ram_usage": ram_percent,
        "timestamp": datetime.utcnow()
    })
```

### Health Checks
- Broker connection: Every 60s
- Account balance: Every 5min
- Disk space: Every 1h
- Internet connectivity: Every 30s

---

## 📞 Communication

**Code Reviews:** Via `.agents/PROPOSALS.md`  
**Bug Reports:** Create issue in `agents/dev/bugs.md`  
**System Status:** Update `.agents/STATE.md`

---

## 🎯 Current Focus

**Priority 1:** Research broker API options (MT5 vs OANDA vs IB)  
**Priority 2:** Create broker connector skeleton  
**Priority 3:** Setup error handling framework

**Blockers:**
- Need to choose broker platform
- API credentials not yet available

---

## 🔐 Security Best Practices

### API Credentials
```python
# .env file (NEVER commit to git)
BROKER_API_KEY=your_key_here
BROKER_API_SECRET=your_secret_here
BROKER_ACCOUNT_ID=12345678

# Load in code
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("BROKER_API_KEY")
```

### Gitignore Essentials
```
# Credentials
.env
credentials.yaml
*.key
*.pem

# API tokens
tokens/
secrets/
```

---

**Last Updated:** 2026-02-05 21:07 UTC  
**Status:** 🟢 READY
