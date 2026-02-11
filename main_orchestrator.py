"""
MAIN ORCHESTRATOR - PRODUCTION (Sniper Strategy)
Runs the FTMO Bot using the Volatility Breakout Strategy.
"""
import asyncio
import os
import sys
import logging
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime
import yaml

# Add path
sys.path.append(os.getcwd())
from agents.quant.strategies.sniper_strategy import SniperStrategy
from agents.ml.deepseek_assistant import DeepSeekAssistant

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ORCHESTRATOR")

class BotOrchestrator:
    def __init__(self, symbol='ETH/USDT'):
        self.symbol = symbol
        self.running = False
        self.strategy = SniperStrategy()
        self.risk_config = self.load_risk_config()
        
        # State
        self.balance = 10000.0 # Paper Trading Balance
        self.position = None # None or {'shares': float, 'entry': float, 'highest': float}
        
        # Exchange (Public Data Only)
        self.exchange = ccxt.binanceus({'enableRateLimit': True}) 
        # Note: Using binanceus for public data as it's often more accessible from US IPs, 
        # but for real trading we'd need the specific user exchange.
        
    def load_risk_config(self):
        path = "agents/quant/risk_parameters.yaml"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {'ftmo_limits': {'max_daily_loss_percent': 0.05}}

    async def fetch_live_data(self):
        """Fetches last 100 candles"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '15m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Data Fetch Error: {e}")
            return None

    async def execute_trade(self, signal, current_price, reason):
        if signal == 1 and self.position is None:
            # BUY
            risk_pct = 0.01 # 1% risk
            # Simple sizing: Invest 50% of balance (Leverage 1)
            invest = self.balance * 0.5
            shares = invest / current_price
            
            self.position = {
                'shares': shares, 
                'entry': current_price, 
                'highest': current_price
            }
            self.balance -= invest
            logger.info(f"🟢 OPEN LONG: {self.symbol} @ {current_price} | Reason: {reason}")
            
    async def manage_position(self, current_price):
        if self.position is None: return
        
        # Update Highest for Trailing
        if current_price > self.position['highest']:
            self.position['highest'] = current_price
            
        entry = self.position['entry']
        highest = self.position['highest']
        pnl_pct = (current_price - entry) / entry
        
        # 1. Hard Stop (-1.72%)
        if pnl_pct <= -0.0172:
            await self.close_position(current_price, "Hard Stop Loss")
            return

        # 2. Trailing Stop
        # Trigger: +0.96%
        if pnl_pct >= 0.0096:
            drop = (current_price - highest) / highest
            # Dist: 0.8%
            if drop <= -0.008:
                await self.close_position(current_price, "Trailing Stop (Profit Locked)")
                
    async def close_position(self, price, reason):
        shares = self.position['shares']
        revenue = shares * price * (1 - 0.001) # Comm
        self.balance += revenue
        pnl = revenue - (shares * self.position['entry'])
        
        logger.info(f"🔴 CLOSE POSITION @ {price} | PnL: ${pnl:.2f} | Reason: {reason}")
        logger.info(f"💰 New Balance: ${self.balance:.2f}")
        self.position = None

    async def start(self):
        self.running = True
        logger.info(f"🚀 BOT STARTED. Strategy: Sniper. Asset: {self.symbol}")
        
        while self.running:
            try:
                # 1. Get Data
                df = await self.fetch_live_data()
                if df is not None:
                    current_price = df['Close'].iloc[-1]
                    
                    # 2. Manage Existing
                    if self.position:
                        await self.manage_position(current_price)
                    
                    # 3. Look for Entries
                    else:
                        signal, meta = self.strategy.analyze(df)
                        if signal == 1:
                            await self.execute_trade(signal, current_price, meta['reason'])
                    
                    # Log heartbeat every hour (approx)
                    # logger.info(f"💓 Price: {current_price}")
                    
                await asyncio.sleep(60 * 15) # Wait 15 mins (Candle close)
                # In production we might check more often for stops, but for candle strategy 15m is okay
                
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                await asyncio.sleep(60)
        
        await self.exchange.close()

if __name__ == "__main__":
    bot = BotOrchestrator()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped")
