"""
MAIN ORCHESTRATOR - Entry point for Dockerized Bot
Coordinates the 4 Agents logic in production.
"""
import asyncio
import os
import sys
import logging
from agents.ml.deepseek_assistant import DeepSeekAssistant
import yaml

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
    def __init__(self):
        self.risk_config = self.load_risk_config()
        self.running = False
        
        # Initialize DeepSeek Assistant
        try:
            self.assistant = DeepSeekAssistant()
            logger.info("✅ DeepSeek Assistant connected")
        except Exception as e:
            logger.error(f"⚠️ DeepSeek unavailable: {e}")
            self.assistant = None

    def load_risk_config(self):
        """Load settings from Quant Agent"""
        path = "agents/quant/risk_parameters.yaml"
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info("✅ Risk parameters loaded")
                return config
        else:
            logger.critical("❌ Risk parameters missing! Halting.")
            sys.exit(1)

    async def health_check(self):
        """Periodic system health check (QA Agent Logic)"""
        while self.running:
            # 1. Check Internet/Exchange connection logic here
            # 2. Check Disk Space
            # 3. Check Memory usage
            
            # 4. Check Daily Drawdown (Simulated for production)
            # In real production, this would call brokerage API info
            # current_equity = await self.broker.get_equity()
            # if current_equity < self.daily_start_equity * (1 - self.risk_config['ftmo_limits']['max_daily_loss_percent']):
            #     await self.emergency_close_all("Daily Drawdown Limit Hit!")
            
            logger.info("💓 System Pulse Code: Green")
            await asyncio.sleep(60)

    async def emergency_close_all(self, reason):
        """Close ALL positions immediately and halt"""
        logger.critical(f"🚨 EMERGENCY HALT TRIGGERED: {reason}")
        # await self.broker.close_all_positions()
        # await self.assistant.alert(f"Bot Halted: {reason}")
        self.running = False
        sys.exit(1)

    async def trading_loop(self):
        """Main automated trading loop"""
        logger.info("🚀 Starting Trading Loop...")
        
        while self.running:
            try:
                # 1. Fetch Market Data (Dev Agent)
                # 2. Get Signal from ML Model (ML Agent)
                # 3. Validate Signal against Risk Pars (Quant Agent)
                # 4. Execute (Dev Agent)
                
                await asyncio.sleep(1) # wait 1s
                
            except Exception as e:
                logger.error(f"💥 Critical Loop Error: {e}")
                if self.assistant:
                    # Ask DeepSeek for advice on the error
                    advice = self.assistant.debug_code(str(e), "trading_loop_main")
                    logger.info(f"🤖 DeepSeek Advice: {advice}")
                await asyncio.sleep(10)

    async def start(self):
        self.running = True
        logger.info(f"🏁 System Start. Target: FTMO Challenge (Crypto Vehicle)")
        logger.info(f"🛡️  Max Daily Loss: {self.risk_config['ftmo_limits']['max_daily_loss_percent']*100}%")
        
        # Run tasks concurrently
        await asyncio.gather(
            self.health_check(),
            self.trading_loop()
        )

if __name__ == "__main__":
    bot = BotOrchestrator()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
