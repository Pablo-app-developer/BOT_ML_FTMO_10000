"""
SNIPER STRATEGY (Volatility Breakout)
Adapted from 'sol_sniper_bot.py' for FTMO Framework.
"""
import pandas as pd
import numpy as np

class SniperStrategy:
    def __init__(self):
        # Optimized Params from external repo
        self.params = {
            "breakout_period": 35,
            "ema_period": 23
        }
        print(f"🔫 Sniper Strategy Loaded. Params: {self.params}")

    def analyze(self, df):
        """
        Analyzes the dataframe and returns a signal.
        Returns:
            signal (int): 0=Hold, 1=Buy, 2=Sell
            metadata (dict): Info about the decision
        """
        # Ensure enough data
        if len(df) < self.params['breakout_period'] + 5:
            return 0, {}

        df = df.copy()
        
        # 1. Calculate Indicators
        # Breakout Level: Max of last N candles (shifted 1 to avoid lookahead)
        df['Roll_Max'] = df['High'].rolling(window=self.params['breakout_period']).max().shift(1)
        
        # Trend Filter: EMA
        df['EMA'] = df['Close'].ewm(span=self.params['ema_period'], adjust=False).mean()
        
        # Get latest closed candle (assuming df includes current partial candle, we usually want the last COMPLETE one or current tick check)
        # For simulation/backtest consistency, let's look at the last row
        current_price = df['Close'].iloc[-1]
        breakout_level = df['Roll_Max'].iloc[-1]
        ema_value = df['EMA'].iloc[-1]
        
        signal = 0
        reason = ""
        
        # 2. Logic
        # BUY: Price breaks above local High AND Trend is Bullish (Price > EMA)
        if current_price > breakout_level and current_price > ema_value:
            signal = 1
            reason = f"Breakout ({current_price:.2f} > {breakout_level:.2f}) & Trend Up"
        
        # SELL: Strategy doesn't have explicit sell signal (uses Stop Loss / Trailing), 
        # but we could implement an EMA cross under if needed. 
        # For now, we rely on the Risk Manager for exits, so signal 0 (Hold) is fine until stop/trail hit.
        
        return signal, {
            "price": current_price,
            "breakout_level": breakout_level,
            "ema": ema_value,
            "reason": reason
        }
