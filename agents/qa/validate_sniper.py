"""
VALIDATE SNIPER STRATEGY
Tests the Heuristic Breakout Strategy on fresh Binance Data.
"""
import pandas as pd
import numpy as np
import os
import sys

# Fix paths
project_root = os.path.join(os.getcwd())
sys.path.append(project_root)

from agents.quant.strategies.sniper_strategy import SniperStrategy

DATA_FILE = "shared/data/processed_ETHUSDT_15m.csv"

def run_test():
    print("🧪 STARTING SNIPER STRATEGY TEST (ETH)...")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        return
        
    print("📚 Loading Market Data...")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Use last 6 months
    start_idx = len(df) - (4 * 24 * 30 * 6) 
    df_test = df.iloc[start_idx:].copy()
    print(f"📅 Testing Period: {df_test.index[0]} to {df_test.index[-1]}")
    
    # 2. Init Strategy
    brain = SniperStrategy()

    # 3. Simulation Loop
    balance = 10000.0
    shares = 0.0
    entry_price = 0.0
    highest_price = 0.0
    
    trades = []
    
    print("🚀 Running Simulation...")
    
    # Iteration
    # Start after need period
    start_loop = brain.params['breakout_period'] + 10
    
    for i in range(start_loop, len(df_test)):
        # Data Slice
        df_slice = df_test.iloc[:i+1] # Pass full context growing
        # Optimization: In real backtest we optimize, here we rely on basic slicing (kinda slow but ok for logic check)
        
        current_price = df_test['Close'].iloc[i]
        timestamp = df_test.index[i]
        
        # Get Signal
        if shares == 0:
            signal, meta = brain.analyze(df_slice)
            
            if signal == 1:
                # BUY
                invest_size = balance * 0.5 # 50% equity
                shares = invest_size / current_price
                entry_price = current_price
                highest_price = current_price
                balance -= invest_size
                # print(f"[{timestamp}] 🟢 BUY @ {current_price:.2f} | {meta['reason']}")
        
        else:
            # MANAGE TRADE (Sniper Logic Transplanted)
            # Update Peak
            if current_price > highest_price:
                highest_price = current_price
                
            pct_change = (current_price - entry_price) / entry_price
            
            # Hard Stop (-1.72% from repo params)
            if pct_change <= -0.0172:
                revenue = shares * current_price * (1 - 0.001)
                balance += revenue
                pnl = revenue - (shares * entry_price)
                trades.append(pnl)
                shares = 0
                # print(f"[{timestamp}] 🛡️ HARD STOP @ {current_price:.2f} | PnL: {pnl:.2f}")
                
            # Trailing Stop
            # Trigger: 0.96% profit
            # Dist: 0.8%
            elif pct_change >= 0.0096:
                drop_from_peak = (current_price - highest_price) / highest_price
                if drop_from_peak <= -0.008:
                    revenue = shares * current_price * (1 - 0.001)
                    balance += revenue
                    pnl = revenue - (shares * entry_price)
                    trades.append(pnl)
                    shares = 0
                    # print(f"[{timestamp}] 🎯 TRAILING STOP @ {current_price:.2f} | PnL: {pnl:.2f}")

    # Final Liquidation
    if shares > 0:
        balance += shares * df_test['Close'].iloc[-1]
    
    # Report
    print("\n============== RESULTADOS SNIPER =============")
    print(f"Final Balance: ${balance:.2f}")
    profit_total = balance - 10000.0
    print(f"Profit/Loss:   ${profit_total:.2f} ({(profit_total/10000)*100:.2f}%)")
    print(f"Total Trades:  {len(trades)}")
    
    if len(trades) > 0:
        wins = [t for t in trades if t > 0]
        win_rate = len(wins) / len(trades)
        print(f"Win Rate:      {win_rate*100:.1f}%")
    
    if profit_total > 0:
        print("✅ SUCCESS: Sniper works!")
    else:
        print("⚠️ CAUTION: Sniper lost money.")

if __name__ == "__main__":
    run_test()
