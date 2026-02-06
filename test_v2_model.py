"""
Quick test of V2 model
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2

def add_indicators(df):
    """Quick TA"""
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
    
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df

print("="*60)
print("🔍 TESTING V2 MODEL")
print("="*60)

# Load V2 model
model = PPO.load("models/FTMO_V2/best_model.zip")
print("✅ V2 Model loaded\n")

# Test on fresh BTC data
df = pd.read_csv("data/btc_validation_60d.csv")
df = add_indicators(df)
print(f"Test data: {len(df)} candles (fresh BTC)\n")

# Run 5 episodes
results = []
for ep in range(5):
    env = FTMOTradingEnvV2(df)
    obs, _ = env.reset()
    
    done = False
    steps = 0
    actions_taken = {'hold': 0, 'buy': 0, 'sell': 0}
    
    while not done and steps < 5000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if action == 0:
            actions_taken['hold'] += 1
        elif action == 1:
            actions_taken['buy'] += 1
        elif action == 2:
            actions_taken['sell'] += 1
        
        steps += 1
    
    final = info['net_worth']
    profit = ((final - 10000) / 10000) * 100
    
    results.append({
        'final': final,
        'profit': profit,
        'trades': info['total_trades'],
        'win_rate': info.get('win_rate', 0) * 100,
        'steps': steps,
        'actions': actions_taken
    })
    
    print(f"Ep {ep+1}: ${final:,.0f} ({profit:+.2f}%) | Trades: {info['total_trades']} | Steps: {steps}")
    print(f"       Actions: Hold={actions_taken['hold']}, Buy={actions_taken['buy']}, Sell={actions_taken['sell']}")

print("\n" + "="*60)
avg_profit = np.mean([r['profit'] for r in results])
avg_trades = np.mean([r['trades'] for r in results])
print(f"Avg Profit: {avg_profit:+.2f}%")
print(f"Avg Trades: {avg_trades:.1f}")

# Check if the model is just holding
all_holds = sum([r['actions']['hold'] for r in results])
all_actions = sum([r['steps'] for r in results])
hold_pct = (all_holds / all_actions) * 100

print(f"Hold Percentage: {hold_pct:.1f}%")

if hold_pct > 90:
    print("\n⚠️ PROBLEM: Model learned to just HOLD (do nothing)")
    print("   This is a common failure mode - needs better reward shaping")
elif avg_profit < -5:
    print("\n⚠️ PROBLEM: Model is losing money consistently")
else:
    print("\n✅ Model is at least attempting to trade")

print("="*60)
