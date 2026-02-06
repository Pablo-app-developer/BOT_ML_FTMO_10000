"""
Test del modelo V3 (Imitation Learning) recién entrenado
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2

def add_indicators(df):
    """Add TA indicators"""
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
print("🧪 TESTING V3 MODEL (IMITATION LEARNING)")
print("="*60)

# Load V3 model
if not os.path.exists("models/FTMO_IMITATION/ftmo_bot.zip"):
    print("❌ Model not found! Please train first with: python train_imitation.py")
    exit(1)

model = PPO.load("models/FTMO_IMITATION/ftmo_bot.zip")
print("✅ V3 Model loaded\n")

# Test on all fresh assets
datasets = [
    ('data/btc_validation_60d.csv', 'BTC Fresh 60d'),
    ('data/eth_validation_60d.csv', 'ETH Fresh 60d'),
    ('data/sol_validation_60d.csv', 'SOL Fresh 60d')
]

all_results = []

for data_file, asset_name in datasets:
    if not os.path.exists(data_file):
        print(f"⚠️ {data_file} not found, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"🔍 Testing on: {asset_name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(data_file)
    df = add_indicators(df)
    print(f"Data: {len(df)} candles\n")
    
    asset_results = []
    
    for ep in range(10):
        env = FTMOTradingEnvV2(df)
        obs, _ = env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        final = info['net_worth']
        profit = ((final - 10000) / 10000) * 100
        passed = final >= 11000 and info['daily_drawdown'] < 0.05
        
        asset_results.append({
            'asset': asset_name,
            'profit': profit,
            'final': final,
            'passed': passed,
            'trades': info.get('total_trades', 0)
        })
        
        status = "✅" if passed else "❌"
        print(f"Ep {ep+1:2d}: ${final:,.0f} ({profit:+.2f}%) {status} | Trades: {info.get('total_trades', 0)}")
    
    # Asset summary
    avg_profit = np.mean([r['profit'] for r in asset_results])
    pass_rate = sum([r['passed'] for r in asset_results]) / len(asset_results) * 100
    
    print(f"\n{asset_name} Summary:")
    print(f"   Pass Rate: {pass_rate:.0f}%")
    print(f"   Avg Profit: {avg_profit:+.2f}%")
    
    all_results.extend(asset_results)

# Overall summary
print("\n" + "="*60)
print("📊 OVERALL V3 RESULTS")
print("="*60)

df_results = pd.DataFrame(all_results)
overall_pass_rate = df_results['passed'].sum() / len(df_results) * 100
overall_avg_profit = df_results['profit'].mean()

by_asset = df_results.groupby('asset').agg({
    'passed': 'sum',
    'profit': 'mean'
})

for asset in by_asset.index:
    passed = by_asset.loc[asset, 'passed']
    avg_pnl = by_asset.loc[asset, 'profit']
    print(f"{asset:20s}: {int(passed)}/10 passed ({passed*10:.0f}%) | Avg: {avg_pnl:+.2f}%")

print(f"\n{'OVERALL':20s}: {int(df_results['passed'].sum())}/{len(df_results)} ({overall_pass_rate:.0f}%) | Avg: {overall_avg_profit:+.2f}%")
print("="*60)

if overall_pass_rate >= 70:
    print("🏆 EXCELENTE! Bot está listo para FTMO")
elif overall_pass_rate >= 50:
    print("👍 BUENO - Tiene buen potencial")
elif overall_pass_rate >= 30:
    print("⚠️ REGULAR - Necesita ajustes")
else:
    print("❌ INSUFICIENTE - Requiere más trabajo")

print("="*60)
