"""
Quick FTMO Bot Evaluation
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.strategy.ftmo_env import FTMOTradingEnv

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
print("🔍 EVALUANDO BOT FTMO")
print("="*60)

# Load
model = PPO.load("models/FTMO_CHAMPION/best_model.zip")
df = pd.read_csv("data/btc_binance_15m.csv")
df = add_indicators(df)

# Test data (last 20%)
split = int(len(df) * 0.80)
df_test = df.iloc[split:].reset_index(drop=True)
print(f"Datos de prueba: {len(df_test)} velas\n")

results = []
for ep in range(20):
    env = FTMOTradingEnv(df_test)
    obs, _ = env.reset()
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    final = info['net_worth']
    profit = ((final - 10000) / 10000) * 100
    passed = final >= 11000 and info['daily_drawdown'] < 0.05
    
    results.append({
        'profit': profit,
        'final': final,
        'passed': passed
    })
    
    print(f"Ep {ep+1:2d}: ${final:,.0f} ({profit:+.2f}%) {'✅' if passed else '❌'}")

print("\n" + "="*60)
df_r = pd.DataFrame(results)
pass_rate = (df_r['passed'].sum() / len(df_r)) * 100
avg_profit = df_r['profit'].mean()

print(f"TASA DE ÉXITO FTMO: {pass_rate:.0f}%")
print(f"GANANCIA PROMEDIO: {avg_profit:+.2f}%")
print(f"MEJOR: {df_r['profit'].max():+.2f}%")
print(f"PEOR: {df_r['profit'].min():+.2f}%")
print("="*60)

if pass_rate >= 70:
    print("🏆 EXCELENTE! Listo para FTMO real")
elif pass_rate >= 50:
    print("👍 BUENO - Tiene potencial")
elif pass_rate >= 30:
    print("⚠️ NECESITA MÁS ENTRENAMIENTO")
else:
    print("❌ REQUIERE CAMBIOS MAYORES")
