"""
QA VALIDATOR V4
Simulates a full MONTH of trading to verify FTMO compliance using V4 Model.
"""
import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime

# Fix paths
sys.path.insert(0, os.getcwd())
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2

MODEL_PATH = "shared/models/ftmo_crypto_v4_gpu_best.zip" # Test the BEST model
DATA_DIR = "shared/data"

def load_test_data():
    """Load fresh data (last 20%)"""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("processed_")]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    
    # Last 20% strictly
    split = int(len(combined) * 0.8)
    return combined.iloc[split:]

def validate():
    print("============== QA VALIDATION REPORT ==============")
    print(f"Model: {MODEL_PATH}")
    print(f"Date: {datetime.now()}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Training might still be running.")
        return

    # Load Model
    model = PPO.load(MODEL_PATH)
    
    # Load Environment
    df_test = load_test_data()
    env = FTMOTradingEnvV2(df_test)
    
    obs, _ = env.reset()
    done = False
    
    balance_history = [env.initial_balance]
    equity_history = []
    
    print("\n🚀 Running simulation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        balance_history.append(info['net_worth'])
        equity_history.append(info['net_worth']) # Approx same for this env
        
    # Analysis
    final_balance = balance_history[-1]
    profit = final_balance - 10000
    profit_pct = (profit / 10000) * 100
    
    balances = np.array(balance_history)
    max_balance = np.maximum.accumulate(balances)
    drawdowns = (max_balance - balances) / max_balance
    max_dd = drawdowns.max() * 100
    
    print("\n---------------- RESULTS ----------------")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Profit:  ${profit:.2f} ({profit_pct:.2f}%)")
    print(f"Max Drawdown:  {max_dd:.2f}%")
    print(f"Total Trades:  {info['total_trades']}")
    print(f"Win Rate:      {info['win_rate']*100:.1f}%")
    
    print("\n---------------- VERDICT ----------------")
    
    failed = False
    if max_dd >= 10.0:
        print("❌ FAILED: Max Total Loss Limit Hit (10%)")
        failed = True
    elif profit_pct < 10.0:
        print("⚠️ WARNING: Profit Target (10%) Not Met")
        # Not a fail, just not passed yet
    else:
        print("✅ PASSED: Profit Target Met!")
        
    if not failed and max_dd < 5.0 and profit_pct > 0:
        print("🏆 EXCELLENT: Low Risk (<5% DD) and Profitable")
        
    print("==================================================")

if __name__ == "__main__":
    validate()
