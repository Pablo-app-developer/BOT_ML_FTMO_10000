"""
EXPERT DATA GENERATOR (Sniper Strategy)
Runs the Sniper Strategy on historical data to create a dataset for Imitation Learning.
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
from stable_baselines3.common.buffers import DictRolloutBuffer # Not needed for BC directly, usually separate format

# Fix paths
project_root = os.path.join(os.getcwd())
sys.path.append(project_root)

from agents.quant.strategies.sniper_strategy import SniperStrategy
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2 # To check observation structure

DATA_FILE = "shared/data/processed_ETHUSDT_15m.csv"
OUTPUT_FILE = "shared/data/expert_sniper_eth.npz"

def generate_expert_trajectories():
    print("🎓 PRODUCING EXPERT DEMONSTRATIONS (Sniper Strategy)...")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        return
        
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    # Only use recent relevant data (last 2 years) to keep it focused
    df = df.iloc[-50000:].reset_index(drop=True) 

    # 2. Init Environment & Strategy
    # We need the Env to generate correct Observations for the AI
    env = FTMOTradingEnvV2(df)
    expert = SniperStrategy()
    
    # Storage
    observations = []
    actions = []
    rewards = []
    episode_starts = []
    
    obs, _ = env.reset()
    done = False
    
    # Variables for Strategy State
    balance = env.initial_balance
    shares = 0
    entry_price = 0
    highest_price = 0
    
    # PRE-CALCULATE Indicators Vectorized (Massive Speedup)
    # 1. Breakout Level
    df['Roll_Max'] = df['High'].rolling(window=expert.params['breakout_period']).max().shift(1)
    # 2. EMA
    df['EMA'] = df['Close'].ewm(span=expert.params['ema_period'], adjust=False).mean()
    
    print(f"🚀 Running Expert on {len(df)} steps (Optimized)...")
    
    step_count = 0
    expert_trades = 0
    
    while not done:
        # Get Current Index
        current_idx = env.current_step
        
        # Expert needs at least 'breakout_period' history
        if current_idx < expert.params['breakout_period'] + 50:
             action = 0
        else:
            # FAST ACCESS (O(1)) instead of O(N) re-calc
            current_price = df['Close'].iloc[current_idx]
            breakout_level = df['Roll_Max'].iloc[current_idx]
            ema_value = df['EMA'].iloc[current_idx]
            
            # Default: HOLD
            action = 0 
            
            if env.shares_held == 0:
                # ENTRY LOGIC
                # Buy if Price > Breakout AND Price > EMA
                if current_price > breakout_level and current_price > ema_value:
                    action = 1 # BUY
                    expert_trades += 1
            else:
                # EXIT LOGIC (Replicated from Sniper Strategy)
                # Sync state
                entry_price = env.entry_price
                if current_price > highest_price: highest_price = current_price
                
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check Hard Stop (-1.72%)
                if pnl_pct <= -0.0172:
                    action = 2 # Sell signal
                # Check Trailing (+0.96 trigger, 0.8 dist)
                elif pnl_pct >= 0.0096:
                    drop = (current_price - highest_price) / highest_price
                    if drop <= -0.008:
                        action = 2 # Sell signal

        # Record Transition
        observations.append(obs)
        actions.append(action)
        episode_starts.append(done)
        
        # Step Env
        # Note: If action=1 (Buy), env updates entry_price. If action=2 (Sell), env resets shares_held.
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Update internal tracking for next step logic
        if action == 1: # Buy (Env executed buy)
            highest_price = current_price
            # Important: Env entry price is set inside step based on current_price
        elif action == 2 or (env.shares_held == 0 and shares > 0): # Sold (Env executed sell)
            highest_price = 0
            
        shares = env.shares_held # Sync
        
        step_count += 1
        if step_count % 5000 == 0:
            print(f"   Step {step_count}: {expert_trades} expert trades so far...")

    # Save to .npz
    print(f"✅ Finished! Saving {len(observations)} transitions.")
    print(f"📊 Total Expert Trades: {expert_trades}")
    
    # Convert to numpy
    obs_np = np.array(observations).astype(np.float32)
    actions_np = np.array(actions).astype(np.int32) # Discrete actions
    # Reshape actions for SB3 imitation (N, 1) or (N,) depending on algo. usually (N,) for discrete.
    
    # Save dictionary
    np.savez_compressed(
        OUTPUT_FILE,
        observations=obs_np,
        actions=actions_np,
        rewards=np.array(rewards),
        episode_starts=np.array(episode_starts)
    )
    print(f"💾 Saved expert dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_expert_trajectories()
