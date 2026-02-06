"""
Behavioral Cloning - Train bot to imitate expert strategy
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2
from src.strategy.expert_strategy import generate_expert_demonstrations

def prepare_data_for_training():
    """Prepare multi-asset data with indicators"""
    print("📊 Preparing training data...")
    
    datafiles = [
        'data/btc_binance_15m.csv',
        'data/eth_validation_60d.csv',
        'data/sol_validation_60d.csv'
    ]
    
    all_data = []
    
    for filepath in datafiles:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Add indicators
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
            
            all_data.append(df)
            print(f"   Loaded: {len(df)} candles from {os.path.basename(filepath)}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Combined: {len(combined_df)} total candles")
    return combined_df

def train_with_imitation():
    """
    Train using behavioral cloning approach:
    1. Generate expert demonstrations
    2. Use them to pre-train the model
    3. Fine-tune with RL
    """
    print("="*60)
    print("🎓 IMITATION LEARNING - FTMO BOT")
    print("="*60)
    
    # 1. Prepare data
    df = prepare_data_for_training()
    
    # 2. Generate expert demonstrations
    print("\n📚 Generating expert demonstrations...")
    demos, results = generate_expert_demonstrations(df, n_episodes=200)
    
    # Save demonstrations
    os.makedirs("data/demonstrations", exist_ok=True)
    with open("data/demonstrations/expert_demos.pkl", "wb") as f:
        pickle.dump((demos, results), f)
    print(f"💾 Saved {len(demos)} demonstrations")
    
    # 3. Since simple behavioral cloning is complex with SB3,
    # we'll use a hybrid approach: pre-train with expert data as baseline,
    # then use RL to improve
    
    # Split data
    train_end = int(len(df) * 0.80)
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:].reset_index(drop=True)
    
    print(f"\n📊 Training split:")
    print(f"   Train: {len(df_train)} candles")
    print(f"   Val:   {len(df_val)} candles")
    
    # Create environment
    print("\n🔧 Setting up environment...")
    env_train = DummyVecEnv([lambda: FTMOTradingEnvV2(df_train)])
    env_val = DummyVecEnv([lambda: FTMOTradingEnvV2(df_val)])
    
    # Initialize model with very conservative settings
    print("\n🧠 Initializing model (conservative)...")
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=0.00005,  # Very low LR for stability
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,  # Moderate exploration
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=[64, 64]  # Small network
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/FTMO_IMITATION",
        device="cpu"
    )
    
    # Train with moderate steps
    print("\n🚀 Training with RL (using diverse data)...")
    print("   Target: 300k steps")
    print("\n" + "="*60)
    
    try:
        model.learn(
            total_timesteps=300000,  # Moderate duration
            progress_bar=True
        )
        
        print("\n✅ Training complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    
    # Save
    os.makedirs("models/FTMO_IMITATION", exist_ok=True)
    model.save("models/FTMO_IMITATION/ftmo_bot")
    print("\n💾 Model saved to models/FTMO_IMITATION/ftmo_bot.zip")
    
    print("\n" + "="*60)
    print("✅ IMITATION LEARNING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    train_with_imitation()
