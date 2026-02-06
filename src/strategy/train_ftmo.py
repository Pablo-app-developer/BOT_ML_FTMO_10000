"""
FTMO Challenge Training System - Professional Edition
Goal: Pass 10k FTMO Challenge WITHOUT losing money
Rules:
- Max Daily Loss: 5% ($500)
- Max Total Loss: 10% ($1000)  
- Profit Target: 10% ($1000)
"""
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
from src.strategy.ftmo_env import FTMOTradingEnv

def add_technical_indicators(df):
    """Add comprehensive TA indicators optimized for FTMO"""
    print("📊 Calculating Technical Indicators...")
    
    # Trend Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Momentum
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volatility (Bollinger Bands)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # ATR (Risk Management)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price Action
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
    
    # Clean
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"✅ Indicators ready. Final dataset: {len(df)} rows")
    return df

def train_ftmo_champion(data_path="data/btc_binance_15m.csv", total_steps=2000000):
    """
    Train the FTMO challenge bot with maximum steps for robustness.
    
    Args:
        data_path: Path to CSV data
        total_steps: Training duration (2M default = ~24 hours on CPU)
    """
    print("=" * 60)
    print("🏆 FTMO CHALLENGE TRAINING - PROFESSIONAL SYSTEM")
    print("=" * 60)
    
    # 1. Load & Prepare Data
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Run: python src/utils/download_data.py")
        return
    
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Raw data: {len(df)} rows")
    
    # Add indicators
    df = add_technical_indicators(df)
    
    # Train/Val split (80/20)
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(df_train)} candles")
    print(f"   Validation: {len(df_val)} candles")
    
    # 2. Create Environments
    print("\n🔧 Setting up FTMO environments...")
    env_train = DummyVecEnv([lambda: FTMOTradingEnv(df_train)])
    env_val = DummyVecEnv([lambda: FTMOTradingEnv(df_val)])
    
    # 3. Model Configuration (Conservative for FTMO)
    print("\n🧠 Initializing PPO Agent (Conservative Settings)...")
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=0.00005,  # Lower LR for stability
        n_steps=2048,
        batch_size=128,
        gamma=0.995,  # High gamma = long-term thinking
        ent_coef=0.01,  # Moderate exploration
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256, 128]),  # Large network
        verbose=1,
        tensorboard_log="./tensorboard_logs/FTMO_CHAMPION",
        device="cpu"  # Change to "cuda" if you have GPU
    )
    
    # 4. Callbacks
    models_dir = "models/FTMO_CHAMPION"
    os.makedirs(models_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="ftmo_checkpoint"
    )
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=models_dir,
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # 5. Train
    print(f"\n🚀 Starting Training: {total_steps:,} steps")
    print("   (This will take several hours. You can monitor with TensorBoard)")
    print("   Command: tensorboard --logdir ./tensorboard_logs/FTMO_CHAMPION")
    print("\n" + "=" * 60)
    
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # 6. Final Save
        final_path = os.path.join(models_dir, "ftmo_final_model")
        model.save(final_path)
        print(f"\n💾 Model saved: {final_path}.zip")
        
        # 7. Final Evaluation
        print("\n📈 Final Evaluation on Validation Set...")
        mean_reward, std_reward = evaluate_policy(model, env_val, n_eval_episodes=10)
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        if mean_reward > 0:
            print("\n🎉 SUCCESS! Bot shows positive performance!")
        else:
            print("\n⚠️ Bot needs more training or parameter tuning.")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        model.save(os.path.join(models_dir, "ftmo_interrupted"))
        print("💾 Model saved at interruption point.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    # Start the marathon training session
    train_ftmo_champion(
        data_path="data/btc_binance_15m.csv",
        total_steps=2000000  # 2 Million steps - adjust based on your time
    )
