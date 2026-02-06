"""
FTMO Training System V2 - Anti-Overfitting Edition
- Multi-asset training
- Early stopping
- Better validation
- Fewer steps, better quality
"""
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
import os
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2

def add_indicators(df):
    """Add technical indicators"""
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

def combine_multi_asset_data():
    """Combine data from multiple assets for diverse training"""
    print("📊 Loading and combining multi-asset data...")
    
    datafiles = [
        ('data/btc_binance_15m.csv', 'BTC'),
        ('data/eth_validation_60d.csv', 'ETH'),
        ('data/sol_validation_60d.csv', 'SOL')
    ]
    
    all_data = []
    
    for filepath, name in datafiles:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df = add_indicators(df)
            all_data.append(df)
            print(f"   {name}: {len(df)} candles")
        else:
            print(f"   ⚠️ {name} not found, skipping")
    
    if not all_data:
        raise ValueError("No data files found!")
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle to mix assets (prevents sequential learning)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Combined dataset: {len(combined_df)} total candles")
    return combined_df

def train_ftmo_v2():
    """
    V2 Training: Multi-asset, early stopping, better validation
    """
    print("="*60)
    print("🚀 FTMO TRAINING V2 - ANTI-OVERFITTING")
    print("="*60)
    
    # 1. Load multi-asset data
    df_all = combine_multi_asset_data()
    
    # 2. Split: 70% train, 15% validation, 15% test
    train_end = int(len(df_all) * 0.70)
    val_end = int(len(df_all) * 0.85)
    
    df_train = df_all.iloc[:train_end].reset_index(drop=True)
    df_val = df_all.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df_all.iloc[val_end:].reset_index(drop=True)
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(df_train):,} candles (70%)")
    print(f"   Validation: {len(df_val):,} candles (15%)")
    print(f"   Test:       {len(df_test):,} candles (15%)")
    
    # 3. Create environments
    print("\n🔧 Creating environments...")
    env_train = DummyVecEnv([lambda: FTMOTradingEnvV2(df_train)])
    env_val = DummyVecEnv([lambda: FTMOTradingEnvV2(df_val)])
    
    # 4. Model with regularization
    print("\n🧠 Initializing PPO with regularization...")
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=0.0001,  # Lower LR for stability
        n_steps=2048,
        batch_size=64,  # Smaller batch for better generalization
        gamma=0.99,
        ent_coef=0.02,  # Higher entropy = more exploration
        clip_range=0.2,
        max_grad_norm=0.5,  # Gradient clipping for stability
        policy_kwargs=dict(
            net_arch=[128, 128],  # Smaller network = less overfitting
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/FTMO_V2",
        device="cpu"
    )
    
    # 5. Callbacks with early stopping
    models_dir = "models/FTMO_V2"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Stop if no improvement for 10 evaluations
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=models_dir,
        log_path="./logs/",
        eval_freq=5000,  # Evaluate every 5k steps
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback  # Attach early stopping
    )
    
    # 6. Train (max 800k steps, but will stop early if not improving)
    print(f"\n🚀 Training (max 800k steps with early stopping)...")
    print("   Will stop automatically if validation stops improving")
    print("\n" + "="*60)
    
    try:
        model.learn(
            total_timesteps=800000,  # Less than before, but with early stopping
            callback=eval_callback,
            progress_bar=True
        )
        
        print("\n✅ Training completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # 7. Final test on held-out data
    print("\n" + "="*60)
    print("🧪 FINAL TEST ON HELD-OUT DATA")
    print("="*60)
    
    env_test = DummyVecEnv([lambda: FTMOTradingEnvV2(df_test)])
    best_model = PPO.load(os.path.join(models_dir, "best_model"))
    
    mean_reward, std_reward = evaluate_policy(best_model, env_test, n_eval_episodes=10)
    print(f"\nTest Set Performance: {mean_reward:.2f} ± {std_reward:.2f}")
    
    if mean_reward > 0:
        print("✅ Positive performance on unseen data!")
    else:
        print("⚠️ Needs more tuning")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE - Model saved to models/FTMO_V2/")
    print("="*60)

if __name__ == "__main__":
    train_ftmo_v2()
