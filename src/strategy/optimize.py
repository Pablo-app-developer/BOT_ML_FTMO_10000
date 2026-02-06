import optuna
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import yfinance as yf
from src.strategy.ftmo_env import FTMOTradingEnv


def prepare_data_advanced():
    data_path = "data/btc_15m_advanced.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ Loaded cached data: {len(df)} rows")
        return df
        
    print("📥 Downloading advanced dataset (1 year, 15min candles)...")
    df = yf.download("BTC-USD", period="1y", interval="15m")
    
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index(drop=True)
    
    print(f"📊 Calculating Technical Indicators... ({len(df)} raw rows)")
    
    # Manual TA calculations (more reliable than library)
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Clean up
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"✅ Indicators calculated. Final dataset: {len(df)} rows")
    
    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    return df

def optimize_agent(trial):
    """ Optimization objective function for Optuna """
    
    # 1. Hyperparameters to search
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.000001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    
    # Network Architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_type == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    else:
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    # 2. Setup Environment (Train/Test Split)
    # Using a subset for faster optimization
    df = prepare_data_advanced()
    
    # Ensure we have enough data
    min_rows = 200  # window_size (60) + buffer (100) + margin
    if len(df) < min_rows:
        print(f"❌ Not enough data ({len(df)} rows). Need at least {min_rows}. Pruning trial.")
        return -10000
    
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    
    # Double-check train split has enough rows
    if len(df_train) < min_rows:
        print(f"❌ Train split too small ({len(df_train)} rows). Pruning trial.")
        return -10000
    
    env = DummyVecEnv([lambda: FTMOTradingEnv(df_train)])
    
    # 3. Create Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0
    )
    
    # 4. Train (shorter duration for optimization trials)
    # 50,000 steps is enough to see if it learns convergence
    try:
        model.learn(total_timesteps=50000)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -10000 # Prune bad trials

    # 5. Evaluate
    # FTMO Env returns reward based on PnL/Rules. 
    # High reward = Passed challenge / Profitable.
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    
    return mean_reward

def run_optimization():
    print("🚀 Starting Hyperparameter Optimization for FTMO Bot...")
    study = optuna.create_study(direction="maximize")
    
    # Run optimization (e.g., 20 trials, can go all day)
    # User said "all day", so let's do more trials or loops
    study.optimize(optimize_agent, n_trials=30) 
    
    print("\n✅ Optimization Complete!")
    print(f"Best Trial: {study.best_trial.value}")
    print(f"Best Params: {study.best_trial.params}")
    
    return study.best_trial.params

def train_best_model(best_params):
    print("\n🏆 Training Final 'Best' Model with optimized parameters...")
    df = prepare_data_advanced() # Use full data
    env = DummyVecEnv([lambda: FTMOTradingEnv(df)])
    
    # Construct net_arch from string choice if needed, but we pass kwarg dict manually
    net_arch_type = best_params.pop("net_arch")
    if net_arch_type == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    else:
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        
    model = PPO(
        "MlpPolicy",
        env,
        **best_params,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=1,
        tensorboard_log="./tensorboard_logs/BEST_FTMO"
    )
    
    # Massive training run
    model.learn(total_timesteps=1000000) # 1 Million steps
    
    model.save("models/best_ftmo_agent")
    print("🎉 Bot is ready. Saved to models/best_ftmo_agent.zip")

if __name__ == "__main__":
    best_params = run_optimization()
    train_best_model(best_params)
