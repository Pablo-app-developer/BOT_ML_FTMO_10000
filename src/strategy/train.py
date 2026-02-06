import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf
from src.strategy.ftmo_env import FTMOTradingEnv

def download_data(symbol="BTC-USD", period="60d", interval="15m"):
    print(f"Downloading data for {symbol}...")
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        raise ValueError("No data downloaded!")
    
    # Flatten MultiIndex columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # Calculate indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

def train_model():
    # 1. Prepare Data
    data_path = "data/btc_15m.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = download_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
    
    print(f"Data ready: {len(df)} rows.")

    # 2. Create Environment
    env = DummyVecEnv([lambda: FTMOTradingEnv(df)])

    # 3. Define Model (PPO)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

    # 4. Train
    print("Starting training...")
    model.learn(total_timesteps=100000)
    
    # 5. Save
    os.makedirs("models", exist_ok=True)
    model.save("models/ftmo_ppo_model")
    print("Model saved to models/ftmo_ppo_model.zip")

if __name__ == "__main__":
    train_model()
