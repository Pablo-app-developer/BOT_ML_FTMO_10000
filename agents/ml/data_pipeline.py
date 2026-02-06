"""
Data Pipeline - ML Agent
Downloads, cleans, and engineers features for Crypto-FTMO training.
"""
import ccxt
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime, timedelta
import os

# Config
EXCHANGE_ID = 'binance'
TIMEFRAME = '15m'
PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
YEARS_BACK = 2
DATA_DIR = 'shared/data'

def download_historical_data(symbol, days=365):
    """Download OHLCV data from Binance via CCXT"""
    print(f"📥 Downloading {symbol} for last {days} days...")
    exchange = getattr(ccxt, EXCHANGE_ID)()
    
    # Calculate since timestamp
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    all_candles = []
    while since < exchange.milliseconds():
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=1000)
            if not candles:
                break
            
            since = candles[-1][0] + 1
            all_candles += candles
            print(f"   Fetched {len(candles)} candles... Last: {datetime.fromtimestamp(candles[-1][0]/1000)}")
            time.sleep(0.5) # Rate limit respect
            
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            break
            
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    return df

def feature_engineering(df):
    """Add technical indicators for ML model"""
    print("🛠️ Engineering features (RSI, MACD, ATR, Bollinger)...")
    
    # Trend
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    
    # Momentum
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    
    # Volatility (CRITICAL FOR QUANT AGENT)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Close'], window=20, window_dev=2)
    
    # Data Cleaning for FTMO Simulation
    # Drop NaN from indicator warmup
    df.dropna(inplace=True)
    
    return df

def run_pipeline():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for pair in PAIRS:
        # Download
        filename = f"{pair.replace('/', '')}_{TIMEFRAME}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"♻️  Found existing data for {pair}, checking freshness...")
            # TODO: Implement update logic
            df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        else:
            df = download_historical_data(pair, days=YEARS_BACK*365)
        
        # Engineer
        df_processed = feature_engineering(df)
        
        # Save
        save_path = os.path.join(DATA_DIR, f"processed_{filename}")
        df_processed.to_csv(save_path)
        print(f"✅ Saved processed data to {save_path} ({len(df_processed)} rows)\n")

if __name__ == "__main__":
    run_pipeline()
