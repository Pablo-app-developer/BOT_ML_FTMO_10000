"""
Data downloader using CCXT (Binance) for comprehensive historical data.
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def download_binance_data(symbol='BTC/USDT', timeframe='15m', days_back=365):
    """
    Download historical OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle interval ('5m', '15m', '1h', etc.)
        days_back: How many days of history to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"📡 Connecting to Binance...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Calculate timestamp range
    now = datetime.now()
    start_date = now - timedelta(days=days_back)
    since = int(start_date.timestamp() * 1000)  # CCXT uses milliseconds
    
    print(f"📥 Downloading {symbol} {timeframe} candles...")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
    
    all_candles = []
    current_since = since
    
    while True:
        try:
            # Fetch batch (Binance limit is 1000 candles per request)
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Update timestamp for next batch
            current_since = candles[-1][0] + 1
            
            # If we've reached current time, stop
            if current_since >= int(now.timestamp() * 1000):
                break
            
            print(f"   Downloaded {len(all_candles)} candles...", end='\r')
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
            
        except Exception as e:
            print(f"\n⚠️ Error downloading batch: {e}")
            break
    
    print(f"\n✅ Download complete! Total candles: {len(all_candles)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    return df

if __name__ == "__main__":
    # Test download
    df = download_binance_data('BTC/USDT', '15m', days_back=180)  # 6 months
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    
    # Save
    df.to_csv("data/btc_binance_15m.csv", index=False)
    print("💾 Saved to data/btc_binance_15m.csv")
