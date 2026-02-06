"""
Download fresh data for multiple assets to test FTMO bot robustness
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def download_asset(symbol='BTC/USDT', timeframe='15m', days=60, filename='btc_fresh.csv'):
    """Download fresh market data"""
    print(f"📥 Downloading {symbol} {timeframe} (last {days} days)...")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    now = datetime.now()
    start_date = now - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)
    
    all_candles = []
    current_since = since
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            current_since = candles[-1][0] + 1
            
            if current_since >= int(now.timestamp() * 1000):
                break
            
            print(f"   {len(all_candles)} candles...", end='\r')
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            break
    
    print(f"\n✅ Downloaded {len(all_candles)} candles")
    
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    df.to_csv(f"data/{filename}", index=False)
    print(f"💾 Saved to data/{filename}\n")
    
    return df

if __name__ == "__main__":
    print("="*60)
    print("📡 DOWNLOADING FRESH VALIDATION DATA")
    print("="*60)
    print()
    
    # Download last 60 days (completely unseen by the model)
    download_asset('BTC/USDT', '15m', 60, 'btc_validation_60d.csv')
    download_asset('ETH/USDT', '15m', 60, 'eth_validation_60d.csv')
    download_asset('SOL/USDT', '15m', 60, 'sol_validation_60d.csv')
    
    print("="*60)
    print("✅ All validation data downloaded!")
    print("="*60)
