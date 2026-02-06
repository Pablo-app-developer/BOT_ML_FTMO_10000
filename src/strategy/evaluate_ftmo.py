"""
FTMO Bot Backtest & Evaluation
Analyze how the trained bot performs on unseen data
"""
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.strategy.ftmo_env import FTMOTradingEnv
import matplotlib.pyplot as plt
import os

def add_technical_indicators(df):
    """Copy from train_ftmo.py"""
    print("📊 Calculating Technical Indicators...")
    
    # Trend
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
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # ATR
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
    
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def backtest_ftmo_bot(model_path, data_path, n_episodes=10):
    """
    Run comprehensive backtest of FTMO bot
    """
    print("="*60)
    print("🔍 FTMO BOT BACKTEST")
    print("="*60)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model not found!")
        return
    
    model = PPO.load(model_path)
    
    # Load data
    print(f"📂 Loading test data: {data_path}")
    df = pd.read_csv(data_path)
    df = add_technical_indicators(df)
    
    # Use only UNSEEN validation portion (last 20%)
    split_idx = int(len(df) * 0.80)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"   Test set: {len(df_test)} candles (unseen data)")
    
    # Run episodes
    results = []
    
    for ep in range(n_episodes):
        env = FTMOTradingEnv(df_test)
        obs, _ = env.reset()
        
        episode_data = {
            'net_worth_history': [env.initial_balance],
            'actions': [],
            'daily_drawdowns': [],
            'total_trades': 0
        }
        
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data['net_worth_history'].append(info['net_worth'])
            episode_data['actions'].append(action)
            episode_data['daily_drawdowns'].append(info['daily_drawdown'])
            
            step_count += 1
        
        # Episode summary
        final_balance = episode_data['net_worth_history'][-1]
        profit_pct = ((final_balance - env.initial_balance) / env.initial_balance) * 100
        max_dd = max(episode_data['daily_drawdowns'])
        
        result = {
            'episode': ep + 1,
            'final_balance': final_balance,
            'profit_pct': profit_pct,
            'max_daily_drawdown': max_dd * 100,
            'steps': step_count,
            'passed_ftmo': final_balance >= 11000 and max_dd < 0.05
        }
        
        results.append(result)
        
        status = "✅ PASS" if result['passed_ftmo'] else "❌ FAIL"
        print(f"\n Episode {ep+1}/{n_episodes} {status}")
        print(f"   Final Balance: ${final_balance:,.2f}")
        print(f"   Profit: {profit_pct:+.2f}%")
        print(f"   Max Daily DD: {max_dd*100:.2f}%")
    
    # Overall statistics
    print("\n" + "="*60)
    print("📊 OVERALL RESULTS")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    pass_rate = (df_results['passed_ftmo'].sum() / len(df_results)) * 100
    avg_profit = df_results['profit_pct'].mean()
    avg_dd = df_results['max_daily_drawdown'].mean()
    
    print(f"\n✅ FTMO Pass Rate: {pass_rate:.0f}% ({df_results['passed_ftmo'].sum()}/{len(df_results)})")
    print(f"💰 Average Profit: {avg_profit:+.2f}%")
    print(f"📉 Average Max DD: {avg_dd:.2f}%")
    print(f"🎯 Best Performance: {df_results['profit_pct'].max():+.2f}%")
    print(f"📊 Worst Performance: {df_results['profit_pct'].min():+.2f}%")
    
    # Verdict
    print("\n" + "="*60)
    if pass_rate >= 70:
        print("🏆 EXCELENTE: Bot listo para FTMO!")
    elif pass_rate >= 50:
        print("⚠️ BUENO: Puede pasar FTMO con algo de suerte")
    elif pass_rate >= 30:
        print("⚠️ REGULAR: Necesita más entrenamiento")
    else:
        print("❌ INSUFICIENTE: Requiere rediseño")
    print("="*60)
    
    return df_results

if __name__ == "__main__":
    # Test the best model
    backtest_ftmo_bot(
        model_path="models/FTMO_CHAMPION/best_model.zip",
        data_path="data/btc_binance_15m.csv",
        n_episodes=20  # Run 20 simulations
    )
