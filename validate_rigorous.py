"""
RIGOROUS FTMO BOT VALIDATION
Test on completely UNSEEN data from multiple assets
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.strategy.ftmo_env import FTMOTradingEnv

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

def test_on_asset(model, data_file, asset_name, n_runs=10):
    """Test bot on a specific asset"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING ON: {asset_name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(data_file)
    print(f"Raw data: {len(df)} candles")
    
    df = add_indicators(df)
    print(f"After indicators: {len(df)} candles")
    
    results = []
    net_worth_histories = []
    
    for run in range(n_runs):
        env = FTMOTradingEnv(df)
        obs, _ = env.reset()
        
        net_worth_history = [10000]
        max_daily_dd = 0
        
        done = False
        steps = 0
        
        while not done and steps < 10000:  # Safety limit
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            net_worth_history.append(info['net_worth'])
            max_daily_dd = max(max_daily_dd, info['daily_drawdown'])
            steps += 1
        
        final_balance = info['net_worth']
        profit_pct = ((final_balance - 10000) / 10000) * 100
        
        # FTMO Pass criteria
        passed_profit = final_balance >= 11000  # Hit 10% target
        passed_daily_dd = max_daily_dd < 0.05   # Never exceeded 5% daily loss
        
        max_total_dd = ((10000 - min(net_worth_history)) / 10000)
        passed_total_dd = max_total_dd < 0.10   # Never exceeded 10% total loss
        
        passed_ftmo = passed_profit and passed_daily_dd and passed_total_dd
        
        results.append({
            'run': run + 1,
            'final_balance': final_balance,
            'profit_pct': profit_pct,
            'max_daily_dd': max_daily_dd * 100,
            'max_total_dd': max_total_dd * 100,
            'steps': steps,
            'passed': passed_ftmo
        })
        
        net_worth_histories.append(net_worth_history)
        
        status = "✅" if passed_ftmo else "❌"
        reason = ""
        if not passed_profit:
            reason = "(no profit target)"
        elif not passed_daily_dd:
            reason = f"(daily DD: {max_daily_dd*100:.2f}%)"
        elif not passed_total_dd:
            reason = f"(total DD: {max_total_dd*100:.2f}%)"
        
        print(f"Run {run+1:2d}: ${final_balance:,.0f} ({profit_pct:+.2f}%) {status} {reason}")
    
    # Statistics
    df_results = pd.DataFrame(results)
    pass_rate = (df_results['passed'].sum() / len(df_results)) * 100
    avg_profit = df_results['profit_pct'].mean()
    std_profit = df_results['profit_pct'].std()
    
    print(f"\n📊 SUMMARY FOR {asset_name}:")
    print(f"   Pass Rate: {pass_rate:.0f}% ({df_results['passed'].sum()}/{len(df_results)})")
    print(f"   Avg Profit: {avg_profit:+.2f}% (±{std_profit:.2f}%)")
    print(f"   Best: {df_results['profit_pct'].max():+.2f}%")
    print(f"   Worst: {df_results['profit_pct'].min():+.2f}%")
    print(f"   Avg Max Daily DD: {df_results['max_daily_dd'].mean():.2f}%")
    print(f"   Avg Max Total DD: {df_results['max_total_dd'].mean():.2f}%")
    
    return df_results

def main():
    print("="*60)
    print("🔬 RIGOROUS FTMO BOT VALIDATION")
    print("   Testing on COMPLETELY UNSEEN DATA")
    print("="*60)
    
    # Load model
    print("\n📦 Loading trained model...")
    model = PPO.load("models/FTMO_CHAMPION/best_model.zip")
    print("✅ Model loaded")
    
    # Test on multiple assets
    all_results = {}
    
    datasets = [
        ('data/btc_validation_60d.csv', 'BTC (Fresh 60d)'),
        ('data/eth_validation_60d.csv', 'ETH (Fresh 60d)'),
        ('data/sol_validation_60d.csv', 'SOL (Fresh 60d)')
    ]
    
    for data_file, asset_name in datasets:
        if os.path.exists(data_file):
            results = test_on_asset(model, data_file, asset_name, n_runs=10)
            all_results[asset_name] = results
        else:
            print(f"\n⚠️ {data_file} not found, skipping...")
    
    # Overall verdict
    print("\n" + "="*60)
    print("🏆 OVERALL VALIDATION RESULTS")
    print("="*60)
    
    total_tests = 0
    total_passed = 0
    
    for asset_name, results in all_results.items():
        passed = results['passed'].sum()
        total = len(results)
        total_tests += total
        total_passed += passed
        print(f"{asset_name:20s}: {passed}/{total} passed ({(passed/total)*100:.0f}%)")
    
    overall_pass_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n{'OVERALL':20s}: {total_passed}/{total_tests} passed ({overall_pass_rate:.0f}%)")
    print("="*60)
    
    # Final verdict
    if overall_pass_rate >= 80:
        print("🏆 EXCELENTE: Bot es ROBUSTO y confiable!")
    elif overall_pass_rate >= 60:
        print("👍 BUENO: Bot tiene potencial sólido")
    elif overall_pass_rate >= 40:
        print("⚠️ REGULAR: Requiere más entrenamiento")
    elif overall_pass_rate >= 20:
        print("⚠️ DÉBIL: Necesita rediseño significativo")
    else:
        print("❌ INSUFICIENTE: Overfitting detectado")
    
    print("="*60)

if __name__ == "__main__":
    main()
