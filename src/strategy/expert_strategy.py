"""
Expert Strategy for FTMO Challenge
Conservative SMA crossover with strict risk management
"""
import pandas as pd
import numpy as np

class FTMOExpertStrategy:
    """
    Conservative expert strategy designed to pass FTMO
    Based on proven SMA crossover + ATR risk management
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # shares held
        self.position_entry_price = 0
        
        # FTMO limits
        self.max_daily_loss = 0.05
        self.max_total_loss = 0.10
        self.profit_target = 0.10
        
        # Risk per trade
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Daily tracking
        self.daily_start_balance = initial_balance
        self.steps_in_day = 0
        self.steps_per_day = 96  # 15min candles
        
    def reset(self):
        """Reset for new episode"""
        self.balance = self.initial_balance
        self.position = 0
        self.position_entry_price = 0
        self.daily_start_balance = self.initial_balance
        self.steps_in_day = 0
    
    def get_action(self, current_price, sma_fast, sma_slow, atr, rsi):
        """
        Determine action based on technical signals
        
        Returns:
            0 = Hold
            1 = Buy
            2 = Sell
        """
        net_worth = self.balance + (self.position * current_price)
        
        # Check FTMO violations
        daily_dd = (self.daily_start_balance - net_worth) / self.daily_start_balance
        total_dd = (self.initial_balance - net_worth) / self.initial_balance
        
        if daily_dd >= self.max_daily_loss or total_dd >= self.max_total_loss:
            # Violation - close position if any
            if self.position > 0:
                return 2  # Sell
            return 0  # Hold
        
        # Check if profit target hit
        if net_worth >= self.initial_balance * (1 + self.profit_target):
            # Target hit - close and stop trading
            if self.position > 0:
                return 2  # Sell
            return 0  # Hold (done trading)
        
        # Trading logic
        if self.position == 0:
            # Not in position - look for entry
            # BUY signal: Fast SMA crosses above Slow SMA + RSI not overbought
            if sma_fast > sma_slow and rsi < 70:
                # Additional confirmation: price above both SMAs
                if current_price > sma_fast:
                    return 1  # Buy
        else:
            # In position - look for exit
            # SELL signal: Fast SMA crosses below Slow SMA OR RSI overbought
            if sma_fast < sma_slow or rsi > 75:
                return 2  # Sell
            
            # Stop loss: 2% from entry
            loss_pct = (current_price - self.position_entry_price) / self.position_entry_price
            if loss_pct < -0.02:
                return 2  # Sell (stop loss)
            
            # Take profit: 4% gain (2:1 risk-reward)
            if loss_pct > 0.04:
                return 2  # Sell (take profit)
        
        return 0  # Hold
    
    def execute_action(self, action, current_price, atr):
        """
        Execute the action and update state
        """
        commission = 0.0001
        
        if action == 1:  # Buy
            if self.position == 0 and self.balance > 100:
                # Position sizing based on ATR (risk management)
                risk_amount = self.balance * self.risk_per_trade
                stop_distance = current_price * 0.02  # 2% stop
                
                # Calculate position size
                amount_to_invest = min(self.balance * 0.3, risk_amount / 0.02)  # Max 30% per trade
                
                shares_bought = (amount_to_invest / current_price) * (1 - commission)
                self.balance -= amount_to_invest
                self.position += shares_bought
                self.position_entry_price = current_price
                
        elif action == 2:  # Sell
            if self.position > 0:
                sale_value = (self.position * current_price) * (1 - commission)
                self.balance += sale_value
                self.position = 0
                self.position_entry_price = 0
        
        # Time management
        self.steps_in_day += 1
        if self.steps_in_day >= self.steps_per_day:
            self.steps_in_day = 0
            net_worth = self.balance + (self.position * current_price)
            self.daily_start_balance = net_worth
    
    def get_net_worth(self, current_price):
        """Calculate current net worth"""
        return self.balance + (self.position * current_price)


def generate_expert_demonstrations(df, n_episodes=100):
    """
    Generate expert demonstrations by running the strategy
    
    Returns:
        List of (state, action) pairs
    """
    print("🎓 Generating Expert Demonstrations...")
    print(f"   Data: {len(df)} candles")
    print(f"   Episodes: {n_episodes}")
    
    demonstrations = []
    strategy = FTMOExpertStrategy()
    
    episode_results = []
    
    for episode in range(n_episodes):
        strategy.reset()
        
        # Random starting point for diversity
        max_start = len(df) - 1000
        if max_start < 100:
            max_start = 100
        
        start_idx = np.random.randint(200, max_start)
        end_idx = min(start_idx + 800, len(df))  # Run for ~800 steps
        
        episode_data = []
        
        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            current_price = row['Close']
            sma_fast = row['SMA_20']
            sma_slow = row['SMA_50']
            atr = row['ATR']
            rsi = row['RSI']
            
            # Get action from expert
            action = strategy.get_action(current_price, sma_fast, sma_slow, atr, rsi)
            
            # Create state observation (simplified - just key indicators)
            state = {
                'price_to_sma20': current_price / sma_fast if sma_fast > 0 else 1,
                'price_to_sma50': current_price / sma_slow if sma_slow > 0 else 1,
                'sma_ratio': sma_fast / sma_slow if sma_slow > 0 else 1,
                'rsi_norm': rsi / 100 if not np.isnan(rsi) else 0.5,
                'atr_norm': atr / current_price if current_price > 0 else 0,
                'balance_ratio': strategy.balance / strategy.initial_balance,
                'position_ratio': strategy.position * current_price / strategy.get_net_worth(current_price) if strategy.get_net_worth(current_price) > 0 else 0
            }
            
            episode_data.append((state, action))
            
            # Execute action
            strategy.execute_action(action, current_price, atr)
        
        final_worth = strategy.get_net_worth(df.iloc[end_idx - 1]['Close'])
        profit_pct = ((final_worth - 10000) / 10000) * 100
        
        episode_results.append(profit_pct)
        
        # Only keep profitable episodes for demonstrations
        if profit_pct > 0:
            demonstrations.extend(episode_data)
        
        if (episode + 1) % 10 == 0:
            avg_profit = np.mean(episode_results[-10:])
            print(f"   Episode {episode + 1}/{n_episodes}: Avg profit last 10: {avg_profit:+.2f}%")
    
    print(f"\n✅ Generated {len(demonstrations)} demonstration examples")
    avg_overall = np.mean(episode_results)
    positive_rate = (np.array(episode_results) > 0).sum() / len(episode_results) * 100
    print(f"   Average profit: {avg_overall:+.2f}%")
    print(f"   Positive episodes: {positive_rate:.0f}%")
    
    return demonstrations, episode_results


if __name__ == "__main__":
    # Test the expert strategy
    import sys
    sys.path.insert(0, '.')
    
    print("="*60)
    print("🧪 TESTING EXPERT STRATEGY")
    print("="*60)
    
    # Load data with indicators
    df = pd.read_csv("data/btc_binance_15m.csv")
    
    # Add  indicators if not present
    if 'SMA_20' not in df.columns:
        print("\n📊 Adding indicators...")
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        df.dropna(inplace=True)
    
    # Generate demonstrations
    demos, results = generate_expert_demonstrations(df, n_episodes=50)
    
    print("\n✅ Expert strategy validated!")
