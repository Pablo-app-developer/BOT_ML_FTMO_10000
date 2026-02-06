"""
FTMO Environment V2 - Anti-Overfitting Edition
Better reward shaping for generalization
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class FTMOTradingEnvV2(gym.Env):
    """
    Improved FTMO Environment focusing on generalization
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, risk_config_path='agents/quant/risk_parameters.yaml', window_size=60):
        super(FTMOTradingEnvV2, self).__init__()
        import yaml
        import os

        # Load Risk Config
        if os.path.exists(risk_config_path):
            with open(risk_config_path, 'r') as f:
                self.risk_config = yaml.safe_load(f)
        else:
            # Fallback default
            self.risk_config = {
                'ftmo_limits': {'max_daily_loss_percent': 0.05, 'max_total_loss_percent': 0.10, 'profit_target_percent': 0.10},
                'crypto_specifics': {'max_spread_tolerance_pips': 5.0}
            }
            
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = self.risk_config['ftmo_limits'].get('account_size', 10000)
        self.commission = 0.001  # 0.1% per trade (Taker)
        self.spread_pct = 0.0002 # 0.02% simulated spread
        
        # FTMO Rules from YAML
        self.max_daily_loss_pct = self.risk_config['ftmo_limits']['max_daily_loss_percent']
        self.max_total_loss_pct = self.risk_config['ftmo_limits']['max_total_loss_percent']
        self.profit_target_pct = self.risk_config['ftmo_limits']['profit_target_percent']
        
        self.action_space = spaces.Discrete(3)
        
        # Features
        self.obs_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Open', 'High', 'Low', 'Volume'] 
        self.obs_cols = [c for c in self.obs_cols if c not in exclude_cols]
        
        if not self.obs_cols:
             self.obs_cols = ['Close']

        self.n_features = len(self.obs_cols) + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.n_features), dtype=np.float32
        )

        if len(self.df) < self.window_size + 100:
            raise ValueError(f"Dataframe too small. Need at least {self.window_size + 100} rows, got {len(self.df)}")

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0
        
        self.current_step = self.window_size
        self.end_step = len(self.df) - 1
        
        # Daily tracking
        self.daily_start_balance = self.balance
        self.steps_in_day = 0
        self.steps_per_day = 96
        
        # Track for better rewards
        self.last_net_worth = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.steps_held = 0

        return self._next_observation(), {}

    def _next_observation(self):
        market_obs = self.df.iloc[self.current_step - self.window_size : self.current_step][self.obs_cols].values
        
        balance_ratio = self.balance / self.initial_balance
        position_ratio = (self.shares_held * self.df.iloc[self.current_step]['Close']) / self.net_worth if self.net_worth > 0 else 0
        
        account_obs = np.full((self.window_size, 2), [balance_ratio, position_ratio], dtype=np.float32)
        
        obs = np.hstack((market_obs, account_obs))
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        done = False
        truncated = False
        
        # Track previous for reward calculation
        prev_net_worth = self.net_worth
        
        # Execute Action
        if action == 1:  # Buy
            if self.shares_held == 0 and self.balance > 100:
                # Only buy if not already holding
                amount = self.balance * 0.25  # 25% position size (conservative)
                shares_bought = (amount / current_price) * (1 - self.commission)
                self.balance -= amount
                self.shares_held += shares_bought
                self.entry_price = current_price
                self.steps_held = 0
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                sale_value = (self.shares_held * current_price) * (1 - self.commission)
                profit = sale_value - (self.shares_held * self.entry_price)
                
                self.balance += sale_value
                self.shares_held = 0
                
                self.total_trades += 1
                if profit > 0:
                    self.winning_trades += 1
                    reward += 0.1  # Reward winning trades
                else:
                    reward -= 0.05  # Small penalty for losing trades
        
        # Update net worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Track if holding
        if self.shares_held > 0:
            self.steps_held += 1
            
            # Reward for holding winning positions
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            if unrealized_pnl > 0:
                reward += unrealized_pnl * 0.05  # Small reward for unrealized gains
            
            # Penalty for holding too long without action (opportunity cost)
            if self.steps_held > 200:  # ~50 hours at 15min candles
                reward -= 0.002
        
        # Calculate drawdowns
        daily_drawdown = (self.daily_start_balance - self.net_worth) / self.daily_start_balance if self.daily_start_balance > 0 else 0
        total_drawdown = (self.initial_balance - self.net_worth) / self.initial_balance
        
        # FTMO Rule Checks
        if daily_drawdown >= self.max_daily_loss_pct:
            done = True
            reward = -50  # Large penalty for daily limit violation
        elif total_drawdown >= self.max_total_loss_pct:
            done = True
            reward = -50  # Large penalty for total loss limit
        elif self.net_worth >= self.initial_balance * (1 + self.profit_target_pct):
            done = True
            reward = 50  # Large reward for hitting target
        else:
            # Normal step reward - focus on net worth change
            step_profit = (self.net_worth - prev_net_worth) / self.initial_balance
            reward += step_profit * 10  # Scale up the reward
            
            # Small penalty for excessive drawdown even if not violating
            if daily_drawdown > 0.03:  # Warning at 3%
                reward -= 0.05
            if total_drawdown > 0.05:  # Warning at 5%
                reward -= 0.05

        # Time management
        self.current_step += 1
        self.steps_in_day += 1
        
        if self.steps_in_day >= self.steps_per_day:
            self.steps_in_day = 0
            self.daily_start_balance = self.net_worth
            
        if self.current_step >= self.end_step:
            done = True
            # Final reward based on performance
            final_profit_pct = (self.net_worth - self.initial_balance) / self.initial_balance
            reward += final_profit_pct * 20  # Bonus for final profit

        info = {
            "net_worth": self.net_worth,
            "daily_drawdown": daily_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        }
        
        return self._next_observation(), reward, done, truncated, info
