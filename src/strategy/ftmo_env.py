import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class FTMOTradingEnv(gym.Env):
    """
    Custom Environment for FTMO Challenge (10k account).
    Enforces strict risk management rules:
    - Max Daily Loss: 5% ($500)
    - Max Total Loss: 10% ($1000)
    - Profit Target: 10% ($1000)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission=0.0001, window_size=60):
        super(FTMOTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        
        # FTMO Rules
        self.max_daily_loss_pct = 0.05
        self.max_total_loss_pct = 0.10
        self.profit_target_pct = 0.10
        
        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Feature Engineering (Simplified for robust training)
        # We expect df to have 'Close', 'RSI', 'MACD', etc. calculated beforehand or here.
        # For simplicity, we assume df has pre-calculated features.
        # We'll use a subset of columns if they exist, else we calculate simple ones.
        
        # Filter strictly for numeric columns to avoid Timestamp/String issues
        self.obs_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove core OHLCV if present to focus on indicators, but keep Close if needed.
        # usually Env uses internal Close for pnl, so we might keep it in obs or not.
        # Let's clean up explicitly.
        exclude_cols = ['Open', 'High', 'Low', 'Volume'] 
        self.obs_cols = [c for c in self.obs_cols if c not in exclude_cols]
        
        # Fallback
        if not self.obs_cols:
             self.obs_cols = ['Close']

        self.n_features = len(self.obs_cols) + 2 # +2 for account info (balance, position)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.n_features), dtype=np.float32
        )

        # Validate dataframe size
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
        
        # Daily Drawdown Tracking
        self.daily_start_balance = self.balance
        self.steps_in_day = 0
        self.steps_per_day = 96 # Assuming 15m candles (24 * 4)

        return self._next_observation(), {}

    def _next_observation(self):
        # Market Stats
        market_obs = self.df.iloc[self.current_step - self.window_size : self.current_step][self.obs_cols].values
        
        # Account Stats
        balance_ratio = self.balance / self.initial_balance
        position_ratio = (self.shares_held * self.df.iloc[self.current_step]['Close']) / self.net_worth
        
        account_obs = np.full((self.window_size, 2), [balance_ratio, position_ratio], dtype=np.float32)
        
        obs = np.hstack((market_obs, account_obs))
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        done = False
        truncated = False
        
        # 1. Execute Action
        if action == 1: # Buy
            # Simple logic: invest 20% of balance per trade to manage risk
            amount_to_invest = self.balance * 0.20
            if amount_to_invest > 100 and self.balance > amount_to_invest:
                shares_bought = (amount_to_invest / current_price) * (1 - self.commission)
                self.balance -= amount_to_invest
                self.shares_held += shares_bought
                
        elif action == 2: # Sell (Close Position)
            if self.shares_held > 0:
                sale_value = (self.shares_held * current_price) * (1 - self.commission)
                self.balance += sale_value
                self.shares_held = 0
        
        # 2. Update Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # 3. Calculate Daily Drawdown
        daily_drawdown = (self.daily_start_balance - self.net_worth) / self.daily_start_balance
        total_drawdown = (self.initial_balance - self.net_worth) / self.initial_balance
        
        # 4. FTMO Rule Checks
        if daily_drawdown >= self.max_daily_loss_pct:
            done = True
            reward = -100 # Huge penalty for failing daily limit
        elif total_drawdown >= self.max_total_loss_pct:
            done = True
            reward = -100 # Huge penalty for blowing account
        elif self.net_worth >= self.initial_balance * (1 + self.profit_target_pct):
            done = True
            reward = 100 # Win!
        else:
            # Normal Reward: Change in Net Worth
            reward = (self.net_worth - self.initial_balance) / self.initial_balance
            
            # Tiny penalty for inactivity/holding too long if needed? 
            # For now, keep it simple: PnL driven.

        # 5. Time Management
        self.current_step += 1
        self.steps_in_day += 1
        
        if self.steps_in_day >= self.steps_per_day:
            self.steps_in_day = 0
            self.daily_start_balance = self.net_worth # Reset daily reference
            
        if self.current_step >= self.end_step:
            done = True

        info = {
            "net_worth": self.net_worth,
            "daily_drawdown": daily_drawdown
        }
        
        return self._next_observation(), reward, done, truncated, info
