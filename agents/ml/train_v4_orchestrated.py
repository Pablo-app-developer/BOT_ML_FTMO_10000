"""
ORCHESTRATED TRAINING V4 (GPU Driven)
Collaborative training where Quant, ML, and QA agents interact.
"""
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# 🔵 FIX PATHS BEFORE IMPORTS
# Add project root to path (2 levels up from agents/ml)
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can safely import local modules
from agents.ml.deepseek_assistant import DeepSeekAssistant
from src.strategy.ftmo_env_v2 import FTMOTradingEnvV2

# 🟢 AGENT CONFIGURATION
QUANT_CONFIG = os.path.join(project_root, "agents/quant/risk_parameters.yaml")
DATA_DIR = os.path.join(project_root, "shared/data")
MODELS_DIR = os.path.join(project_root, "shared/models")
LOG_DIR = os.path.join(project_root, "shared/backtest")
MODEL_NAME = "ftmo_crypto_v4_gpu"

class AgentCouncilCallback(BaseCallback):
    """
    The 'Council of Agents' evaluating the model during training.
    """
    def __init__(self, eval_env, check_freq=10000):
        super(AgentCouncilCallback, self).__init__(verbose=1)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.best_sharpe = -np.inf
        
        # Determine strictness from config
        with open(QUANT_CONFIG, 'r') as f:
            self.risk_cfg = yaml.safe_load(f)
            
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"\n📢 [Step {self.n_calls}] The Agent Council is convening...")
            
            # QA Agent: Runs a quick validation
            mean_reward, std_reward, info = self.run_validation()
            
            # Quant Agent: Evaluates Risk
            passed_risk_check = self.quant_risk_audit(info)
            
            if passed_risk_check:
                print("✅ Quant Agent: RISK APPROVED. Model respects drawdowns.")
                
                # ML Agent: Saves if it's a new best
                sharpe = info.get('sharpe_ratio', 0)
                if sharpe > self.best_sharpe:
                    self.best_sharpe = sharpe
                    path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.zip")
                    self.model.save(path)
                    print(f"💾 ML Engineer: New Best Model Saved! Sharpe: {sharpe:.2f}")
            else:
                print("❌ Quant Agent: RISK REJECTED. Drawdown too high.")
            
            print("-" * 50)
            
        return True

    def run_validation(self):
        """Simulate QA runs"""
        obs = self.eval_env.reset()
        done = False
        total_rewards = []
        infos = {}
        
        # Run 5 episodes
        for _ in range(5):
            episode_rew = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_rew += reward
            total_rewards.append(episode_rew)
            infos = info[0] # Take last info
            done = False
            
        return np.mean(total_rewards), np.std(total_rewards), infos

    def quant_risk_audit(self, info):
        """Quant checks if FTMO rules are violated in validation"""
        daily_dd = info.get('daily_drawdown', 0)
        limit = self.risk_cfg['ftmo_limits']['max_daily_loss_percent']
        
        # Quant allows 80% of the limit during training (safety margin)
        threshold = limit * 0.8 
        
        if daily_dd > threshold:
            print(f"⚠️ Risk Alert: Daily DD {daily_dd:.2%} exceeds threshold {threshold:.2%}")
            return False
        return True

def load_data():
    """ML Agent loads and splits data"""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("processed_") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No processed data found! Run data_pipeline.py first.")
    
    print(f"📚 Found {len(files)} datasets: {files}")
    
    # Combine datasets (Interleaved or Concat)
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True) # Simple concat for now
    
    # Split: Train (80%), Test (20%) - QA Agent requirement
    split_idx = int(len(combined) * 0.8)
    train_df = combined.iloc[:split_idx]
    test_df = combined.iloc[split_idx:]
    
    return train_df, test_df

def main():
    print("🚀 STARTING TRAINING SESSION V4")
    print(f"running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Prepare Data
    train_df, test_df = load_data()
    print(f"📊 Training Data: {len(train_df)} rows")
    print(f"🔍 Validation Data: {len(test_df)} rows (QA Holdout)")
    
    # 2. Setup Environments
    # We use vector environments for speed
    train_env = DummyVecEnv([lambda: FTMOTradingEnvV2(train_df)])
    eval_env = DummyVecEnv([lambda: FTMOTradingEnvV2(test_df)])
    
    # 3. Setup Model (PPO) with GPU
    # ML Agent optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=LOG_DIR
    )
    
    # 4. Setup Council Callback
    council = AgentCouncilCallback(eval_env, check_freq=5000) # Check every 5k steps
    
    # 5. Train
    try:
        total_steps = 1000000
        print(f"🏋️ Training for {total_steps} steps...")
        model.learn(total_timesteps=total_steps, callback=council, progress_bar=True)
        print("✅ Training Complete.")
        
        # Save Final
        model.save(os.path.join(MODELS_DIR, f"{MODEL_NAME}_final.zip"))
        
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted manually.")
        model.save(os.path.join(MODELS_DIR, f"{MODEL_NAME}_interrupted.zip"))

if __name__ == "__main__":
    main()
