import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.strategy.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Antigravity FTMO Bot")
    parser.add_argument('mode', choices=['train', 'trade', 'backtest'], help="Mode to run the bot in")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting Training Protocol...")
        train_model()
    elif args.mode == 'trade':
        print("Live trading not yet implemented. Use 'train' first.")
    elif args.mode == 'backtest':
        print("Backtesting not yet implemented.")

if __name__ == "__main__":
    main()
