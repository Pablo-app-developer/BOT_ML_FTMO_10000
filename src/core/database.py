"""
Core Database Manager for FTMO Challenger
Stores records and ensures data integrity.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="data/ftmo_trading.db"):
        self.db_path = db_path
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        print(f"📊 FTMO Database initialized: {db_path}")
    
    def create_tables(self):
        """Create optimized tables for high performance"""
        cursor = self.conn.cursor()
        
        # Trades table - Core operational data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                order_type TEXT DEFAULT 'MARKET',
                price REAL,
                volume REAL,
                fee REAL DEFAULT 0,
                pnl_realized REAL DEFAULT 0,
                balance_after REAL,
                notes TEXT
            )
        ''')
        
        # Daily Stats - Critical for FTMO Rules (Max Daily Loss)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                starting_balance REAL,
                current_balance REAL,
                min_balance REAL,  # For Max Drawdown tracking
                max_balance REAL,
                pnl_daily REAL,
                is_locked BOOLEAN DEFAULT 0  # To lock trading if limit reached
            )
        ''')
        
        # Bot State - For seamless recovery
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_state (
                symbol TEXT PRIMARY KEY,
                position_size REAL DEFAULT 0,
                entry_price REAL DEFAULT 0,
                stop_loss REAL DEFAULT 0,
                take_profit REAL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def log_trade(self, symbol, action, price, volume=0, pnl=0, balance=0, notes=""):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, action, price, volume, pnl_realized, balance_after, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, action, price, volume, pnl, balance, notes))
        self.conn.commit()
        return cursor.lastrowid
        
    def get_bot_state(self, symbol):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bot_state WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        if row:
            return {
                "symbol": row[0],
                "position_size": row[1],
                "entry_price": row[2],
                "stop_loss": row[3],
                "take_profit": row[4]
            }
        return None

    def update_bot_state(self, symbol, size, entry, sl, tp):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO bot_state (symbol, position_size, entry_price, stop_loss, take_profit, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (symbol, size, entry, sl, tp))
        self.conn.commit()

    def update_daily_stats(self, balance):
        """Updates daily statistics to track FTMO Drawdown limits"""
        date = datetime.now().strftime("%Y-%m-%d")
        cursor = self.conn.cursor()
        
        # Check if today exists
        cursor.execute("SELECT starting_balance, min_balance FROM daily_stats WHERE date=?", (date,))
        row = cursor.fetchone()
        
        if row:
            start_bal, min_bal = row
            new_min = min(min_bal, balance)
            new_max = max(min_bal, balance) # Simplification, should track real max
            pnl = balance - start_bal
            
            cursor.execute('''
                UPDATE daily_stats 
                SET current_balance=?, min_balance=?, pnl_daily=?
                WHERE date=?
            ''', (balance, new_min, pnl, date))
        else:
            # First update of the day
            cursor.execute('''
                INSERT INTO daily_stats (date, starting_balance, current_balance, min_balance, max_balance, pnl_daily)
                VALUES (?, ?, ?, ?, ?, 0)
            ''', (date, balance, balance, balance, balance))
            
        self.conn.commit()

    def close(self):
        self.conn.close()
