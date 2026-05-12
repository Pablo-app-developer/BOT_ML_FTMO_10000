"""Multi-Session Opening Range Breakout.

Thesis: Every major session open (Asia, London, NY) creates a new range during
the first N minutes. Breakout of that range in the direction of the breakout
often continues for the session. Trading all three sessions triples the signal
frequency vs. single-session ORB.

Mechanics (M5 bars):
  - Three sessions, each with its own "opening range" window:
      Asia:   00:00-02:00 UTC → range complete at 02:00
      London: 07:00-08:30 UTC → range complete at 08:30
      NY:     13:00-13:30 UTC → range complete at 13:30
  - After the range window closes, watch for first M5 close outside the range.
  - Buy if close > range_high (and prev close was inside range).
  - Sell if close < range_low (and prev close was inside range).
  - SL = opposite side of range + atr_buffer * ATR(14).
  - TP = entry + rr_ratio * SL_dist.
  - Cutoff: if no breakout within cutoff_bars bars of the range end, skip.
  - Filter: range must be between min_atr_frac and max_atr_frac * ATR(14).
  - One trade per session per day.

Expected frequency: 2-3 signals/day (not all sessions break out cleanly).
At 40 signals/month with PF 1.3+, FTMO target is achievable.

Instruments: XAUUSD (trades all sessions), EURUSD (London+NY), USDJPY (Asia+London)
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal


# Session definitions (UTC)
SESSIONS = {
    'asia':   {'range_start': 0,  'range_end': 2,  'cutoff_offset_h': 3},
    'london': {'range_start': 7,  'range_end': 9,  'cutoff_offset_h': 3},
    'ny':     {'range_start': 13, 'range_end': 14, 'cutoff_offset_h': 3},
}


class MultiSessionORB(Strategy):
    name = "ms_orb"

    def __init__(self,
                 rr_ratio: float = 2.0,
                 atr_buffer: float = 0.3,
                 atr_n: int = 14,
                 min_range_atr_frac: float = 0.2,
                 max_range_atr_frac: float = 2.0,
                 cutoff_bars: int = 36,   # 3h of M5 bars after range close
                 sessions: tuple = ('asia', 'london', 'ny')):
        self.rr_ratio = rr_ratio
        self.atr_buffer = atr_buffer
        self.atr_n = atr_n
        self.min_range_atr_frac = min_range_atr_frac
        self.max_range_atr_frac = max_range_atr_frac
        self.cutoff_bars = cutoff_bars
        self.sessions_to_trade = list(sessions)
        # Per-day-session tracking: (date, session) -> traded?
        self._traded: set = set()
        # Active range: (session_key) -> {high, low, range_end_idx}
        self._active: dict = {}

    def prepare(self, df: pd.DataFrame):
        if 'ATR' in df.columns:
            return
        prev_c = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_c).abs(),
            (df['Low'] - prev_c).abs(),
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.atr_n).mean()

        # Pre-compute session ranges
        for sess_name, sess in SESSIONS.items():
            if sess_name not in self.sessions_to_trade:
                continue
            mask = (df.index.hour >= sess['range_start']) & (df.index.hour < sess['range_end'])
            sess_bars = df[mask]
            if sess_bars.empty:
                continue
            daily = sess_bars.groupby(sess_bars.index.date).agg(
                **{f'{sess_name}_high': ('High', 'max'),
                   f'{sess_name}_low':  ('Low',  'min')}
            )
            daily.index = pd.to_datetime(daily.index)
            date_idx = df.index.normalize()
            df[f'{sess_name}_high'] = date_idx.map(daily[f'{sess_name}_high'].to_dict())
            df[f'{sess_name}_low']  = date_idx.map(daily[f'{sess_name}_low'].to_dict())

    def _get_session(self, hour: int) -> str:
        """Return which session's breakout window we are in."""
        for sess_name, sess in SESSIONS.items():
            if sess_name not in self.sessions_to_trade:
                continue
            if sess['range_end'] <= hour < sess['range_end'] + sess['cutoff_offset_h']:
                return sess_name
        return None

    def on_bar(self, idx: int, df: pd.DataFrame, has_position: bool) -> Signal:
        if has_position:
            return None
        if idx < self.atr_n + 2:
            return None

        bar = df.iloc[idx]
        ts = bar.name
        hour = ts.hour
        date = ts.date()

        sess_name = self._get_session(hour)
        if sess_name is None:
            return None

        sess_key = (date, sess_name)
        if sess_key in self._traded:
            return None

        atr = bar['ATR']
        if pd.isna(atr) or atr <= 0:
            return None

        rh_col = f'{sess_name}_high'
        rl_col = f'{sess_name}_low'
        if rh_col not in df.columns or rl_col not in df.columns:
            return None

        range_high = bar[rh_col]
        range_low  = bar[rl_col]

        if pd.isna(range_high) or pd.isna(range_low):
            return None

        range_width = range_high - range_low
        if range_width <= 0:
            return None

        # Range quality filter
        if range_width < self.min_range_atr_frac * atr:
            return None
        if range_width > self.max_range_atr_frac * atr:
            return None

        close = bar['Close']
        if idx == 0:
            return None
        prev_close = df.iloc[idx - 1]['Close']

        # Bullish breakout: close crosses above range high
        if close > range_high and prev_close <= range_high:
            sl = range_low - self.atr_buffer * atr
            sl_dist = abs(close - sl)
            if sl_dist <= 0:
                return None
            tp = close + self.rr_ratio * sl_dist
            self._traded.add(sess_key)
            return Signal(side="BUY", entry=close, sl=sl, tp=tp,
                         tag=f"ms_orb_{sess_name}_up")

        # Bearish breakout: close crosses below range low
        if close < range_low and prev_close >= range_low:
            sl = range_high + self.atr_buffer * atr
            sl_dist = abs(sl - close)
            if sl_dist <= 0:
                return None
            tp = close - self.rr_ratio * sl_dist
            self._traded.add(sess_key)
            return Signal(side="SELL", entry=close, sl=sl, tp=tp,
                         tag=f"ms_orb_{sess_name}_dn")

        return None
