"""London Open Momentum Breakout.

Thesis: The London open (08:00 UTC) is the highest-volume, most directional
window of the day. Institutional order flow creates sustained moves. Fade the
Asian session's low-volume drift and trade the first real direction at London open.

Mechanics:
  - Build the "Asia range" from 00:00-07:55 UTC: high and low of that window.
  - At 08:00 UTC, wait for price to break out of the Asia range on M5.
  - Entry on the first M5 close above (BUY) or below (SELL) the range extreme.
  - SL = opposite side of range + 0.5*ATR buffer.
  - TP = entry + rr * SL_distance.
  - Filter 1: Asia range width > min_range_atr_frac * ATR(14) — avoid ultra-thin ranges.
  - Filter 2: Asia range width < max_range_atr_frac * ATR(14) — avoid pre-expanded days.
  - Filter 3: Only trade until cutoff_hour (e.g. 11:00) — if no breakout by then, skip.
  - One trade per day.

Why this is different from current ORB:
  Current ORB just trades a fixed-time window breakout regardless of conditions.
  This version uses the overnight Asian range as the setup, only enters when range
  is "tight enough to be interesting but not trivially small", and respects a cutoff.

Instruments: EURUSD, GBPUSD, DE40 (European session plays well here)
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal


class LondonMomentum(Strategy):
    name = "london_momentum"

    def __init__(self,
                 asia_start_h: int = 0,    # Asia session start (UTC)
                 asia_end_h: int = 8,       # Asia session end = London open
                 cutoff_h: int = 11,        # Stop looking after this hour
                 min_range_atr_frac: float = 0.3,  # Asia range must be >= 30% of ATR
                 max_range_atr_frac: float = 1.5,  # Asia range must be <= 150% of ATR
                 rr_ratio: float = 2.5,
                 atr_n: int = 14):
        self.asia_start_h = asia_start_h
        self.asia_end_h = asia_end_h
        self.cutoff_h = cutoff_h
        self.min_range_atr_frac = min_range_atr_frac
        self.max_range_atr_frac = max_range_atr_frac
        self.rr_ratio = rr_ratio
        self.atr_n = atr_n
        self._last_signal_date = None

    def prepare(self, df: pd.DataFrame):
        if 'H1_ATR' in df.columns:
            return

        # H1 ATR — use H1 so the Asia range comparison is on the same scale
        h1 = df[['Open', 'High', 'Low', 'Close']].resample(
            '1h', label='right', closed='right'
        ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        prev_c = h1['Close'].shift(1)
        tr_h1 = pd.concat([
            h1['High'] - h1['Low'],
            (h1['High'] - prev_c).abs(),
            (h1['Low'] - prev_c).abs(),
        ], axis=1).max(axis=1)
        h1['ATR'] = tr_h1.rolling(self.atr_n).mean()
        df['H1_ATR'] = h1['ATR'].reindex(df.index, method='ffill')

        # Pre-compute daily Asia range: high and low during asia_start_h to asia_end_h
        asia_mask = (df.index.hour >= self.asia_start_h) & (df.index.hour < self.asia_end_h)
        asia_bars = df[asia_mask]
        asia_daily = asia_bars.groupby(asia_bars.index.date).agg(
            AsiaHigh=('High', 'max'),
            AsiaLow=('Low', 'min'),
        )
        # Forward-fill: each bar gets today's Asia range (only valid after asia_end_h)
        asia_daily.index = pd.to_datetime(asia_daily.index)
        date_idx = df.index.normalize()
        df['AsiaHigh'] = date_idx.map(asia_daily['AsiaHigh'].to_dict())
        df['AsiaLow'] = date_idx.map(asia_daily['AsiaLow'].to_dict())

    def on_bar(self, idx: int, df: pd.DataFrame, has_position: bool) -> Signal:
        if has_position:
            return None
        if idx < self.atr_n + 1:
            return None

        bar = df.iloc[idx]
        ts = bar.name
        hour = ts.hour

        # Only look for entries during London session up to cutoff
        if not (self.asia_end_h <= hour < self.cutoff_h):
            return None
        if self._last_signal_date == ts.date():
            return None

        asia_high = bar['AsiaHigh']
        asia_low = bar['AsiaLow']
        atr = bar['H1_ATR']   # H1 ATR for range comparison
        close = bar['Close']

        if pd.isna(asia_high) or pd.isna(asia_low) or pd.isna(atr) or atr <= 0:
            return None

        asia_range = asia_high - asia_low
        if asia_range <= 0:
            return None
        # Range filters
        if asia_range < self.min_range_atr_frac * atr:
            return None
        if asia_range > self.max_range_atr_frac * atr:
            return None

        prev_close = df.iloc[idx - 1]['Close']

        # Breakout up
        if close > asia_high and prev_close <= asia_high:
            sl = asia_low - 0.5 * atr
            sl_dist = abs(close - sl)
            if sl_dist <= 0:
                return None
            tp = close + self.rr_ratio * sl_dist
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=close, sl=sl, tp=tp, tag="london_break_up")

        # Breakout down
        if close < asia_low and prev_close >= asia_low:
            sl = asia_high + 0.5 * atr
            sl_dist = abs(sl - close)
            if sl_dist <= 0:
                return None
            tp = close - self.rr_ratio * sl_dist
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=close, sl=sl, tp=tp, tag="london_break_dn")

        return None
