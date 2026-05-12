"""H1: Asian-Range Breakout at London Open.

Thesis: During Asian session (00:00-07:00 broker time), FX majors compress into a range.
At London open (07:00-10:00), institutional order flow breaks the range in the dominant
direction, producing an asymmetric expansion. Holding until NY close captures the move.

Rules:
  - At 07:00 broker time each day, compute Asian range high/low (from 00:00-07:00 bars).
  - During 07:00-10:00 window: first 5-min close above Asian high → BUY, below low → SELL.
  - Entry at bar open AFTER breakout bar (harness handles this via next-bar fill).
  - SL = opposite end of Asian range. TP = entry + RR * SL_distance.
  - No more than 1 signal per day. EOD close handled by harness.

Why this is a falsifiable edge: it's been documented since the 1980s (Toby Crabel's
"Opening Range Breakout"), has a clear behavioral/structural rationale, and is trivial
to backtest honestly. If it fails here, it fails.
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import Strategy, Signal


class AsianBreakout(Strategy):
    name = "asian_breakout"

    def __init__(self, asian_start_h: int = 0, asian_end_h: int = 7,
                 breakout_window_h: int = 3, rr_ratio: float = 1.5,
                 min_range_points: float = 50.0, max_range_points: float = 400.0):
        """
        asian_start_h / asian_end_h: Asian session window (broker time hours).
        breakout_window_h: How many hours after Asian close we accept a breakout.
        rr_ratio: Take profit = entry + RR * (entry - SL).
        min/max_range_points: Skip days with abnormal Asian range (too tight = no signal,
                              too wide = volatile regime, edge disappears).
        """
        self.asian_start_h = asian_start_h
        self.asian_end_h = asian_end_h
        self.breakout_window_h = breakout_window_h
        self.rr_ratio = rr_ratio
        self.min_range_points = min_range_points
        self.max_range_points = max_range_points

        self._last_signal_date = None  # one-per-day

    def prepare(self, df):
        # Precompute Asian high/low per date on a vectorized pass
        if 'Asian_High' in df.columns:
            return
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        asian_mask = (df['hour'] >= self.asian_start_h) & (df['hour'] < self.asian_end_h)
        asian_bars = df[asian_mask]
        asian_hl = asian_bars.groupby('date').agg(
            Asian_High=('High', 'max'),
            Asian_Low=('Low', 'min'),
        )
        df['Asian_High'] = df['date'].map(asian_hl['Asian_High'])
        df['Asian_Low'] = df['date'].map(asian_hl['Asian_Low'])

    def on_bar(self, idx: int, df: pd.DataFrame, has_position: bool) -> Optional[Signal]:
        if has_position:
            return None
        if idx == 0:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        # Only within breakout window
        bw_end = self.asian_end_h + self.breakout_window_h
        if not (self.asian_end_h <= ts.hour < bw_end):
            return None

        # One signal per day
        if self._last_signal_date == ts.date():
            return None

        asian_hi = bar['Asian_High']
        asian_lo = bar['Asian_Low']
        if pd.isna(asian_hi) or pd.isna(asian_lo):
            return None

        # Range sanity filter (in "points" — here 1 point = instrument's point size;
        # we compute range in price units and let the caller tune min/max via instrument)
        # To stay instrument-agnostic, express filter as percent of Asian midpoint.
        asian_mid = (asian_hi + asian_lo) / 2
        if asian_mid <= 0:
            return None
        range_pct = (asian_hi - asian_lo) / asian_mid
        # Skip if range <0.05% (too tight) or >0.6% (volatile regime)
        if range_pct < 0.0005 or range_pct > 0.006:
            return None

        close = bar['Close']
        # BUY breakout: current close above Asian high
        if close > asian_hi:
            entry = close  # executes at next open
            sl = asian_lo
            tp = entry + self.rr_ratio * (entry - sl)
            if sl >= entry:
                return None
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=entry, sl=sl, tp=tp, tag="asian_hi_break")

        # SELL breakout: current close below Asian low
        if close < asian_lo:
            entry = close
            sl = asian_hi
            tp = entry - self.rr_ratio * (sl - entry)
            if sl <= entry:
                return None
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=entry, sl=sl, tp=tp, tag="asian_lo_break")

        return None
