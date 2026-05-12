"""H2: Opening Range Breakout (ORB) — NY session.

Thesis: The first N minutes after a major session open establish the day's liquidity
grab. Price that breaks the opening range in the first few hours tends to extend.
Classic edge on indices (SPY, QQQ, US30, DAX) and high-volume FX (EURUSD at NY open).

Rules:
  - Opening range = High/Low of first `orb_minutes` after `open_hour` (broker time).
  - After opening range forms, FIRST close beyond it within `breakout_window_minutes`
    triggers entry (next bar open).
  - SL = opposite side of opening range. TP = entry + RR * SL_distance.
  - One trade per day. EOD close via harness.

References: Toby Crabel "Day Trading with Short Term Price Patterns", Linda Raschke's
"Turtle Soup Plus One", Laurent Bernut's public ORB studies.
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import Strategy, Signal


class OpeningRangeBreakout(Strategy):
    name = "opening_range_breakout"

    def __init__(self, open_hour: int = 13, open_minute: int = 30,
                 orb_minutes: int = 30, breakout_window_minutes: int = 120,
                 rr_ratio: float = 1.5):
        """
        open_hour/open_minute: Session open in broker time. Default 13:30 = NY open (US30 on FTMO
                               typical broker serves EET = UTC+2/+3, making NY open 13:30/14:30).
        orb_minutes: Length of the opening range.
        breakout_window_minutes: Minutes after ORB close during which we accept a break.
        """
        self.open_hour = open_hour
        self.open_minute = open_minute
        self.orb_minutes = orb_minutes
        self.breakout_window_minutes = breakout_window_minutes
        self.rr_ratio = rr_ratio
        self._last_signal_date = None

    def _session_start(self, ts):
        return ts.replace(hour=self.open_hour, minute=self.open_minute, second=0, microsecond=0)

    def prepare(self, df):
        if 'ORB_High' in df.columns:
            return
        df['date'] = df.index.date

        # Compute ORB high/low per day from bars within [open, open + orb_minutes)
        orb_high = {}
        orb_low = {}
        for day, grp in df.groupby('date'):
            session_start = pd.Timestamp(day).replace(hour=self.open_hour, minute=self.open_minute)
            orb_end = session_start + pd.Timedelta(minutes=self.orb_minutes)
            orb_bars = grp[(grp.index >= session_start) & (grp.index < orb_end)]
            if len(orb_bars) == 0:
                orb_high[day] = np.nan
                orb_low[day] = np.nan
            else:
                orb_high[day] = orb_bars['High'].max()
                orb_low[day] = orb_bars['Low'].min()

        df['ORB_High'] = df['date'].map(orb_high)
        df['ORB_Low'] = df['date'].map(orb_low)

    def on_bar(self, idx, df, has_position):
        if has_position:
            return None
        if idx == 0:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        session_start = self._session_start(ts)
        orb_end = session_start + pd.Timedelta(minutes=self.orb_minutes)
        window_end = orb_end + pd.Timedelta(minutes=self.breakout_window_minutes)

        if ts < orb_end or ts >= window_end:
            return None
        if self._last_signal_date == ts.date():
            return None

        orb_hi = bar['ORB_High']
        orb_lo = bar['ORB_Low']
        if pd.isna(orb_hi) or pd.isna(orb_lo) or orb_hi <= orb_lo:
            return None

        close = bar['Close']
        if close > orb_hi:
            entry = close
            sl = orb_lo
            tp = entry + self.rr_ratio * (entry - sl)
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=entry, sl=sl, tp=tp, tag="orb_hi_break")
        if close < orb_lo:
            entry = close
            sl = orb_hi
            tp = entry - self.rr_ratio * (sl - entry)
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=entry, sl=sl, tp=tp, tag="orb_lo_break")

        return None
