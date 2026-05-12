"""H5: Gap-and-Go Continuation on Index Cash Open.

Thesis: US equity indices gap at cash open (13:30 broker / 09:30 NY) because
overnight futures and global news accumulate while the cash market is closed.
When the gap is large AND the first minute(s) of trading confirm direction
(instead of filling), the momentum tends to continue for the session.

Mechanics:
  - Reference price = prior day's 20:00 broker-time close (approx cash close).
  - Gap = today's 13:30 broker-time Open − reference price.
  - Gap must exceed threshold (e.g. ATR(20) * 0.5) to matter.
  - Wait for the first `confirm_minutes` of cash open. If first-5m-candle
    closes in the direction of the gap (continuation), enter at that close.
  - SL = opposite extreme of the confirmation candle (tight — typically 10-25 pts on US30).
  - TP = 3R fixed (trail could help but adds complexity; keep it simple first).
  - One trade per day. Exit forced at EOD if still open.

Why it fits FTMO: 1 trade/day, typically 3R payoff, tight SL → small loss when
wrong, big win when right. A single winning streak of 3-4 days hits +10%.

Applicable only to indices (US30, NAS100, DE40) — FX does not have true gaps.
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import Strategy, Signal


class GapAndGo(Strategy):
    name = "gap_and_go"

    def __init__(self,
                 cash_open_h: int = 13, cash_open_m: int = 30,
                 cash_close_h: int = 20, cash_close_m: int = 0,
                 atr_window: int = 20, min_gap_atr_frac: float = 0.5,
                 rr: float = 3.0):
        self.cash_open_h = cash_open_h
        self.cash_open_m = cash_open_m
        self.cash_close_h = cash_close_h
        self.cash_close_m = cash_close_m
        self.atr_window = atr_window
        self.min_gap_atr_frac = min_gap_atr_frac
        self.rr = rr
        self._last_signal_date = None

    def prepare(self, df):
        if 'Daily_ATR' in df.columns:
            return

        # Build a daily series of prior-day 20:00 close and today's 13:30 open
        # using the actual M5 bars.
        df['Date'] = df.index.date

        # Daily range / ATR for gap-size filter
        daily = df.groupby('Date').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        daily['prev_close'] = daily['Close'].shift(1)
        daily['TR'] = pd.concat([
            daily['High'] - daily['Low'],
            (daily['High'] - daily['prev_close']).abs(),
            (daily['Low'] - daily['prev_close']).abs(),
        ], axis=1).max(axis=1)
        daily['ATR'] = daily['TR'].rolling(self.atr_window).mean()

        # Prior-day reference close at cash_close time (use last bar at-or-before cash_close)
        cash_close_min = self.cash_close_h * 60 + self.cash_close_m
        bars_min = df.index.hour * 60 + df.index.minute
        cash_close_mask = bars_min == cash_close_min
        ref_by_date = df.loc[cash_close_mask].groupby('Date')['Close'].last()
        ref_by_date = ref_by_date.shift(1)  # prior-day reference

        # Cash open bar (OHLC of the 5m bar starting at cash_open)
        cash_open_min = self.cash_open_h * 60 + self.cash_open_m
        cash_open_mask = bars_min == cash_open_min
        open_by_date_O = df.loc[cash_open_mask].groupby('Date')['Open'].first()
        open_by_date_H = df.loc[cash_open_mask].groupby('Date')['High'].max()
        open_by_date_L = df.loc[cash_open_mask].groupby('Date')['Low'].min()
        open_by_date_C = df.loc[cash_open_mask].groupby('Date')['Close'].last()

        # Broadcast into df as columns keyed by Date
        df['RefPrevClose'] = df['Date'].map(ref_by_date)
        df['CashOpen_O'] = df['Date'].map(open_by_date_O)
        df['CashOpen_H'] = df['Date'].map(open_by_date_H)
        df['CashOpen_L'] = df['Date'].map(open_by_date_L)
        df['CashOpen_C'] = df['Date'].map(open_by_date_C)
        df['Daily_ATR'] = df['Date'].map(daily['ATR'])

    def on_bar(self, idx, df, has_position):
        if has_position:
            return None
        if idx < 1:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        if self._last_signal_date == ts.date():
            return None

        cash_open_min = self.cash_open_h * 60 + self.cash_open_m
        bar_min = ts.hour * 60 + ts.minute
        # We trade ONLY on the first 5m bar AFTER the cash open bar closes — i.e. bar_min == cash_open_min + 5
        if bar_min != cash_open_min + 5:
            return None

        ref_close = bar['RefPrevClose']
        co_o = bar['CashOpen_O']
        co_c = bar['CashOpen_C']
        co_h = bar['CashOpen_H']
        co_l = bar['CashOpen_L']
        atr = bar['Daily_ATR']

        if pd.isna(ref_close) or pd.isna(co_o) or pd.isna(co_c) or pd.isna(atr) or atr <= 0:
            return None

        gap = co_o - ref_close
        min_gap = self.min_gap_atr_frac * atr
        if abs(gap) < min_gap:
            return None

        # Continuation confirmation: cash-open candle closes in gap direction.
        gap_up = gap > 0
        confirmed_up = co_c > co_o
        confirmed_dn = co_c < co_o

        entry = bar['Open']  # we execute at the open of THIS bar (the bar right after open candle)
        if gap_up and confirmed_up:
            sl = co_l - 1.0  # tick below confirmation candle low
            sl_dist = abs(entry - sl)
            if sl_dist <= 0:
                return None
            tp = entry + self.rr * sl_dist
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=entry, sl=sl, tp=tp, tag="gap_up_continue")

        if (not gap_up) and confirmed_dn:
            sl = co_h + 1.0
            sl_dist = abs(sl - entry)
            if sl_dist <= 0:
                return None
            tp = entry - self.rr * sl_dist
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=entry, sl=sl, tp=tp, tag="gap_dn_continue")

        return None
