"""EMA Momentum Burst — M15-frequency trend-following.

Thesis: When multiple EMAs align and short-term momentum confirms, a
trend move is more likely to continue. This targets 30-50 signals/month
(much higher frequency than H1 Donchian), which mathematically helps
reach FTMO's 10% target within 30 days.

Mechanics (data is M5, resampled to M15 internally):
  - EMA(8) > EMA(21) > EMA(50) on M15 → bullish alignment (reverse for bear).
  - Last 2 consecutive M15 closes are positive (momentum confirmation).
  - ATR(14) on M15 is expanding (current ATR > 10-period median ATR).
  - Entry: on next M5 bar after signal.
  - SL: entry - sl_atr_mult * ATR(M15) for BUY.
  - TP: entry + tp_atr_mult * ATR(M15).
  - Max 1 trade per session (London: 7-12, NY: 13-20). Resets each session.
  - Session filter avoids dead hours (Asian for EUR pairs).

Why higher frequency matters:
  At 40 trades/month, 55% WR, 2:1 RR, 0.5% risk:
  E = 0.55*(1%) - 0.45*(0.5%) = 0.325%/trade * 40 = 13%/month expected.
  FTMO is mathematically achievable.
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal


class EMAMomentum(Strategy):
    name = "ema_momentum"

    def __init__(self,
                 ema_fast: int = 8,
                 ema_mid: int = 21,
                 ema_slow: int = 50,
                 atr_n: int = 14,
                 atr_expansion_bars: int = 10,
                 sl_atr_mult: float = 1.5,
                 tp_atr_mult: float = 3.0,
                 session_start_h: int = 7,
                 session_end_h: int = 20):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.atr_n = atr_n
        self.atr_expansion_bars = atr_expansion_bars
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.session_start_h = session_start_h
        self.session_end_h = session_end_h
        self._last_signal_session = None  # (date, session_name)

    def prepare(self, df: pd.DataFrame):
        if 'M15_EMA8' in df.columns:
            return

        # Resample to M15
        m15 = df[['Open', 'High', 'Low', 'Close']].resample(
            '15min', label='right', closed='right'
        ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

        # EMAs
        m15['EMA8'] = m15['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        m15['EMA21'] = m15['Close'].ewm(span=self.ema_mid, adjust=False).mean()
        m15['EMA50'] = m15['Close'].ewm(span=self.ema_slow, adjust=False).mean()

        # ATR on M15
        prev_c = m15['Close'].shift(1)
        tr = pd.concat([
            m15['High'] - m15['Low'],
            (m15['High'] - prev_c).abs(),
            (m15['Low'] - prev_c).abs(),
        ], axis=1).max(axis=1)
        m15['ATR'] = tr.rolling(self.atr_n).mean()
        m15['ATR_median'] = m15['ATR'].rolling(self.atr_expansion_bars).median()

        # Momentum: consecutive positive M15 closes
        m15['close_change'] = m15['Close'].diff()
        m15['consec_up'] = (
            (m15['close_change'] > 0).astype(int)
            .groupby((m15['close_change'] <= 0).cumsum())
            .cumsum()
        )
        m15['consec_dn'] = (
            (m15['close_change'] < 0).astype(int)
            .groupby((m15['close_change'] >= 0).cumsum())
            .cumsum()
        )

        # Broadcast M15 → M5 (ffill so each M5 bar gets latest closed M15)
        to_broadcast = ['EMA8', 'EMA21', 'EMA50', 'ATR', 'ATR_median', 'consec_up', 'consec_dn']
        reindexed = m15[to_broadcast].reindex(df.index, method='ffill')
        for col in to_broadcast:
            df[f'M15_{col}'] = reindexed[col]

    def on_bar(self, idx: int, df: pd.DataFrame, has_position: bool) -> Signal:
        if has_position:
            return None
        if idx < 3:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        if not (self.session_start_h <= ts.hour < self.session_end_h):
            return None

        # Session key — one trade per session per day
        session = 'london' if ts.hour < 13 else 'ny'
        session_key = (ts.date(), session)
        if self._last_signal_session == session_key:
            return None

        e8 = bar['M15_EMA8']
        e21 = bar['M15_EMA21']
        e50 = bar['M15_EMA50']
        atr = bar['M15_ATR']
        atr_med = bar['M15_ATR_median']
        cup = bar['M15_consec_up']
        cdn = bar['M15_consec_dn']
        close = bar['Close']

        if any(pd.isna(v) for v in [e8, e21, e50, atr, atr_med]):
            return None
        if atr <= 0:
            return None
        # ATR expansion filter
        if atr < atr_med:
            return None

        sl_dist = self.sl_atr_mult * atr
        tp_dist = self.tp_atr_mult * atr

        # Bullish alignment
        if e8 > e21 > e50 and cup >= 2:
            sl = close - sl_dist
            tp = close + tp_dist
            self._last_signal_session = session_key
            return Signal(side="BUY", entry=close, sl=sl, tp=tp, tag="ema_bull")

        # Bearish alignment
        if e8 < e21 < e50 and cdn >= 2:
            sl = close + sl_dist
            tp = close - tp_dist
            self._last_signal_session = session_key
            return Signal(side="SELL", entry=close, sl=sl, tp=tp, tag="ema_bear")

        return None
