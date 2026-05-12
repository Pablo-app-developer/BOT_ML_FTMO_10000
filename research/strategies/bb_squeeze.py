"""Bollinger Band Squeeze Breakout.

Thesis: Volatility is mean-reverting. When Bollinger Bands contract to their
narrowest width in N bars (a "squeeze"), the market is coiling for a move.
The first decisive close outside the bands after a squeeze tends to be the
start of a sustained directional move.

Mechanics (on H1, data is M5 resampled internally):
  - Bollinger Bands(N, k_std) on H1 Close.
  - BB Width = (upper - lower) / middle.
  - Squeeze = True when current BB Width < rolling minimum of last squeeze_lookback bars.
  - Entry: on first H1 close ABOVE upper band (after squeeze) → BUY.
             on first H1 close BELOW lower band (after squeeze) → SELL.
  - Require squeeze to have been active in last 3 H1 bars.
  - SL: entry ± sl_atr_mult * ATR(14, H1).
  - TP: entry ± tp_atr_mult * ATR(14, H1).
  - One trade per day.
  - Session filter: session_start_h to session_end_h.

Why this is different from standard ORB:
  ORB fires at a fixed time every session. Squeeze breakout fires only when the
  market has SPECIFICALLY contracted before expanding. This dramatically reduces
  false breakouts and focuses on high-probability move initiations.

References: John Carter "Mastering the Trade", chapter on TTM Squeeze.
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal


class BBSqueeze(Strategy):
    name = "bb_squeeze"

    def __init__(self,
                 bb_n: int = 20,
                 bb_k: float = 2.0,
                 squeeze_lookback: int = 20,
                 atr_n: int = 14,
                 sl_atr_mult: float = 1.5,
                 tp_atr_mult: float = 3.0,
                 require_squeeze_bars: int = 2,
                 session_start_h: int = 7,
                 session_end_h: int = 20):
        self.bb_n = bb_n
        self.bb_k = bb_k
        self.squeeze_lookback = squeeze_lookback
        self.atr_n = atr_n
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.require_squeeze_bars = require_squeeze_bars
        self.session_start_h = session_start_h
        self.session_end_h = session_end_h
        self._last_signal_date = None

    def prepare(self, df: pd.DataFrame):
        if 'H1_BB_Upper' in df.columns:
            return

        h1 = df[['Open', 'High', 'Low', 'Close']].resample(
            '1h', label='right', closed='right'
        ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

        # Bollinger Bands on H1 Close
        mid = h1['Close'].rolling(self.bb_n).mean()
        std = h1['Close'].rolling(self.bb_n).std()
        h1['BB_Upper'] = mid + self.bb_k * std
        h1['BB_Lower'] = mid - self.bb_k * std
        h1['BB_Mid'] = mid
        h1['BB_Width'] = (h1['BB_Upper'] - h1['BB_Lower']) / mid.replace(0, np.nan)

        # Squeeze: current width < rolling min of last N bars (shift 1 to avoid lookahead)
        h1['BB_Width_Min'] = h1['BB_Width'].rolling(self.squeeze_lookback).min().shift(1)
        h1['Squeeze'] = (h1['BB_Width'] < h1['BB_Width_Min']).astype(int)
        # Count consecutive squeeze bars
        h1['Squeeze_Count'] = (
            h1['Squeeze']
            .groupby((h1['Squeeze'] == 0).cumsum())
            .cumsum()
        )

        # ATR on H1
        prev_c = h1['Close'].shift(1)
        tr = pd.concat([
            h1['High'] - h1['Low'],
            (h1['High'] - prev_c).abs(),
            (h1['Low'] - prev_c).abs(),
        ], axis=1).max(axis=1)
        h1['ATR'] = tr.rolling(self.atr_n).mean()

        # Broadcast to M5
        cols = ['BB_Upper', 'BB_Lower', 'BB_Mid', 'BB_Width', 'Squeeze_Count', 'ATR']
        reindexed = h1[cols].reindex(df.index, method='ffill')
        for col in cols:
            df[f'H1_{col}'] = reindexed[col]

    def on_bar(self, idx: int, df: pd.DataFrame, has_position: bool) -> Signal:
        if has_position:
            return None
        if idx < 2:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        if not (self.session_start_h <= ts.hour < self.session_end_h):
            return None
        if self._last_signal_date == ts.date():
            return None

        bb_upper = bar['H1_BB_Upper']
        bb_lower = bar['H1_BB_Lower']
        atr = bar['H1_ATR']
        sq_count = bar['H1_Squeeze_Count']
        close = bar['Close']
        prev_close = df.iloc[idx - 1]['Close']

        if any(pd.isna(v) for v in [bb_upper, bb_lower, atr, sq_count]):
            return None
        if atr <= 0:
            return None

        # Require squeeze to have been active recently
        if sq_count < self.require_squeeze_bars:
            return None

        sl_dist = self.sl_atr_mult * atr
        tp_dist = self.tp_atr_mult * atr

        # Break above upper band
        if close > bb_upper and prev_close <= bb_upper:
            sl = close - sl_dist
            tp = close + tp_dist
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=close, sl=sl, tp=tp, tag="bb_squeeze_up")

        # Break below lower band
        if close < bb_lower and prev_close >= bb_lower:
            sl = close + sl_dist
            tp = close - tp_dist
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=close, sl=sl, tp=tp, tag="bb_squeeze_dn")

        return None
