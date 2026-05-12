"""H3: Z-Score Mean Reversion in Low-Volatility Regimes.

Thesis: In low-volatility regimes, short-term price deviations mean-revert.
In high-volatility regimes they trend. A z-score of recent returns, combined with
a volatility filter (current ATR below its rolling median), gives a filtered signal:
fade extreme deviations only when volatility says "no trend right now."

Rules:
  - Compute 20-bar return z-score.
  - Compute ATR(14) and its 100-bar rolling median.
  - Only signal when current ATR < median (low-vol regime).
  - SHORT when z > 2, LONG when z < -2.
  - SL = N*ATR from entry. TP = entry + RR * SL_distance (reversion target).
  - One trade per day.

References: Ernest Chan "Algorithmic Trading: Winning Strategies", cap 2.
QuantConnect's "Bollinger Bands Mean Reversion" research piece.
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import Strategy, Signal


class ZScoreMeanReversion(Strategy):
    name = "zscore_meanreversion"

    def __init__(self, z_window: int = 20, z_threshold: float = 2.0,
                 atr_window: int = 14, regime_window: int = 100,
                 atr_sl_mult: float = 1.5, rr_ratio: float = 1.2,
                 session_start_h: int = 0, session_end_h: int = 7):
        """
        Trade only during the specified session hours (default: Asian = low-vol window).
        atr_sl_mult: SL distance = atr * this.
        rr_ratio: TP distance = sl_distance * this (mean revert = tighter targets).
        """
        self.z_window = z_window
        self.z_threshold = z_threshold
        self.atr_window = atr_window
        self.regime_window = regime_window
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.session_start_h = session_start_h
        self.session_end_h = session_end_h
        self._last_signal_date = None

    def prepare(self, df):
        if 'ZScore' in df.columns:
            return
        ret = df['Close'].pct_change()
        df['ZScore'] = (ret - ret.rolling(self.z_window).mean()) / ret.rolling(self.z_window).std()

        # ATR
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.atr_window).mean()
        df['ATR_Median'] = df['ATR'].rolling(self.regime_window).median()

    def on_bar(self, idx, df, has_position):
        if has_position:
            return None
        if idx < max(self.z_window, self.regime_window) + 1:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        if not (self.session_start_h <= ts.hour < self.session_end_h):
            return None
        if self._last_signal_date == ts.date():
            return None

        z = bar['ZScore']
        atr = bar['ATR']
        atr_med = bar['ATR_Median']

        if pd.isna(z) or pd.isna(atr) or pd.isna(atr_med):
            return None
        # Regime filter: only low-vol
        if atr >= atr_med:
            return None

        close = bar['Close']
        sl_dist = atr * self.atr_sl_mult
        if sl_dist <= 0:
            return None

        if z >= self.z_threshold:
            entry = close
            sl = entry + sl_dist
            tp = entry - self.rr_ratio * sl_dist
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=entry, sl=sl, tp=tp, tag="z_extreme_high")
        if z <= -self.z_threshold:
            entry = close
            sl = entry - sl_dist
            tp = entry + self.rr_ratio * sl_dist
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=entry, sl=sl, tp=tp, tag="z_extreme_low")

        return None
