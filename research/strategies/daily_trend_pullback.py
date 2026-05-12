"""Daily Trend + H1 Pullback Entry.

Thesis: The highest probability trades occur when:
  1. The daily trend is clear (price above/below D1 50 EMA).
  2. Price has pulled back to an H1 key level (H1 EMA or Fibonacci zone).
  3. A reversal candle appears on H1 confirming the end of pullback.

This is the "higher timeframe alignment" approach used by many professional
prop traders. It results in fewer but higher-quality entries (15-25/month).

Mechanics:
  - D1 trend: H1 50 EMA slope (proxy for daily trend, computed on H1 so we
    can efficiently detect direction in M5 data).
  - Entry zone: price between H1 EMA(21) and H1 EMA(50) = pullback zone.
  - Trigger: H1 bar closes back above H1 EMA(21) after touching EMA(50) → entry.
  - ADX(14) on H1 > 20 to confirm trend is active.
  - SL: below H1 EMA(50) minus 0.5*ATR (for BUY).
  - TP: sl_dist * rr_ratio.
  - Session: London + NY only (7-20 UTC).
  - One trade per day.
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal


class DailyTrendPullback(Strategy):
    name = "daily_trend_pullback"

    def __init__(self,
                 ema_fast: int = 21,
                 ema_slow: int = 50,
                 adx_n: int = 14,
                 adx_min: float = 20.0,
                 atr_n: int = 14,
                 sl_atr_mult: float = 1.0,
                 rr_ratio: float = 3.0,
                 session_start_h: int = 7,
                 session_end_h: int = 20):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_n = adx_n
        self.adx_min = adx_min
        self.atr_n = atr_n
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio
        self.session_start_h = session_start_h
        self.session_end_h = session_end_h
        self._last_signal_date = None

    def prepare(self, df: pd.DataFrame):
        if 'H1_EMA21' in df.columns:
            return

        h1 = df[['Open', 'High', 'Low', 'Close']].resample(
            '1h', label='right', closed='right'
        ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

        h1['EMA21'] = h1['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        h1['EMA50'] = h1['Close'].ewm(span=self.ema_slow, adjust=False).mean()

        # ATR + ADX on H1
        prev_c = h1['Close'].shift(1)
        tr = pd.concat([
            h1['High'] - h1['Low'],
            (h1['High'] - prev_c).abs(),
            (h1['Low'] - prev_c).abs(),
        ], axis=1).max(axis=1)
        h1['ATR'] = tr.rolling(self.atr_n).mean()

        up_move = h1['High'].diff()
        dn_move = -h1['Low'].diff()
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        atr_adx = tr.rolling(self.adx_n).mean().replace(0, np.nan)
        plus_di = 100 * pd.Series(plus_dm, index=h1.index).rolling(self.adx_n).mean() / atr_adx
        minus_di = 100 * pd.Series(minus_dm, index=h1.index).rolling(self.adx_n).mean() / atr_adx
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        h1['ADX'] = dx.rolling(self.adx_n).mean()

        # Trend direction: EMA50 slope over last 5 H1 bars
        h1['EMA50_slope'] = h1['EMA50'].diff(5)

        cols = ['EMA21', 'EMA50', 'ATR', 'ADX', 'EMA50_slope']
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

        e21 = bar['H1_EMA21']
        e50 = bar['H1_EMA50']
        atr = bar['H1_ATR']
        adx = bar['H1_ADX']
        slope = bar['H1_EMA50_slope']
        close = bar['Close']
        prev_close = df.iloc[idx - 1]['Close']
        prev_e21 = df.iloc[idx - 1]['H1_EMA21']

        if any(pd.isna(v) for v in [e21, e50, atr, adx, slope]):
            return None
        if atr <= 0:
            return None
        if adx < self.adx_min:
            return None

        sl_dist = self.sl_atr_mult * atr
        tp_dist = self.rr_ratio * sl_dist

        # Uptrend: EMA50 slope positive, price was below EMA21, now crosses back above
        if slope > 0 and prev_close < prev_e21 and close > e21:
            # Pullback entry: price came down to EMA zone and recovered
            sl = e50 - sl_dist
            if close <= sl:
                return None
            tp = close + tp_dist
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=close, sl=sl, tp=tp, tag="dtp_bull")

        # Downtrend: EMA50 slope negative, price was above EMA21, now crosses back below
        if slope < 0 and prev_close > prev_e21 and close < e21:
            sl = e50 + sl_dist
            if close >= sl:
                return None
            tp = close - tp_dist
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=close, sl=sl, tp=tp, tag="dtp_bear")

        return None
