"""H4: H1 Trend-Following via Donchian Breakout + ATR Trailing Stop.

Thesis: FTMO +10%/30d target is mathematically unfriendly to high-frequency,
low-R strategies. A lower-frequency strategy (5-10 trades/month) with higher
reward:risk (3-5R via trailing stop) can hit +10% on a *single winning streak*
in a trending month, and stay flat when there's no trend.

Mechanics (data is M5, we resample to H1 inside prepare):
  - Compute H1 OHLC.
  - Donchian channel(N=20 on H1) — highest high / lowest low of prior 20 H1 bars.
  - ATR(14) on H1.
  - ADX(14) on H1 — only allow entries when ADX > 25 (trend regime).
  - Signal at H1 close breaking above upper Donchian (BUY) or below lower (SELL).
    We implement it on the M5 level by comparing each M5 close to the CURRENT H1
    Donchian level (which we update only at H1 close to avoid lookahead).
  - SL = entry - 2*ATR(H1) for BUY; mirror for SELL.
  - TP = none fixed — the harness will take SL or EOD. We cannot trail inside the
    current harness (no per-bar SL adjust), so we encode the trail by re-emitting
    the trade at a wider TP: TP = entry + 4*ATR(H1) for BUY. This gives a 1:2 R:R
    realized payoff per trade; over the trend months the winners are big.
  - One signal per day.

Why this is different from ORB: ORB fires *every* session at the open.
Donchian-breakout only fires when H1 price escapes a 20-bar channel — rare,
and typically only during trend regimes. ADX filter gates out chop.

References:
  - Pardo, "Evaluation and Optimization of Trading Strategies" — walk-forward.
  - Tom Basso / turtle rules — Donchian breakout classic.
  - Andreas Clenow "Following the Trend".
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import Strategy, Signal


class H1TrendATR(Strategy):
    name = "h1_trend_atr"

    def __init__(self, donchian_n: int = 20, atr_n: int = 14, adx_n: int = 14,
                 adx_threshold: float = 25.0,
                 sl_atr_mult: float = 2.0, tp_atr_mult: float = 4.0,
                 session_start_h: int = 7, session_end_h: int = 20):
        self.donchian_n = donchian_n
        self.atr_n = atr_n
        self.adx_n = adx_n
        self.adx_threshold = adx_threshold
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.session_start_h = session_start_h
        self.session_end_h = session_end_h
        self._last_signal_date = None

    def prepare(self, df):
        if 'H1_Donchian_High' in df.columns:
            return

        # Resample M5 -> H1 OHLC
        h1 = df[['Open', 'High', 'Low', 'Close']].resample('1h', label='right', closed='right').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()

        # Donchian channels on H1 (shift(1) so current bar isn't included in its own channel)
        h1['Donch_H'] = h1['High'].rolling(self.donchian_n).max().shift(1)
        h1['Donch_L'] = h1['Low'].rolling(self.donchian_n).min().shift(1)

        # ATR(14) on H1
        prev_close = h1['Close'].shift(1)
        tr = pd.concat([
            h1['High'] - h1['Low'],
            (h1['High'] - prev_close).abs(),
            (h1['Low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        h1['ATR'] = tr.rolling(self.atr_n).mean()

        # ADX(14) on H1 — Wilder's smoothing
        up_move = h1['High'] - h1['High'].shift(1)
        down_move = h1['Low'].shift(1) - h1['Low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_adx = tr.rolling(self.adx_n).mean()
        plus_di = 100.0 * pd.Series(plus_dm, index=h1.index).rolling(self.adx_n).mean() / atr_adx
        minus_di = 100.0 * pd.Series(minus_dm, index=h1.index).rolling(self.adx_n).mean() / atr_adx
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        h1['ADX'] = dx.rolling(self.adx_n).mean()

        # Broadcast H1 values forward onto M5 index — each M5 bar reads the LATEST
        # closed H1 bar's values (so no lookahead).
        h1_shift = h1[['Donch_H', 'Donch_L', 'ATR', 'ADX']].copy()
        reindexed = h1_shift.reindex(df.index, method='ffill')
        df['H1_Donchian_High'] = reindexed['Donch_H']
        df['H1_Donchian_Low'] = reindexed['Donch_L']
        df['H1_ATR'] = reindexed['ATR']
        df['H1_ADX'] = reindexed['ADX']

    def on_bar(self, idx, df, has_position):
        if has_position:
            return None
        if idx < 1:
            return None

        bar = df.iloc[idx]
        ts = bar.name

        # Session gate — don't trade dead hours
        if not (self.session_start_h <= ts.hour < self.session_end_h):
            return None
        if self._last_signal_date == ts.date():
            return None

        donch_h = bar['H1_Donchian_High']
        donch_l = bar['H1_Donchian_Low']
        atr = bar['H1_ATR']
        adx = bar['H1_ADX']
        close = bar['Close']

        if pd.isna(donch_h) or pd.isna(donch_l) or pd.isna(atr) or pd.isna(adx):
            return None
        if atr <= 0:
            return None
        if adx < self.adx_threshold:
            return None

        # Breakout logic — current close clears channel AND prior close was inside
        prev_close = df.iloc[idx - 1]['Close']

        if close > donch_h and prev_close <= donch_h:
            sl = close - self.sl_atr_mult * atr
            tp = close + self.tp_atr_mult * atr
            self._last_signal_date = ts.date()
            return Signal(side="BUY", entry=close, sl=sl, tp=tp, tag="donch_break_up")

        if close < donch_l and prev_close >= donch_l:
            sl = close + self.sl_atr_mult * atr
            tp = close - self.tp_atr_mult * atr
            self._last_signal_date = ts.date()
            return Signal(side="SELL", entry=close, sl=sl, tp=tp, tag="donch_break_dn")

        return None
