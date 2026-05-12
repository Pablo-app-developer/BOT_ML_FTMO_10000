"""Impulse-Pullback Scalp — high frequency intraday continuation.

Thesis: After a strong momentum impulse candle (large body relative to ATR),
price often pulls back briefly before continuing in the impulse direction.
Trading the continuation after the pullback gives high-quality entries with
a defined invalidation point.

Mechanics (M5 bars):
  - Define impulse: bar body >= impulse_atr_mult * ATR(14) AND close near
    high/low (body proportion >= body_frac, e.g. 70% of range).
  - After a bull impulse, wait for price to retrace into the lower half of
    the impulse candle (close <= impulse_mid) over the next pullback_bars.
  - Entry on the bar that shows a bullish close back above impulse_mid.
  - SL = impulse candle's low - 0.2*ATR buffer.
  - TP = impulse candle's high + rr_ratio * SL_distance.
  - Session filter: only trade during active hours (session_start_h to session_end_h).
  - Max one trade per impulse (resets after TP/SL).
  - Cooldown: must see N bars without another impulse before looking again.

Why this gives more signals:
  XAUUSD typically has 4-8 strong impulse candles per day during London/NY.
  With this strategy we target 2-4 trades per day = 40-80 per month.
  Even with 50% WR and 2:1 RR, at 0.5% risk:
    E = 0.5*1% - 0.5*0.5% = 0.25%/trade * 60 trades = 15%/month expected.

Instruments: XAUUSD, US30, DE40 (volatile, trending intraday)
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal


class ImpulsePullback(Strategy):
    name = "impulse_pb"

    def __init__(self,
                 impulse_atr_mult: float = 1.2,
                 body_frac: float = 0.60,
                 pullback_bars: int = 6,
                 rr_ratio: float = 2.0,
                 atr_n: int = 14,
                 session_start_h: int = 7,
                 session_end_h: int = 20,
                 cooldown_bars: int = 3):
        self.impulse_atr_mult = impulse_atr_mult
        self.body_frac = body_frac
        self.pullback_bars = pullback_bars
        self.rr_ratio = rr_ratio
        self.atr_n = atr_n
        self.session_start_h = session_start_h
        self.session_end_h = session_end_h
        self.cooldown_bars = cooldown_bars

        self._impulse_idx = None   # Index of the last impulse bar
        self._impulse_high = None
        self._impulse_low = None
        self._impulse_mid = None
        self._impulse_dir = None   # +1 bull, -1 bear
        self._pullback_seen = False
        self._last_trade_idx = -999

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

    def _reset_setup(self):
        self._impulse_idx = None
        self._impulse_dir = None
        self._pullback_seen = False

    def on_bar(self, idx: int, df: pd.DataFrame, has_position: bool) -> Signal:
        if has_position:
            # Trade is on — reset any pending setup so we don't enter mid-trade
            self._reset_setup()
            return None
        if idx < self.atr_n + 2:
            return None

        bar = df.iloc[idx]
        ts = bar.name
        if not (self.session_start_h <= ts.hour < self.session_end_h):
            self._reset_setup()
            return None

        atr = bar['ATR']
        if pd.isna(atr) or atr <= 0:
            return None

        open_ = bar['Open']
        high = bar['High']
        low = bar['Low']
        close = bar['Close']
        body = abs(close - open_)
        candle_range = high - low

        # --- Phase 1: Look for new impulse candle ---
        # Cooldown check
        if idx - self._last_trade_idx < self.cooldown_bars:
            return None

        is_impulse = (
            candle_range > 0
            and body >= self.impulse_atr_mult * atr
            and (body / candle_range) >= self.body_frac
        )

        if is_impulse:
            # Bull impulse: close near high
            if close > open_ and (close - low) / candle_range >= 0.6:
                self._impulse_idx = idx
                self._impulse_high = high
                self._impulse_low = low
                self._impulse_mid = (high + low) / 2
                self._impulse_dir = +1
                self._pullback_seen = False
                return None

            # Bear impulse: close near low
            if close < open_ and (high - close) / candle_range >= 0.6:
                self._impulse_idx = idx
                self._impulse_high = high
                self._impulse_low = low
                self._impulse_mid = (high + low) / 2
                self._impulse_dir = -1
                self._pullback_seen = False
                return None

        # --- Phase 2: Wait for pullback and entry ---
        if self._impulse_idx is None:
            return None

        bars_since = idx - self._impulse_idx
        if bars_since > self.pullback_bars:
            # Setup expired
            self._reset_setup()
            return None

        if self._impulse_dir == +1:
            # Bull setup: need pullback below mid then recovery above mid
            if not self._pullback_seen:
                if close <= self._impulse_mid:
                    self._pullback_seen = True
                return None
            # Pullback seen — entry when close pops back above mid
            if close > self._impulse_mid:
                sl = self._impulse_low - 0.2 * atr
                sl_dist = abs(close - sl)
                if sl_dist <= 0:
                    self._reset_setup()
                    return None
                tp = close + self.rr_ratio * sl_dist
                self._last_trade_idx = idx
                self._reset_setup()
                return Signal(side="BUY", entry=close, sl=sl, tp=tp, tag="imp_pb_bull")

        elif self._impulse_dir == -1:
            # Bear setup: need pullback above mid then recovery below mid
            if not self._pullback_seen:
                if close >= self._impulse_mid:
                    self._pullback_seen = True
                return None
            if close < self._impulse_mid:
                sl = self._impulse_high + 0.2 * atr
                sl_dist = abs(sl - close)
                if sl_dist <= 0:
                    self._reset_setup()
                    return None
                tp = close - self.rr_ratio * sl_dist
                self._last_trade_idx = idx
                self._reset_setup()
                return Signal(side="SELL", entry=close, sl=sl, tp=tp, tag="imp_pb_bear")

        return None
