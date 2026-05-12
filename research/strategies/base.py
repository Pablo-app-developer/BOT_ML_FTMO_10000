"""Abstract Strategy interface — all candidate edges implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class Signal:
    side: Literal["BUY", "SELL"]
    entry: float      # Intended entry price (market order executes at next bar open in backtest)
    sl: float         # Absolute stop-loss price
    tp: float         # Absolute take-profit price
    tag: str = ""     # Optional label for analytics (e.g. "asian_high_break")


class Strategy(ABC):
    """Strategies are pure signal generators. They do NOT manage risk or position state.
    The harness feeds them bars one at a time and asks for a signal. The harness handles
    everything else (sizing, fills, trailing, FTMO rules, EOD close).
    """

    name: str = "base"

    @abstractmethod
    def prepare(self, df):
        """One-time precompute (indicators, session markers). Attach columns to df in-place
        or store cached arrays. Called once before the bar loop starts.
        """

    @abstractmethod
    def on_bar(self, idx: int, df, has_position: bool) -> Optional[Signal]:
        """Return a Signal to open a new trade, or None.
        Only called when has_position is False (harness enforces one-position-at-a-time).
        `idx` is the current bar; the signal executes at the OPEN of idx+1 (no lookahead).
        """
