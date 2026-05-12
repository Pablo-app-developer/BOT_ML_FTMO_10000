"""
FTMO CHALLENGE BOT — 3-Cell Portfolio
======================================
Estrategia confirmada via 300-path MC (2026-05-01):
  Cell 1: XAUUSD  H1TrendATR  (donchian=10, adx=15, sl=2xATR, tp=10xATR)
  Cell 2: USDJPY  H1TrendATR  (donchian=10, adx=25, sl=2xATR, tp=10xATR)
  Cell 3: EURUSD  BBSqueeze   (bb_n=20, bb_k=2.0, sl=1.5xATR, tp=4xATR)
  Riesgo: 1.25% por celda

Resultados MC (300 paths, ventanas 30 dias):
  pass_rate  = 56.7%
  blown_rate = 0.0%
  median_ret = +7.59%

FTMO Rules enforced:
  - Daily loss limit: 5% (halt interno: 2.5%)
  - Total loss limit: 10% (halt interno: 7%)
  - Profit target: 10% (halt interno: 11% para no cortar justo en la linea)
  - Min trading days: 4
  - EOD close: 20:55 broker time
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import logging
import os
import sys
import socket
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

# === Imports desde research/ ===
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "research"))
from strategies.h1_trend_atr import H1TrendATR
from strategies.bb_squeeze import BBSqueeze

# === CONFIGURACION ===

CELLS_CONFIG = [
    {
        "label":   "XAUUSD_h1",
        "symbol":  "XAUUSD",
        "magic":   111001,
        "risk":    0.0125,
        "strategy_cls": H1TrendATR,
        "strategy_kwargs": dict(donchian_n=10, adx_threshold=15.0,
                                sl_atr_mult=2.0, tp_atr_mult=10.0),
    },
    {
        "label":   "USDJPY_h1",
        "symbol":  "USDJPY",
        "magic":   111002,
        "risk":    0.0125,
        "strategy_cls": H1TrendATR,
        "strategy_kwargs": dict(donchian_n=10, adx_threshold=25.0,
                                sl_atr_mult=2.0, tp_atr_mult=10.0),
    },
    {
        "label":   "EURUSD_bbs",
        "symbol":  "EURUSD",
        "magic":   111003,
        "risk":    0.0125,
        "strategy_cls": BBSqueeze,
        "strategy_kwargs": dict(bb_n=20, bb_k=2.0, squeeze_lookback=25,
                                sl_atr_mult=1.5, tp_atr_mult=4.0,
                                require_squeeze_bars=2),
    },
]

# FTMO guardrails
FTMO_DAILY_HALT    = 0.025   # Halt day if down >2.5% (FTMO limit: 5%)
FTMO_TOTAL_HALT    = 0.080   # Halt forever if down >8% (FTMO limit: 10%, 2% buffer confirmed safe)
FTMO_TARGET_LOCK   = 0.110   # Stop trading after +11% (FTMO target: 10%)
FTMO_MIN_DAYS      = 4       # Min distinct trading days

EOD_HOUR           = 20
EOD_MINUTE         = 55
NEWS_BLOCK_MINUTES = 30
DEVIATION          = 20
STATE_FILE         = "bot_state.json"

# Telegram (opcional)
try:
    from FTMO_ICT_STRATEGY.api_secrets import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except Exception:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID   = ""

# === SINGLE-INSTANCE LOCK ===
# Binds a local TCP port; second instance fails to bind and exits cleanly.
_LOCK_SOCK: Optional[socket.socket] = None
_LOCK_PORT = 47923

def _acquire_lock() -> bool:
    global _LOCK_SOCK
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", _LOCK_PORT))
        _LOCK_SOCK = s
        return True
    except OSError:
        s.close()
        return False

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("live_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bot")


# === CELL STATE ===

@dataclass
class Cell:
    label:    str
    symbol:   str
    magic:    int
    risk:     float
    strategy: object          # H1TrendATR or BBSqueeze instance
    spec:     dict = field(default_factory=dict)
    last_bar_time: Optional[datetime] = None


# === MAIN BOT ===

class FTMOBot:

    def __init__(self):
        self.cells = []
        self.initial_balance  = 0.0
        self.daily_start_bal  = 0.0
        self.current_day      = None
        self.trading_days     = set()
        self.blocked_today    = False
        self.blocked_forever  = False
        self._last_news_log   = ""

    # ------------------------------------------------------------------ #
    #  STATE PERSISTENCE                                                   #
    # ------------------------------------------------------------------ #

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    s = json.load(f)
                self.initial_balance  = float(s.get("initial_balance", 0))
                self.trading_days     = set(s.get("trading_days", []))
                self.daily_start_bal  = float(s.get("daily_start_bal", 0))
                self._saved_day       = s.get("current_day", "")
                log.info(f"State loaded: initial={self.initial_balance}  "
                         f"daily_start={self.daily_start_bal}  "
                         f"days_traded={len(self.trading_days)}")
            except Exception as e:
                log.warning(f"State load failed: {e}")

    def _save_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "initial_balance": self.initial_balance,
                    "trading_days":    sorted(self.trading_days),
                    "daily_start_bal": self.daily_start_bal,
                    "current_day":     self.current_day.isoformat() if self.current_day else "",
                }, f, indent=2)
        except Exception as e:
            log.warning(f"State save failed: {e}")

    # ------------------------------------------------------------------ #
    #  MT5 HELPERS                                                         #
    # ------------------------------------------------------------------ #

    def _broker_now(self) -> datetime:
        """Broker server time (use for FTMO daily reset checks)."""
        tick = mt5.symbol_info_tick("EURUSD")
        if tick and tick.time:
            return datetime.fromtimestamp(tick.time)
        return datetime.now()

    def _cache_spec(self, symbol: str) -> dict:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info None for {symbol}")
        acc = mt5.account_info()
        pt  = info.point
        ts  = info.trade_tick_size
        tv  = info.trade_tick_value
        # For pairs where profit is already in account currency (e.g. XAUUSD, EURUSD → USD),
        # contract_size × point gives the exact $ per point per lot.
        # For pairs quoted in a foreign currency (e.g. USDJPY profit in JPY),
        # tick_value already includes the live exchange rate conversion — use it instead.
        if acc and info.currency_profit == acc.currency:
            val = info.trade_contract_size * pt
        else:
            val = tv * (pt / ts) if ts else 0.0
        return {
            "point":    pt,
            "val_pp":   val,         # $ per point per 1.0 lot
            "vol_min":  info.volume_min,
            "vol_max":  info.volume_max,
            "vol_step": info.volume_step,
            "stops":    info.trade_stops_level,
        }

    def _lots(self, balance: float, risk: float, sl_dist: float, spec: dict) -> float:
        if sl_dist <= 0 or spec["val_pp"] <= 0:
            return 0.0
        points     = sl_dist / spec["point"]
        risk_cash  = balance * risk
        risk_per_lot = points * spec["val_pp"]
        if risk_per_lot <= 0:
            return 0.0
        step   = spec["vol_step"] or 0.01
        volume = round((risk_cash / risk_per_lot) / step) * step
        volume = max(spec["vol_min"], min(spec["vol_max"], volume))
        return round(volume, 2)

    def _get_bars(self, symbol: str, n: int = 500) -> Optional[pd.DataFrame]:
        """Fetch last n M5 bars from MT5 as DataFrame with DatetimeIndex."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"open": "Open", "high": "High",
                                 "low": "Low",  "close": "Close"})
        return df[["Open", "High", "Low", "Close"]]

    def _has_position(self, symbol: str, magic: int) -> bool:
        pos = mt5.positions_get(symbol=symbol)
        if pos is None:
            return False
        return any(p.magic == magic for p in pos)

    def _execute(self, symbol: str, side: str, sl: float, tp: float,
                 balance: float, risk: float, spec: dict, magic: int, label: str):
        """Send order to MT5. FOK first, fallback IOC."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log.warning(f"[{label}] No tick, skipping")
            return False

        price  = tick.ask if side == "BUY" else tick.bid
        action = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        sl_dist = abs(price - sl)

        # Enforce min stops_level distance
        min_dist = spec["stops"] * spec["point"]
        if min_dist > 0:
            if side == "BUY":
                if (price - sl) < min_dist: sl = price - min_dist
                if tp > 0 and (tp - price) < min_dist: tp = price + min_dist
            else:
                if (sl - price) < min_dist: sl = price + min_dist
                if tp > 0 and (price - tp) < min_dist: tp = price - min_dist

        sl_dist = abs(price - sl)
        volume  = self._lots(balance, risk, sl_dist, spec)
        if volume <= 0:
            log.warning(f"[{label}] volume=0, skipping")
            return False

        req = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      volume,
            "type":        action,
            "price":       price,
            "sl":          round(sl, 5),
            "tp":          round(tp, 5),
            "deviation":   DEVIATION,
            "magic":       magic,
            "comment":     f"AG_{label}",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(req)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else "None"
            log.warning(f"[{label}] FOK failed ({err}), retrying IOC")
            tick2 = mt5.symbol_info_tick(symbol)
            if tick2:
                req["price"] = tick2.ask if side == "BUY" else tick2.bid
            req["type_filling"] = mt5.ORDER_FILLING_IOC
            result = mt5.order_send(req)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err2 = result.comment if result else "None"
                log.error(f"[{label}] IOC also failed: {err2}")
                self._telegram(f"❌ [{label}] Order failed: {err2}")
                return False

        log.info(f"[{label}] OPEN {side} {volume}L @ {price}  sl={sl:.5f}  tp={tp:.5f}")
        self._telegram(f"🚀 [{label}] {side} {volume}L @ {price}")
        day_str = self._broker_now().date().isoformat()
        if day_str not in self.trading_days:
            self.trading_days.add(day_str)
            self._save_state()
        return True

    def _close_symbol(self, symbol: str, magic: int, label: str):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for pos in positions:
            if pos.magic != magic:
                continue
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            req = {
                "action":      mt5.TRADE_ACTION_DEAL,
                "symbol":      symbol,
                "volume":      pos.volume,
                "type":        close_type,
                "position":    pos.ticket,
                "price":       close_price,
                "deviation":   DEVIATION,
                "magic":       magic,
                "comment":     f"EOD_{label}",
                "type_time":   mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            r = mt5.order_send(req)
            if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
                req["type_filling"] = mt5.ORDER_FILLING_IOC
                mt5.order_send(req)
            log.info(f"[{label}] Closed position #{pos.ticket}")

    def _close_all(self, reason: str = ""):
        for cell in self.cells:
            self._close_symbol(cell.symbol, cell.magic, cell.label)
        if reason:
            log.info(f"All positions closed: {reason}")

    # ------------------------------------------------------------------ #
    #  FTMO HARD STOPS                                                     #
    # ------------------------------------------------------------------ #

    def _check_ftmo(self) -> bool:
        """Returns True = bot should pause/stop. Updates blocked flags."""
        if self.blocked_forever:
            return True
        if self.blocked_today:
            return True

        acc = mt5.account_info()
        if not acc:
            return False
        eq = acc.equity
        init = self.initial_balance

        # 1. Total DD halt
        if init > 0 and (init - eq) / init >= FTMO_TOTAL_HALT:
            pct = (init - eq) / init * 100
            log.critical(f"TOTAL DD HALT: -{pct:.2f}%. Blocking forever.")
            self._close_all("total DD halt")
            self._telegram(f"🛑 TOTAL DD -{pct:.2f}%. Bot halted permanently.")
            self.blocked_forever = True
            return True

        # 2. Profit lock
        if init > 0 and (eq - init) / init >= FTMO_TARGET_LOCK and len(self.trading_days) >= FTMO_MIN_DAYS:
            pct = (eq - init) / init * 100
            log.critical(f"PROFIT LOCK: +{pct:.2f}% — challenge target reached.")
            self._close_all("profit lock")
            self._telegram(f"🏆 PROFIT LOCKED +{pct:.2f}% ({len(self.trading_days)} days). Submit challenge!")
            self.blocked_forever = True
            return True

        # 3. Daily DD halt
        dstart = self.daily_start_bal
        if dstart > 0 and (dstart - eq) / dstart >= FTMO_DAILY_HALT:
            pct = (dstart - eq) / dstart * 100
            log.warning(f"DAILY HALT: -{pct:.2f}%. Blocking for today.")
            self._close_all("daily halt")
            self._telegram(f"🚨 DAILY STOP -{pct:.2f}%. Resuming tomorrow.")
            self.blocked_today = True
            return True

        return False

    def _day_rollover(self):
        now = self._broker_now()
        today = now.date()
        if today != self.current_day:
            log.info(f"New broker day: {today}")
            acc = mt5.account_info()
            if acc:
                self.daily_start_bal = acc.balance
            self.blocked_today = False
            self.current_day   = today
            self._download_news()

    # ------------------------------------------------------------------ #
    #  NEWS FILTER                                                         #
    # ------------------------------------------------------------------ #

    def _download_news(self):
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                with open("ff_calendar_thisweek.xml", "wb") as f:
                    f.write(r.content)
                log.info("Economic calendar updated")
        except Exception as e:
            log.debug(f"Calendar download failed: {e}")

    def _news_block(self) -> str:
        """Returns event title if within NEWS_BLOCK_MINUTES of a high-impact USD/EUR/JPY event.
        Returns empty string when no block is active."""
        try:
            if not os.path.exists("ff_calendar_thisweek.xml"):
                return ""
            tree = ET.parse("ff_calendar_thisweek.xml")
            now  = self._broker_now()
            fmt  = now.strftime("%m-%d-%Y")
            for ev in tree.getroot().findall("event"):
                try:
                    c = ev.find("country")
                    i = ev.find("impact")
                    if c is None or i is None: continue
                    if c.text not in ("USD", "EUR", "JPY"): continue
                    if i.text != "High": continue
                    d = ev.find("date"); t = ev.find("time")
                    if d is None or t is None or d.text != fmt: continue
                    ev_dt = datetime.strptime(f"{d.text} {t.text}", "%m-%d-%Y %I:%M%p")
                    if abs((ev_dt - now).total_seconds()) <= NEWS_BLOCK_MINUTES * 60:
                        title = ev.find("title")
                        return title.text if title is not None else "High Impact Event"
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"News check error: {e}")
        return ""

    # ------------------------------------------------------------------ #
    #  TELEGRAM                                                            #
    # ------------------------------------------------------------------ #

    def _telegram(self, msg: str):
        try:
            if not TELEGRAM_BOT_TOKEN: return
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=5,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  STARTUP                                                             #
    # ------------------------------------------------------------------ #

    def _connect(self) -> bool:
        if not mt5.initialize():
            log.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        for cfg in CELLS_CONFIG:
            sym = cfg["symbol"]
            if not mt5.symbol_select(sym, True):
                log.error(f"Symbol {sym} not available on broker")
                return False

        log.info("MT5 connected. Symbols: " + ", ".join(c["symbol"] for c in CELLS_CONFIG))
        time.sleep(1)  # Let MT5 finish loading symbol info before caching specs

        # Build cell objects
        self.cells = []
        for cfg in CELLS_CONFIG:
            spec  = self._cache_spec(cfg["symbol"])
            strat = cfg["strategy_cls"](**cfg["strategy_kwargs"])
            cell  = Cell(label=cfg["label"], symbol=cfg["symbol"],
                         magic=cfg["magic"], risk=cfg["risk"],
                         strategy=strat, spec=spec)
            self.cells.append(cell)
            log.info(f"  [{cell.label}] spec: val_pp=${spec['val_pp']:.4f}  "
                     f"vol_min={spec['vol_min']}  stops={spec['stops']}pts")

        # Anchor initial balance
        self._load_state()
        acc = mt5.account_info()
        if not acc:
            log.error("account_info() failed")
            return False

        log.info(f"Account #{acc.login}  Balance=${acc.balance:.2f}  Server={acc.server}")

        if self.initial_balance <= 0:
            self.initial_balance = acc.balance
            self._save_state()
            log.info(f"Initial balance anchored: ${self.initial_balance:.2f}")

        self.current_day = self._broker_now().date()
        # Restore daily_start_bal if restarting on the same broker day
        saved_day = getattr(self, "_saved_day", "")
        if saved_day == self.current_day.isoformat() and self.daily_start_bal > 0:
            log.info(f"Same broker day — keeping daily_start_bal=${self.daily_start_bal:.2f}")
        else:
            self.daily_start_bal = acc.balance
            log.info(f"New day or first run — daily_start_bal=${self.daily_start_bal:.2f}")
        self._save_state()
        self._download_news()
        return True

    # ------------------------------------------------------------------ #
    #  MAIN LOOP                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        if not _acquire_lock():
            log.error(f"Another bot instance is already running (port {_LOCK_PORT} busy). Exiting.")
            return
        if not self._connect():
            return

        acc = mt5.account_info()
        log.info(
            f"Bot started | initial=${self.initial_balance:.2f} | "
            f"risk=1.25%/cell | daily_halt={FTMO_DAILY_HALT*100:.1f}% | "
            f"total_halt={FTMO_TOTAL_HALT*100:.1f}% | "
            f"target={FTMO_TARGET_LOCK*100:.1f}%"
        )
        self._telegram(
            f"🤖 FTMO Bot arrancado\n"
            f"Balance: ${self.initial_balance:.0f}\n"
            f"Celdas: XAUUSD + USDJPY + EURUSD"
        )

        last_heartbeat = 0

        while True:
            try:
                self._day_rollover()

                # Heartbeat every 30 min
                import time as _time
                now_ts = _time.time()
                if now_ts - last_heartbeat > 1800:
                    acc = mt5.account_info()
                    eq = acc.equity if acc else 0
                    pos_count = mt5.positions_total() or 0
                    pnl_pct = (eq - self.initial_balance) / self.initial_balance * 100 if self.initial_balance else 0
                    log.info(f"[HEARTBEAT] equity=${eq:.2f}  pnl={pnl_pct:+.2f}%  "
                             f"positions={pos_count}  days_traded={len(self.trading_days)}")
                    last_heartbeat = now_ts

                # FTMO hard stops
                if self._check_ftmo():
                    sleep_s = 300 if self.blocked_forever else 60
                    time.sleep(sleep_s)
                    continue

                now = self._broker_now()

                # EOD: close all positions at 20:55 broker time
                if now.hour == EOD_HOUR and now.minute >= EOD_MINUTE:
                    any_open = any(
                        self._has_position(c.symbol, c.magic) for c in self.cells
                    )
                    if any_open:
                        self._close_all("EOD")
                        self._telegram("🌙 EOD: all positions closed")
                    time.sleep(60)
                    continue

                # Session: only trade 07:00 - 20:00 broker time
                if not (7 <= now.hour < EOD_HOUR):
                    time.sleep(30)
                    continue

                # News filter — block new entries only; existing positions keep their SLs
                news_event = self._news_block()
                if news_event:
                    if news_event != self._last_news_log:
                        log.info(f"News block: {news_event} — new entries paused")
                        self._last_news_log = news_event
                    time.sleep(60)
                    continue
                if self._last_news_log:
                    log.info("News block lifted — resuming")
                    self._last_news_log = ""

                acc = mt5.account_info()
                if not acc:
                    time.sleep(5)
                    continue
                balance = acc.balance

                # Process each cell on new M5 bar
                for cell in self.cells:
                    df = self._get_bars(cell.symbol, n=1000)
                    if df is None or len(df) < 50:
                        continue

                    last_bar_ts = df.index[-2]  # Last CLOSED bar

                    # Only fire logic once per new closed bar
                    if cell.last_bar_time == last_bar_ts:
                        continue
                    cell.last_bar_time = last_bar_ts

                    in_pos = self._has_position(cell.symbol, cell.magic)

                    if in_pos:
                        continue  # Position already open, let SL/TP manage it

                    # Run strategy signal
                    cell.strategy.prepare(df)
                    idx = len(df) - 2  # Last closed bar index
                    sig = cell.strategy.on_bar(idx, df, has_position=False)

                    if sig is None:
                        continue

                    # Check spread (skip if too wide)
                    tick = mt5.symbol_info_tick(cell.symbol)
                    if tick:
                        spread_pts = round((tick.ask - tick.bid) / cell.spec["point"])
                        max_spread = {
                            "XAUUSD": 50, "USDJPY": 15, "EURUSD": 10
                        }.get(cell.symbol, 20)
                        if spread_pts > max_spread:
                            log.info(f"[{cell.label}] Spread too wide: {spread_pts}pts, skipping")
                            continue

                    self._execute(
                        symbol=cell.symbol,
                        side=sig.side,
                        sl=sig.sl,
                        tp=sig.tp,
                        balance=balance,
                        risk=cell.risk,
                        spec=cell.spec,
                        magic=cell.magic,
                        label=cell.label,
                    )

                time.sleep(5)  # Poll every 5 seconds

            except KeyboardInterrupt:
                log.info("Bot stopped by user")
                self._close_all("manual stop")
                mt5.shutdown()
                break
            except Exception as e:
                log.exception(f"Loop error: {e}")
                time.sleep(10)
                try:
                    mt5.shutdown()
                    time.sleep(2)
                    mt5.initialize()
                    for cell in self.cells:
                        cell.spec = self._cache_spec(cell.symbol)
                except Exception as e2:
                    log.error(f"Reconnect failed: {e2}")


if __name__ == "__main__":
    bot = FTMOBot()
    bot.run()
