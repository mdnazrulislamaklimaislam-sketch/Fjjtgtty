# -*- coding: utf-8 -*-
import os
import re
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Optional, Any
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np
from dotenv import load_dotenv
from flask import Flask
from telegram import Update as TGUpdate, Bot as TGBot
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from supabase import create_client, Client

# ===================== env & logging =====================
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
CHAT_ID_RAW = os.getenv("CHAT_ID", "").strip()

if not BOT_TOKEN or not SUPABASE_URL or not SUPABASE_KEY or not CHAT_ID_RAW:
    raise EnvironmentError("Missing env: BOT_TOKEN/SUPABASE_URL/SUPABASE_KEY/CHAT_ID")

try:
    CHAT_ID: Any = int(CHAT_ID_RAW)
except Exception:
    CHAT_ID = CHAT_ID_RAW

BINANCE_BASE = "https://api.binance.com"
BINANCE_BATCH_SIZE = int(os.getenv("BINANCE_BATCH_SIZE", "15"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
SLEEP_TIME = float(os.getenv("SLEEP_TIME", "6"))
ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))
USER_TZ = ZoneInfo(os.getenv("USER_TZ", "Asia/Dhaka"))  # UTC+6
AUTO_ADD_NEW_COINS = os.getenv("AUTO_ADD_NEW_COINS", "true").lower() in ("1","true","yes")

aiohttp_session: Optional[aiohttp.ClientSession] = None
binance_sem: Optional[asyncio.Semaphore] = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("crypto-scanner")

bot = TGBot(token=BOT_TOKEN)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===================== flask keep alive =====================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Bot is alive!"

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

def keep_alive():
    asyncio.get_event_loop().run_in_executor(None, run_flask)

# ===================== Exceptions =====================
class RateLimitError(Exception):
    def __init__(self, status: int, retry_after: Optional[int] = None, msg: Optional[str] = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(msg or f"Rate limit: {status}")

# ===================== Supabase async wrappers =====================
async def _to_thread(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)

async def get_column(table: str, column: str = "coin") -> List[str]:
    try:
        def query():
            res = supabase.table(table).select(column).execute()
            return res.data or []
        rows = await _to_thread(query)
        return [r[column].upper() for r in rows if r.get(column)]
    except Exception:
        logger.exception("Supabase get_column error")
        return []

async def add_coin(table: str, coin: str) -> None:
    coin_up = coin.upper()
    existing = await get_column(table)
    if coin_up in existing:
        return
    try:
        def insert():
            return supabase.table(table).insert({"coin": coin_up}).execute()
        await _to_thread(insert)
        logger.info("Added %s to %s", coin_up, table)
    except Exception:
        logger.exception("Supabase add_coin error")

async def add_coin_with_date(table: str, coin: str) -> None:
    coin_up = coin.upper()
    existing = await get_column(table)
    if coin_up in existing:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        def insert():
            return supabase.table(table).insert({"coin": coin_up, "timestamp": timestamp}).execute()
        await _to_thread(insert)
        logger.info("Added %s to %s with timestamp", coin_up, table)
    except Exception:
        logger.exception("Supabase add_coin_with_date error")

async def remove_coin_from_table(table: str, coin: str) -> bool:
    coin_up = coin.upper()
    try:
        def delete_fn():
            resp = supabase.table(table).select("id, coin").eq("coin", coin_up).execute()
            rows = resp.data or []
            for r in rows:
                supabase.table(table).delete().eq("id", r["id"]).execute()
            return bool(rows)
        return await _to_thread(delete_fn)
    except Exception:
        logger.exception("Supabase remove_coin_from_table error")
        return False

async def get_removed_map() -> Dict[str, str]:
    try:
        def query():
            res = supabase.table("removed").select("coin, timestamp").execute()
            return res.data or []
        rows = await _to_thread(query)
        return {r["coin"].upper(): r.get("timestamp") for r in rows if r.get("coin")}
    except Exception:
        logger.exception("Supabase get_removed_map error")
        return {}

async def log_to_supabase(symbol: str, reasons_text: str, opinion_text: Optional[str]) -> None:
    coin = symbol.upper()
    ts_iso = datetime.now(timezone.utc).isoformat()
    payload = {"coin": coin, "timestamp": ts_iso, "reasons": reasons_text}
    if opinion_text is not None:
        payload["opinion"] = opinion_text
    try:
        def insert():
            return supabase.table("signals").insert(payload).execute()
        await _to_thread(insert)
        logger.info("Logged to supabase: %s | %s", coin, reasons_text)
    except Exception:
        logger.exception("Supabase log error for %s", coin)

async def fetch_signals_since(since_iso: str) -> List[Dict[str, Any]]:
    try:
        def query():
            return supabase.table("signals").select("*").gte("timestamp", since_iso).execute().data or []
        return await _to_thread(query)
    except Exception:
        logger.exception("Supabase fetch_signals_since error")
        return []

# ===================== Binance requests =====================
async def binance_request(path: str, params: Optional[dict] = None) -> Tuple[Any, Optional[str]]:
    global aiohttp_session
    if aiohttp_session is None:
        raise RuntimeError("HTTP session is not initialized")
    url = f"{BINANCE_BASE}{path}"
    params = params or {}
    async with aiohttp_session.get(url, params=params) as resp:
        used_weight = resp.headers.get("X-MBX-USED-WEIGHT-1M") or resp.headers.get("X-MBX-USED-WEIGHT")
        if resp.status == 429:
            retry = resp.headers.get("Retry-After")
            body = await resp.text()
            raise RateLimitError(429, retry_after=int(retry) if retry and retry.isdigit() else None, msg=f"429 {body}")
        if resp.status != 200:
            body = await resp.text()
            raise Exception(f"HTTP {resp.status}: {body}")
        data = await resp.json(content_type=None)
        return data, used_weight

async def get_exchange_info():
    data, used = await binance_request("/api/v3/exchangeInfo")
    if used:
        logger.debug("ExchangeInfo weight: %s", used)
    return data

async def get_klines(symbol: str, interval: str, limit: int = 150):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data, used = await binance_request("/api/v3/klines", params=params)
    return data

async def get_ticker_24h(symbol: str) -> Optional[float]:
    try:
        data, _ = await binance_request("/api/v3/ticker/24hr", {"symbol": symbol})
        pct = float(data.get("priceChangePercent", 0.0))
        return pct
    except Exception:
        logger.exception("24h ticker error for %s", symbol)
        return None

async def get_top_gainers(n=10) -> List[Tuple[str, float]]:
    exinfo = await get_exchange_info()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    ]
    global binance_sem
    # ---------- FIX: Ensure binance_sem is always initialized before use ----------
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def get_pct(symbol):
        try:
            # FIX: Ensure binance_sem is not None before using in async with
            if binance_sem is None:
                raise RuntimeError("binance_sem not initialized")
            async with binance_sem:
                pct = await get_ticker_24h(symbol)
                await asyncio.sleep(0.3)  # rate limit
            return (symbol, pct)
        except Exception:
            return (symbol, None)

    tasks = [get_pct(sym) for sym in symbols]
    all_results = await asyncio.gather(*tasks)
    filtered = [(sym, pct) for sym, pct in all_results if pct is not None]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:n]

# ===================== Numeric helpers =====================
def np_safe(arr: List[float]) -> np.ndarray:
    return np.array(arr, dtype=float) if arr else np.array([], dtype=float)

def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0

def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0

def rsi_series(closes: List[float], length: int = 14) -> np.ndarray:
    closes_np = np_safe(closes)
    if closes_np.size < length + 1:
        return np.full(closes_np.size, 50.0)
    delta = np.diff(closes_np)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    roll_up = np.convolve(gains, np.ones(length, dtype=float), 'valid') / length
    roll_down = np.convolve(losses, np.ones(length, dtype=float), 'valid') / length
    rs = np.divide(roll_up, roll_down, out=np.full_like(roll_up, np.nan), where=roll_down != 0)
    rsi = 100 - (100 / (1 + rs))
    pad = np.full(closes_np.size - rsi.size, 50.0)
    return np.concatenate([pad, rsi])

def atr_series(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> np.ndarray:
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    closes_np = np_safe(closes)
    if closes_np.size < length + 1:
        return np.full(closes_np.size, 0.0)
    prev_close = np.concatenate([[closes_np[0]], closes_np[:-1]])
    tr = np.maximum(highs_np - lows_np, np.maximum(np.abs(highs_np - prev_close), np.abs(lows_np - prev_close)))
    atr = np.convolve(tr, np.ones(length, dtype=float), 'valid') / length
    pad = np.full(tr.size - atr.size, atr[0] if atr.size else 0.0)
    return np.concatenate([pad, atr])

def ema_series(values: List[float], length: int) -> np.ndarray:
    values_np = np_safe(values)
    if values_np.size == 0:
        return values_np
    alpha = 2 / (length + 1)
    out = np.empty_like(values_np)
    out[0] = values_np[0]
    for i in range(1, values_np.size):
        out[i] = alpha * values_np[i] + (1 - alpha) * out[i - 1]
    return out

def cvd_proxy(closes: List[float], volumes: List[float]) -> np.ndarray:
    closes_np = np_safe(closes)
    volumes_np = np_safe(volumes)
    if closes_np.size < 2 or volumes_np.size != closes_np.size:
        return np.cumsum(np.zeros_like(closes_np))
    delta = np.diff(closes_np)
    sign = np.sign(delta)
    sign = np.concatenate([[0.0], sign])
    delta_vol = sign * volumes_np
    return np.cumsum(delta_vol)

def percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else 0.0

def sell_side_liquidity_sweep_bullish(highs, lows, opens, closes, lookback=20):
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if lows_np.size < lookback + 2:
        return False
    prior_low = float(np.min(lows_np[-(lookback+1):-1]))
    sweep = (lows_np[-1] < prior_low) and (closes_np[-1] > prior_low) and (closes_np[-1] > opens_np[-1])
    return bool(sweep)

def displacement_bullish(highs, lows, opens, closes, atr, body_ratio=0.6, atr_mult=1.2):
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if highs_np.size < 2 or atr.size != highs_np.size:
        return False
    rng = highs_np[-1] - lows_np[-1]
    body = closes_np[-1] - opens_np[-1]
    if rng <= 0:
        return False
    cond = (closes_np[-1] > opens_np[-1]) and (rng > atr_mult * atr[-1]) and ((body / rng) >= body_ratio)
    return bool(cond)

def bullish_rsi_divergence(closes: List[float], rsi: np.ndarray, lookback=20) -> bool:
    closes_np = np_safe(closes)
    if closes_np.size < lookback + 5 or rsi.size != closes_np.size:
        return False
    window = closes_np[-(lookback+5):]
    if window.size < 6:
        return False
    rel_idx = np.argsort(window)[:6]
    chosen = []
    for idx in rel_idx:
        abs_idx = idx + (closes_np.size - window.size)
        if not chosen or abs(abs_idx - chosen[-1]) > 2:
            chosen.append(int(abs_idx))
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        return False
    i1, i2 = chosen[0], chosen[1]
    price_ll = closes_np[i1] < closes_np[i2]
    rsi_hl = rsi[i1] > rsi[i2]
    return bool(price_ll and rsi_hl)

def whale_entry(volumes: List[float], closes: List[float], factor=3.0):
    volumes_np = np_safe(volumes)
    closes_np = np_safe(closes)
    if volumes_np.size < 20 or closes_np.size < 2:
        return False
    last = volumes_np[-1]
    mean = float(np.mean(volumes_np[-20:]))
    return bool((last > mean * factor) and (closes_np[-1] > closes_np[-2]))

def cvd_imbalance_up(cvd: np.ndarray, bars=5, mult=1.6):
    if cvd.size < bars + 1:
        return False
    slope = cvd[-1] - cvd[-bars]
    ref_seg = np.diff(cvd[:-bars]) if cvd.size >= bars + 11 else np.diff(cvd)
    ref_std = safe_std(ref_seg)
    if ref_std == 0:
        return bool(slope > 0)
    return bool(slope > mult * ref_std)

def volatility_metric(closes: List[float], win=30):
    closes_np = np_safe(closes)
    if closes_np.size < win:
        return 0.0
    seg = closes_np[-win:]
    mu = float(np.mean(seg))
    return float(np.std(seg) / mu) if mu else 0.0

def find_bullish_fvg_indices(highs: List[float], lows: List[float]) -> List[int]:
    out = []
    for n in range(2, len(highs)):
        if lows[n] > highs[n-2]:
            out.append(n)
    return out

def last_fvg_zone(highs: List[float], lows: List[float]):
    idxs = find_bullish_fvg_indices(highs, lows)
    if not idxs:
        return None
    n = idxs[-1]
    gap_top = lows[n]
    gap_bottom = highs[n-2]
    return (float(gap_top), float(gap_bottom), int(n))

def bullish_fvg_alert_logic(opens, highs, lows, closes, volumes, tf_label: str):
    opens_np = np_safe(opens)
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    closes_np = np_safe(closes)
    volumes_np = np_safe(volumes)
    if closes_np.size < 60:
        return None
    zone = last_fvg_zone(highs_np.tolist(), lows_np.tolist())
    if not zone:
        return None
    gap_top, gap_bottom, idx_fvg = zone
    start = idx_fvg + 1
    if start + 3 >= closes_np.size:
        return None
    inside_idx = None
    for i in range(start, closes_np.size - 1):
        body_low = float(min(opens_np[i], closes_np[i]))
        body_high = float(max(opens_np[i], closes_np[i]))
        if (gap_bottom <= body_low) and (body_high <= gap_top):
            inside_idx = i
            break
    if inside_idx is None:
        return None
    cvd = cvd_proxy(closes_np.tolist(), volumes_np.tolist())
    ref_slice = volumes_np[max(0, inside_idx-20):inside_idx]
    ref_mean = float(np.mean(ref_slice)) if ref_slice.size else 0.0
    vol_rise = bool(volumes_np[inside_idx] > 1.25 * ref_mean)
    cvd_slice = cvd[max(0, inside_idx-20):inside_idx+1]
    cvd_diff = np.diff(cvd_slice) if cvd_slice.size >= 2 else np.array([])
    cvd_std = safe_std(cvd_diff)
    cvd_rise = bool((cvd[inside_idx] - cvd[max(0, inside_idx-5)]) > 1.5 * cvd_std) if cvd_std > 0 else bool(cvd[inside_idx] - cvd[max(0, inside_idx-5)] > 0)
    if not (vol_rise or cvd_rise):
        return None
    for j in range(inside_idx + 1, closes_np.size):
        if (closes_np[j] > gap_top) and (closes_np[j] > opens_np[j]):
            return f"Bullish FVG Confirmed ({tf_label})"
    return None

# ===================== Data fetch & parse =====================
INTERVALS_CORE = ["15m", "1h", "4h"]
INTERVALS_FVG  = ["1h", "4h", "1d"]

async def fetch_intervals(symbol: str, limits: Dict[str, int]):
    async def _one(iv: str, lim: int):
        try:
            data = await get_klines(symbol, iv, lim)
            return iv, data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Klines error %s %s: %s", symbol, iv, e)
            return iv, []
    tfs = set(INTERVALS_CORE + INTERVALS_FVG + ["1d"])
    tasks = [_one(iv, limits.get(iv, 240)) for iv in tfs]
    res = await asyncio.gather(*tasks)
    return {iv: data if isinstance(data, list) else [] for iv, data in res}

def parse_ohlcv(candles: list):
    if not candles:
        return [], [], [], [], [], []
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows  = [float(c[3]) for c in candles]
    closes= [float(c[4]) for c in candles]
    volumes=[float(c[5]) for c in candles]
    times = [int(c[0]) for c in candles]
    return opens, highs, lows, closes, volumes, times

# ===================== Risk filters & cooldown =====================
LAST_ALERT_AT: Dict[str, datetime] = {}

def cooldown_ok(symbol: str) -> bool:
    now = datetime.now(timezone.utc)
    last = LAST_ALERT_AT.get(symbol)
    return True if last is None else (now - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def mark_alert(symbol: str):
    LAST_ALERT_AT[symbol] = datetime.now(timezone.utc)

def htf_bullish_bias(closes_1h: List[float], closes_4h: List[float]) -> bool:
    if len(closes_1h) < 60:
        return False
    ema1h = ema_series(closes_1h, 50)
    cond_1h = (closes_1h[-1] > ema1h[-1]) and (ema1h[-1] > ema1h[-6])
    cond_4h = False
    if len(closes_4h) >= 60:
        ema4h = ema_series(closes_4h, 50)
        cond_4h = (closes_4h[-1] > ema4h[-1]) and (ema4h[-1] > ema4h[-6])
    return bool(cond_1h or cond_4h)

def atr_vol_gate(highs_15, lows_15, closes_15):
    atr15 = atr_series(highs_15, lows_15, closes_15, 14)
    atr_slice = atr15[-100:] if atr15.size >= 100 else atr15
    if atr_slice.size < 20:
        return False
    cur_atr = float(atr15[-1])
    p40 = percentile(atr_slice, 40)
    return bool(cur_atr >= p40 and cur_atr > 0)

# ===================== Historical analysis =====================
def simulate_signal_success(closes: List[float], highs: List[float], lows: List[float], atr: np.ndarray, idx: int, horizon: int = 6, atr_tp_mult: float = 0.8) -> bool:
    if idx >= len(closes) - 1:
        return False
    entry = closes[idx]
    tp = entry + float(atr[idx]) * atr_tp_mult if atr.size > idx else entry * 1.005
    end = min(len(closes)-1, idx + horizon)
    return bool(np.max(np_safe(highs[idx+1:end+1])) >= tp)

def historical_strength(tf_name: str, opens, highs, lows, closes, volumes, lookback_bars: int = 220) -> Tuple[float, int]:
    atr = atr_series(highs, lows, closes, 14)
    rsi = rsi_series(closes, 14)
    cvd = cvd_proxy(closes, volumes)
    signal_indices: List[int] = []
    for i in range(max(30, len(closes)-lookback_bars), len(closes)-1):
        conds = []
        conds.append(displacement_bullish(highs[:i+1], lows[:i+1], opens[:i+1], closes[:i+1], atr[:i+1], 0.6, 1.2))
        conds.append(sell_side_liquidity_sweep_bullish(highs[:i+1], lows[:i+1], opens[:i+1], closes[:i+1], 20))
        conds.append(bullish_rsi_divergence(closes[:i+1], rsi[:i+1], 20))
        conds.append(cvd_imbalance_up(cvd[:i+1], 5, 1.6))
        if sum(1 for x in conds if x) >= 2:
            signal_indices.append(i)
    wins = 0
    for idx in signal_indices[-100:]:
        if simulate_signal_success(closes, highs, lows, atr, idx, horizon=6, atr_tp_mult=0.8):
            wins += 1
    total = len(signal_indices[-100:])
    hit_rate = (wins / total * 100.0) if total > 0 else 0.0
    return hit_rate, total

def opinion_from_scores(scores: Dict[str, float]) -> Tuple[str, float]:
    w = {"15m": 0.4, "1h": 0.35, "4h": 0.25}
    num = sum(scores.get(tf, 0.0) * w[tf] for tf in w)
    den = sum(w.values())
    agg = num / den if den else 0.0
    if agg >= 70:
        verdict = "🔥 Strong"
    elif agg >= 55:
        verdict = "✅ Moderate"
    elif agg > 0:
        verdict = "🟡 Weak"
    else:
        verdict = "⚪ No edge"
    return verdict, agg

# ===================== Signal detection (MTF; no 5m) =====================
async def detect_signals(symbol: str) -> Tuple[Dict[str, List[str]], Optional[str], Dict[str, float]]:
    limits = {"15m": 240, "1h": 240, "4h": 240, "1d": 240}
    data_map = await fetch_intervals(symbol, limits)

    if any(len(data_map.get(iv, [])) == 0 for iv in ["15m", "1h"]):
        return {}, None, {}

    o15, h15, l15, c15, v15, _ = parse_ohlcv(data_map["15m"])
    o1h, h1h, l1h, c1h, v1h, _ = parse_ohlcv(data_map["1h"])
    o4h, h4h, l4h, c4h, v4h, _ = parse_ohlcv(data_map.get("4h", []))
    o1d, h1d, l1d, c1d, v1d, _ = parse_ohlcv(data_map.get("1d", []))

    if not htf_bullish_bias(c1h, c4h):
        return {}, None, {}
    if not atr_vol_gate(h15, l15, c15):
        return {}, None, {}

    atr15 = atr_series(h15, l15, c15, 14)
    atr1h = atr_series(h1h, l1h, c1h, 14)
    rsi15 = rsi_series(c15, 14)
    rsi1h = rsi_series(c1h, 14)
    cvd15 = cvd_proxy(c15, v15)
    cvd1h = cvd_proxy(c1h, v1h)

    reasons_by_tf: Dict[str, List[str]] = {"15m": [], "1h": [], "4h": [], "1d": []}

    if cvd_imbalance_up(cvd15, bars=5, mult=1.6):
        reasons_by_tf["15m"].append("CVD Imbalance Up")
    if whale_entry(v15, c15, factor=3.0):
        reasons_by_tf["15m"].append("Whale Entry")
    if sell_side_liquidity_sweep_bullish(h15, l15, o15, c15, 20):
        reasons_by_tf["15m"].append("SSL Sweep")
    if displacement_bullish(h15, l15, o15, c15, atr15, 0.6, 1.2):
        reasons_by_tf["15m"].append("Smart Money Entry (Displacement)")
    if bullish_rsi_divergence(c15, rsi15, 20):
        reasons_by_tf["15m"].append("RSI Bullish Div")
    if volatility_metric(c15, 30) > 0.5:
        reasons_by_tf["15m"].append("High Volatility")

    if cvd_imbalance_up(cvd1h, bars=3, mult=1.4):
        reasons_by_tf["1h"].append("CVD Imbalance Up")
    if sell_side_liquidity_sweep_bullish(h1h, l1h, o1h, c1h, 20):
        reasons_by_tf["1h"].append("SSL Sweep")
    if displacement_bullish(h1h, l1h, o1h, c1h, atr1h, 0.55, 1.15):
        reasons_by_tf["1h"].append("Smart Money Entry")
    if bullish_rsi_divergence(c1h, rsi1h, 25):
        reasons_by_tf["1h"].append("RSI Bullish Div")

    for tf, candles in [("1h", data_map.get("1h", [])), ("4h", data_map.get("4h", [])), ("1d", data_map.get("1d", []))]:
        if not candles:
            continue
        o, h, lows, c, v, _ = parse_ohlcv(candles)
        alert = bullish_fvg_alert_logic(o, h, lows, c, v, tf.upper())
        if alert:
            reasons_by_tf[tf].append("Bullish FVG Confirmed")

    gate = (
        (len(reasons_by_tf["15m"]) >= 2) or
        (len(reasons_by_tf["1h"]) >= 2) or
        any("Bullish FVG Confirmed" in x for x in reasons_by_tf["1h"] + reasons_by_tf["4h"] + reasons_by_tf["1d"])
    )
    if not gate or not cooldown_ok(symbol):
        return {}, None, {}

    scores: Dict[str, float] = {}
    try:
        if len(c15) > 80:
            s15, _ = historical_strength("15m", o15, h15, l15, c15, v15)
            scores["15m"] = s15
        if len(c1h) > 80:
            s1h, _ = historical_strength("1h", o1h, h1h, l1h, c1h, v1h)
            scores["1h"] = s1h
        if len(c4h) > 80:
            s4h, _ = historical_strength("4h", o4h, h4h, l4h, c4h, v4h)
            scores["4h"] = s4h
    except Exception:
        logger.exception("Historical analysis error")
        scores = {tf: 0.0 for tf in ["15m", "1h", "4h"]}

    verdict, agg = opinion_from_scores(scores)
    final_text = f"{verdict} (hist hit-rate: {round(agg):.0f}%). " \
                 f"15m: {round(scores.get('15m',0)):.0f}% | 1h: {round(scores.get('1h',0)):.0f}% | 4h: {round(scores.get('4h',0)):.0f}%"

    return reasons_by_tf, final_text, scores

# ===================== Alerts & send =====================
async def format_status(symbol: str) -> str:
    watchlist = await get_column("watchlist")
    haram = await get_column("haram")
    removed_map = await get_removed_map()
    s = symbol.upper()
    if s in haram:
        return "HARAM ⚠️"
    if s in removed_map:
        return f"REMOVED@{removed_map[s]}"
    if s in watchlist:
        return "WATCHLIST ✅"
    return "NEW 🚀"

def now_local_str() -> str:
    return datetime.now(USER_TZ).strftime("%Y-%m-%d %H:%M %Z")

async def send_alert(symbol: str, reasons_by_tf: Dict[str, List[str]], final_opinion: str):
    timestamp = now_local_str()
    status_text = await format_status(symbol)
    pct24 = await get_ticker_24h(symbol)
    pct_str = f"{pct24:+.2f}%" if pct24 is not None else "N/A"

    def tf_line(tf: str) -> str:
        r = reasons_by_tf.get(tf, [])
        return f"{tf}: " + (" + ".join(r) if r else "No Signal")

    message = (
        "🚨 Bullish Signal Detected!\n"
        f"Coin: `{symbol}`\n"
        f"Status: {status_text}\n"
        f"Time: {timestamp}\n"
        f"24h Change: {pct_str}\n"
        "---------------------------------\n"
        f"{tf_line('15m')}\n"
        f"{tf_line('1h')}\n"
        f"{tf_line('4h')}\n"
        f"{tf_line('1d')}\n"
        "---------------------------------\n"
        f"{final_opinion}"
    )

    reasons_flat = []
    for tf in ["15m","1h","4h","1d"]:
        for r in reasons_by_tf.get(tf, []):
            reasons_flat.append(f"{tf}: {r}")
    reasons_text = " | ".join(reasons_flat) if reasons_flat else "No Signal"
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
        mark_alert(symbol)
        await log_to_supabase(symbol, reasons_text, final_opinion)
        logger.info("Telegram alert sent for %s", symbol)
    except Exception:
        logger.exception("Telegram error sending alert for %s", symbol)

# ===================== Scanning & batching =====================
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def scan_one_with_backoff(coin):
    max_attempts = 5
    delay = 1
    for attempt in range(max_attempts):
        try:
            reasons_by_tf, final_opinion, _ = await detect_signals(coin)
            if reasons_by_tf:
                await send_alert(coin, reasons_by_tf, final_opinion or "⚪ No edge")
            await asyncio.sleep(SLEEP_TIME)
            break
        except RateLimitError as e:
            ra = e.retry_after or delay
            logger.warning("Rate limit for %s. Retry after %s s (attempt %s)", coin, ra, attempt+1)
            await asyncio.sleep(ra)
            delay = min(delay * 2, 60)
        except Exception:
            logger.exception("Error with %s", coin)
            break

async def auto_add_new_listings(symbols: List[str]):
    if not AUTO_ADD_NEW_COINS:
        return
    watchlist = await get_column("watchlist")
    haram = await get_column("haram")
    removed = await get_removed_map()
    for s in symbols:
        if s in haram or s in removed or s in watchlist:
            continue
        await add_coin_with_date("watchlist", s)

async def batch_scan_all_with_rate_limit():
    global binance_sem
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    exinfo = await get_exchange_info()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
    ]
    symbols = [c for c in symbols if not any(x in c for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])]
    await auto_add_new_listings(symbols)
    logger.info("Total %s USDT spot symbols. Scanning in batches of %s ...", len(symbols), BINANCE_BATCH_SIZE)
    for batch in chunked(symbols, BINANCE_BATCH_SIZE):
        logger.info("Scanning batch of %s coins...", len(batch))
        async def limited_scan(coin):
            if binance_sem is None:
                raise RuntimeError("Semaphore not initialized")
            async with binance_sem:
                await scan_one_with_backoff(coin)
        tasks = [limited_scan(coin) for coin in batch]
        await asyncio.gather(*tasks)
        logger.info("Sleeping 60 seconds for next batch...")
        await asyncio.sleep(60)

async def scanner_loop():
    while True:
        try:
            await batch_scan_all_with_rate_limit()
        except Exception:
            logger.exception("Scanner loop error")
            logger.info("Sleeping 10 minutes...")
            await asyncio.sleep(600)

# ===================== Aggregator (custom times) =====================
LAST_REPORT_SENT: Dict[str, datetime] = {}

AGG_TIMES_LOCAL = [
    (6, 0),
    (10, 0),
    (14, 0),
    (18, 0),
    (22, 0),
]

def _is_agg_time(dt: datetime) -> Optional[str]:
    for hh, mm in AGG_TIMES_LOCAL:
        if dt.hour == hh and dt.minute == mm:
            return dt.strftime("%Y-%m-%d %H:%M")
    return None

async def aggregator_loop():
    global LAST_REPORT_SENT
    while True:
        try:
            now_local = datetime.now(USER_TZ)
            agg_time_key = _is_agg_time(now_local)
            if agg_time_key:
                if agg_time_key in LAST_REPORT_SENT and (datetime.now(USER_TZ) - LAST_REPORT_SENT[agg_time_key]).total_seconds() < 3600:
                    await asyncio.sleep(20)
                    continue
                window_end_local = now_local.replace(second=0, microsecond=0)
                since_utc = (window_end_local.astimezone(timezone.utc) - timedelta(hours=4))
                since_iso = since_utc.isoformat()
                rows = await fetch_signals_since(since_iso)
                if rows:
                    watchlist = await get_column("watchlist")
                    haram = await get_column("haram")
                    removed_map = await get_removed_map()
                    grouped: Dict[str, List[Dict[str, Any]]] = {}
                    for r in rows:
                        coin = r.get("coin", "").upper()
                        grouped.setdefault(coin, []).append(r)
                    lines = [f"📋 Aggregated Signals (`{window_end_local.strftime('%Y-%m-%d %H:%M %Z')}`)", ""]
                    for coin, items in grouped.items():
                        if coin in haram:
                            st = "HARAM ⚠️"
                        elif coin in removed_map:
                            st = f"REMOVED@{removed_map[coin]}"
                        elif coin in watchlist:
                            st = "WATCHLIST ✅"
                        else:
                            st = "NEW 🚀"
                        reasons_set = []
                        for it in items:
                            rtxt = it.get("reasons", "")
                            if rtxt and rtxt not in reasons_set:
                                reasons_set.append(rtxt)
                        opinions = [it.get("opinion") for it in items if it.get("opinion")]
                        opinion_text = opinions[-1] if opinions else None
                        pct24 = await get_ticker_24h(coin)
                        pct_str = f"{pct24:+.2f}%" if pct24 is not None else "N/A"
                        lines.append(f"- `{coin}`: {st}")
                        lines.append("   Reasons: " + (" | ".join(reasons_set) if reasons_set else "No Signal"))
                        if opinion_text:
                            lines.append(f"   Final Opinion: {opinion_text}")
                        lines.append(f"   24h Change: {pct_str}")
                    message = "\n".join(lines)
                    try:
                        await bot.send_message(chat_id=CHAT_ID, text=message)
                        logger.info("Aggregator message sent for window %s", window_end_local.isoformat())
                    except Exception:
                        logger.exception("Telegram send error in aggregator")
                LAST_REPORT_SENT[agg_time_key] = now_local
                await asyncio.sleep(70)
            else:
                await asyncio.sleep(20)
        except Exception:
            logger.exception("Aggregator loop error")
            await asyncio.sleep(30)

# ===================== NEW: Top Gainer Command Handler =====================
async def top_gainer_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    await update.message.reply_text("⏳ Fetching top gainers, please wait...")
    try:
        gainers = await get_top_gainers(10)
        lines = ["🔥 Top 10 Binance USDT Spot Gainers (24h):"]
        for i, (coin, pct) in enumerate(gainers, 1):
            lines.append(f"{i}. `{coin}` : {pct:+.2f}%")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to fetch: {e}")

# ===================== Telegram commands =====================
def parse_command(text: str) -> Tuple[List[str], str]:
    if not text:
        return [], ""
    clean = text.strip()
    tokens = re.split(r'[\s,;]+', clean)
    actions_map = {
        "check": "Check",
        "remove": "Remove",
        "haram": "Haram",
        "add": "Add again",
        "addagain": "Add again",
        "add_again": "Add again",
        "halal": "Halal",
    }
    found_action = ""
    coins = []
    for t in tokens:
        tl = t.lower()
        if tl in actions_map:
            found_action = actions_map[tl]
            continue
        m = re.match(r'^([A-Za-z0-9]{2,10})(?:[/\-]?USDT)?$', t.upper())
        if m:
            base = m.group(1)
            coin = base if base.endswith("USDT") else f"{base}USDT"
            coins.append(coin.upper())
    return coins, found_action

def get_username(update: TGUpdate) -> str:
    user = update.effective_user
    return user.username or str(user.id) if user else "Unknown"

def is_admin(username: Optional[str]) -> bool:
    return username == "RedwanICT"  # change as needed

async def start(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text(
            f"⛔ Sorry @{username},\nyou do not have permission to use the Crypto Watchlist Tracking Bot."
        )
        return
    await update.message.reply_text(
        f"👋 Hello @{username}! Send coin commands like:\n"
        "BTCUSDT Check\nETH Remove\nBNB Haram\nBTC Add\nBNB Halal\nOr send: top gainer list"
    )

async def status(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return

    wl_total = len(watchlist)
    hr_total = len(haram)

    parts = []
    parts.append(f"📊 Watchlist ({wl_total}):")
    parts.extend([f"`{w}`" for w in watchlist] if watchlist else ["—"])
    parts.append("")
    parts.append(f"⚠️ Haram ({hr_total}):")
    parts.extend([f"`{h}`" for h in haram] if haram else ["—"])

    reply = "\n".join(parts)
    await update.message.reply_text(reply)

async def handle_commands(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if text.lower() == "top gainer list":
        await top_gainer_handler(update, context)
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    logger.info("Command from %s: %s", username, text)
    coins, action = parse_command(text)
    valid_actions = {"Check", "Remove", "Haram", "Add again", "Halal"}
    if not coins or action not in valid_actions:
        await update.message.reply_text(
            "❌ Invalid format. Use keywords like:\nCheck, Remove, Haram, Add Again, Halal\nExample: BTC ETH Check\nOr send: top gainer list"
        )
        return
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
        removed_map = await get_removed_map()
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return
    already, added, removed, marked_haram, unharamed = [], [], [], [], []
    for coin in coins:
        if action == "Check":
            if coin in haram:
                marked_haram.append(coin)
            elif coin in removed_map:
                removed.append(f"{coin} - {removed_map[coin]}")
            elif coin in watchlist:
                already.append(coin)
            else:
                await add_coin_with_date("watchlist", coin)
                added.append(coin)
        elif action == "Remove":
            if await remove_coin_from_table("watchlist", coin):
                await add_coin_with_date("removed", coin)
                removed.append(coin)
        elif action == "Haram":
            await add_coin("haram", coin)
            marked_haram.append(coin)
            if coin in watchlist and await remove_coin_from_table("watchlist", coin):
                await add_coin_with_date("removed", coin)
                removed.append(f"{coin} (removed from watchlist due to haram)")
        elif action == "Add again":
            if coin in removed_map:
                await add_coin_with_date("watchlist", coin)
                await remove_coin_from_table("removed", coin)
                added.append(coin)
        elif action == "Halal":
            if await remove_coin_from_table("haram", coin):
                unharamed.append(coin)

    reply_parts = []
    if already:
        reply_parts.append("🟢 Already in Watchlist:\n" + "\n".join(f"`{x}`" for x in already))
    if added:
        reply_parts.append("✅ New Added:\n" + "\n".join(f"`{x}`" for x in added))
    if marked_haram:
        reply_parts.append("⚠️ Marked as Haram:\n" + "\n".join(f"`{x}`" for x in marked_haram))
    if removed:
        reply_parts.append("🗑️ Removed:\n" + "\n".join(f"`{x}`" for x in removed))
    if unharamed:
        reply_parts.append("✅ Removed from Haram:\n" + "\n".join(f"`{x}`" for x in unharamed))
    reply = "\n\n".join(reply_parts) or "✅ No changes made."
    await update.message.reply_text(reply)

async def post_init(application):
    global aiohttp_session, binance_sem
    aiohttp_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, ssl=False),
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        headers={"User-Agent": "CryptoWatchlistBot/1.3 (+contact)"}
    )
    binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    application.create_task(scanner_loop())
    application.create_task(aggregator_loop())
    logger.info("post_init completed: aiohttp session created and background tasks started.")

def main() -> None:
    keep_alive()
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"(?i)^status$"), status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_commands))
    logger.info("Bot starting (press Ctrl+C to stop)...")
    app.run_polling()

if __name__ == "__main__":
    main()
