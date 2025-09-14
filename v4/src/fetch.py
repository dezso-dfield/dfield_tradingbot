"""
Fetch market data from free/public sources with timeouts and safe limits.
"""
import requests
import pandas as pd
from typing import Optional
from .constants import COINGECKO, BINANCE

REQ_TIMEOUT = (5, 20)   # (connect, read) seconds

def ohlcv_daily_coingecko(coin_id: str, days: int = 180) -> pd.DataFrame:
    r = requests.get(
        f"{COINGECKO}/coins/{coin_id}/market_chart",
        params={"vs_currency":"usd","days":days,"interval":"daily"},
        timeout=REQ_TIMEOUT
    )
    r.raise_for_status()
    dat = r.json()
    prices = pd.DataFrame(dat["prices"], columns=["ts","price"])
    volumes = pd.DataFrame(dat["total_volumes"], columns=["ts","volume"])
    df = prices.merge(volumes, on="ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.rename(columns={"price":"close"}, inplace=True)
    # Approximate OHLC around close
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df["close"].rolling(3, min_periods=1).max()
    df["low"]  = df["close"].rolling(3, min_periods=1).min()
    df = df[["ts","open","high","low","close","volume"]]
    return df.dropna().reset_index(drop=True)

# --- Add to fetch.py ---
def binance_spot_usdt_symbols(timeout=(5,20)):
    """
    Return a set of spot symbols that have a USDT quote and are trading-enabled.
    Uses /api/v3/exchangeInfo (no key). Filters out leveraged/margin-only.
    """
    url = f"{BINANCE}/api/v3/exchangeInfo"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        info = r.json()
    except requests.RequestException:
        return set()
    out = set()
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING": 
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        # Spot filter
        if not s.get("isSpotTradingAllowed", True):
            continue
        out.add(s.get("symbol"))
    return out

def binance_klines(symbol_usdt: str, interval: str = "4h", limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch true OHLCV from Binance public API. symbol_usdt like 'BTCUSDT'.
    Clamp limit to 1000 and use timeouts to avoid hanging.
    """
    url = f"{BINANCE}/api/v3/klines"
    safe_limit = max(1, min(int(limit), 1000))
    params = {"symbol": symbol_usdt, "interval": interval, "limit": safe_limit}
    try:
        r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
    except requests.RequestException:
        return None

    if not data:
        return None

    rows = []
    for k in data:
        rows.append({
            "ts": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return pd.DataFrame(rows)

def binance_last_price(symbol_usdt: str, timeout=(5,20)) -> Optional[float]:
    """
    Get the latest traded price for a USDT symbol (e.g., BTCUSDT).
    """
    url = f"{BINANCE}/api/v3/ticker/price"
    try:
        r = requests.get(url, params={"symbol": symbol_usdt}, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        return float(data.get("price"))
    except requests.RequestException:
        return None
    except (TypeError, ValueError):
        return None