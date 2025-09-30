"""
Comprehensive indicators (robust, no-NaNs in outputs):
- Robust BB width (fallback if pandas_ta names differ)
- RVOL(20), ADV USD(20), ATR%
- Candlestick pattern analysis (engulfing, hammer/star, doji, inside, marubozu)
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from .schema import IndicatorPacket


# ---------- Bollinger safe ----------
def _bb_width_safe(close: pd.Series, length: int = 20, stdev: float = 2.0) -> pd.Series:
    """
    Try pandas_ta.bbands first; if the expected columns aren't present or are NaN,
    fall back to a manual rolling-std calculation. Always returns a float Series.
    """
    close = pd.to_numeric(close, errors="coerce")
    bb = ta.bbands(close, length=length, std=stdev)
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        upper = None
        lower = None
        for u_name in (f"BBU_{length}_{stdev}", f"BBU_{length}_{float(stdev)}", "BBU_20_2.0"):
            if u_name in bb.columns:
                upper = pd.to_numeric(bb[u_name], errors="coerce")
                break
        for l_name in (f"BBL_{length}_{stdev}", f"BBL_{length}_{float(stdev)}", "BBL_20_2.0"):
            if l_name in bb.columns:
                lower = pd.to_numeric(bb[l_name], errors="coerce")
                break
        if upper is not None and lower is not None:
            out = (upper - lower) / close.replace(0, pd.NA)
            return pd.to_numeric(out, errors="coerce").fillna(0.0)

    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std()
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    out = (upper - lower) / close.replace(0, pd.NA)
    return pd.to_numeric(out, errors="coerce").fillna(0.0)


# ---------- Candlestick detectors ----------
def _candle_stats(o, h, l, c):
    o = pd.to_numeric(o, errors="coerce")
    h = pd.to_numeric(h, errors="coerce")
    l = pd.to_numeric(l, errors="coerce")
    c = pd.to_numeric(c, errors="coerce")
    body = (c - o).abs()
    rng = (h - l).abs().replace(0, 1e-9)
    upper = (h - o.where(o > c, c))
    lower = (o.where(o < c, c) - l)
    return body, rng, upper, lower


def _is_doji(body, rng, thresh=0.1):
    return (body / rng) <= thresh


def _is_marubozu(upper, lower, rng, max_frac=0.1):
    return (upper / rng <= max_frac) & (lower / rng <= max_frac)


def _is_hammer(body, upper, lower, factor=2.0, max_upper_frac=0.3):
    return (lower >= factor * body) & (upper <= max_upper_frac * body)


def _is_shooting_star(body, upper, lower, factor=2.0, max_lower_frac=0.3):
    return (upper >= factor * body) & (lower <= max_lower_frac * body)


def _is_inside_bar(h, l):
    prev_h, prev_l = h.shift(1), l.shift(1)
    return (h <= prev_h) & (l >= prev_l)


def _engulfing(o, c):
    prev_o, prev_c = o.shift(1), c.shift(1)
    cur_up = c > o
    prev_down = prev_c < prev_o
    bull = cur_up & prev_down & (o <= prev_c) & (c >= prev_o)

    cur_down = c < o
    prev_up = prev_c > prev_o
    bear = cur_down & prev_up & (o >= prev_c) & (c <= prev_o)
    return bull, bear


def _candlestick_signal(df: pd.DataFrame):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body, rng, upper, lower = _candle_stats(o, h, l, c)

    doji = _is_doji(body, rng)
    maru = _is_marubozu(upper, lower, rng)
    hammer = _is_hammer(body, upper, lower)
    star = _is_shooting_star(body, upper, lower)
    inside = _is_inside_bar(h, l)
    bull_eng, bear_eng = _engulfing(o, c)

    s = 0.0
    label = []
    if bool(bull_eng.iloc[-1]):
        s += 3.0; label.append("Bullish Engulfing")
    if bool(bear_eng.iloc[-1]):
        s -= 3.0; label.append("Bearish Engulfing")
    if bool(hammer.iloc[-1]):
        s += 2.0; label.append("Hammer")
    if bool(star.iloc[-1]):
        s -= 2.0; label.append("Shooting Star")
    if bool(doji.iloc[-1]):
        label.append("Doji")
    if bool(maru.iloc[-1]):
        label.append("Marubozu")
    if bool(inside.iloc[-1]):
        label.append("Inside Bar")

    text = ", ".join(label) if label else "None"
    return text, float(s)


# ---------- Main compute ----------
def compute_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> IndicatorPacket:
    """
    Compute all indicators and return an IndicatorPacket with numeric (non-None) fields.
    Any NaNs are filled to reasonable defaults so downstream math never compares against None.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)

    # Ensure numeric dtypes early to avoid downcasting warnings later
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Momentum / trend / vol
    df["rsi"] = pd.to_numeric(ta.rsi(df["close"], length=14), errors="coerce").fillna(50.0)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"] = pd.to_numeric(macd.iloc[:, 0], errors="coerce").fillna(0.0)
        df["macd_signal"] = pd.to_numeric(macd.iloc[:, 1], errors="coerce").fillna(0.0)
        df["macd_hist"] = pd.to_numeric(macd.iloc[:, 2], errors="coerce").fillna(0.0)
    else:
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["macd_hist"] = 0.0

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = pd.to_numeric(stoch.iloc[:, 0], errors="coerce").fillna(50.0)
        df["stoch_d"] = pd.to_numeric(stoch.iloc[:, 1], errors="coerce").fillna(50.0)
    else:
        df["stoch_k"] = 50.0
        df["stoch_d"] = 50.0

    adx = ta.adx(df["high"], df["low"], df["close"])
    if adx is not None and not adx.empty:
        df["adx"] = pd.to_numeric(adx.iloc[:, 0], errors="coerce").fillna(20.0)
    else:
        df["adx"] = 20.0

    # ATR and ATR%
    df["atr"] = pd.to_numeric(ta.atr(df["high"], df["low"], df["close"], length=14), errors="coerce").fillna(0.0)
    close_nonzero = df["close"].replace(0, pd.NA).ffill().bfill()
    df["atr_pct"] = pd.to_numeric((df["atr"] / close_nonzero), errors="coerce").fillna(0.0)

    # Bollinger width
    df["bb_width"] = _bb_width_safe(df["close"], length=20, stdev=2.0)

    # MAs & slopes (explicitly disable forward-fill on pct_change)
    df["sma50"] = pd.to_numeric(ta.sma(df["close"], 50), errors="coerce")
    df["sma200"] = pd.to_numeric(ta.sma(df["close"], 200), errors="coerce")
    slope50 = df["sma50"].pct_change(5, fill_method=None)
    slope200 = df["sma200"].pct_change(5, fill_method=None)
    df["sma50_slope"] = pd.to_numeric(slope50, errors="coerce").fillna(0.0)
    df["sma200_slope"] = pd.to_numeric(slope200, errors="coerce").fillna(0.0)

    # above SMA200: avoid .fillna on object by making boolean dtype explicitly
    above = df["close"] > df["sma200"].replace({None: float("inf")})
    df["above_sma200"] = above.astype("boolean").fillna(False).astype(bool)

    # Volume analytics
    vol = df["volume"].fillna(0.0)
    vol_mean20 = vol.rolling(20, min_periods=20).mean()
    vol_std20 = vol.rolling(20, min_periods=20).std()
    df["volume_trend"] = pd.to_numeric(((vol - vol_mean20) / vol_std20.replace(0, pd.NA)), errors="coerce").fillna(0.0)
    df["rvol_20"] = pd.to_numeric((vol / vol_mean20.replace(0, pd.NA)), errors="coerce").fillna(0.0)
    df["adv_usd_20"] = pd.to_numeric((df["close"].fillna(0.0) * vol).rolling(20, min_periods=20).mean(), errors="coerce").fillna(0.0)

    # Require last row with all needed fields
    req = [
        "close", "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
        "adx", "atr", "atr_pct", "bb_width", "sma50_slope", "sma200_slope", "above_sma200",
        "volume_trend", "rvol_20", "adv_usd_20", "open", "high", "low"
    ]
    df_valid = df.dropna(subset=req)
    if df_valid.empty:
        df_valid = df.ffill().bfill()
        df_valid = df_valid.infer_objects(copy=False)
        if df_valid.empty:
            raise ValueError("indicators not ready (insufficient data)")

    # Candle signal on the last few bars
    tail = df_valid.tail(5)
    candle_signal, candle_score = _candlestick_signal(tail)

    latest_ts = df_valid.tail(1).index[0]
    latest = df_valid.tail(1).iloc[0]

    return IndicatorPacket(
        symbol=symbol, timeframe=timeframe,
        latest_close=float(latest["close"]),
        rsi=float(latest["rsi"]),
        macd=float(latest["macd"]), macd_signal=float(latest["macd_signal"]), macd_hist=float(latest["macd_hist"]),
        stoch_k=float(latest["stoch_k"]), stoch_d=float(latest["stoch_d"]),
        adx=float(latest["adx"]), atr=float(latest["atr"]), atr_pct=float(latest["atr_pct"]),
        bb_width=float(latest["bb_width"]),
        sma50_slope=float(latest["sma50_slope"]), sma200_slope=float(latest["sma200_slope"]),
        above_sma200=bool(latest["above_sma200"]),
        volume_trend=float(latest["volume_trend"]),
        rvol_20=float(latest["rvol_20"]), adv_usd_20=float(latest["adv_usd_20"]),
        price_time_iso=latest_ts.isoformat(),
        candle_signal=candle_signal, candle_score=float(candle_score),
    )