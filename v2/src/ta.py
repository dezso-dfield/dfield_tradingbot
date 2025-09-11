"""
Comprehensive indicators:
- Robust BB width (fallback if pandas_ta names differ)
- RVOL(20), ADV USD(20), ATR%
- Candlestick pattern analysis (engulfing, hammer/star, doji, inside, marubozu)
"""
import pandas as pd
import pandas_ta as ta
from .schema import IndicatorPacket

# ---------- Bollinger safe ----------
def _bb_width_safe(close: pd.Series, length: int = 20, stdev: float = 2.0) -> pd.Series:
    bb = ta.bbands(close, length=length, std=stdev)
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        # try several common names
        for u_name in (f"BBU_{length}_{stdev}", f"BBU_{length}_{float(stdev)}", "BBU_20_2.0"):
            if u_name in bb.columns:
                upper = bb[u_name]
                break
        else:
            upper = None
        for l_name in (f"BBL_{length}_{stdev}", f"BBL_{length}_{float(stdev)}", "BBL_20_2.0"):
            if l_name in bb.columns:
                lower = bb[l_name]
                break
        else:
            lower = None
        if upper is not None and lower is not None:
            return (upper - lower) / close

    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std()
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    return (upper - lower) / close

# ---------- Candlestick detectors ----------
def _candle_stats(o, h, l, c):
    body = (c - o).abs()
    rng = (h - l).abs().replace(0, 1e-9)
    upper = (h - o.where(o>c, c))
    lower = (o.where(o<c, c) - l)
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
    # current inside previous
    prev_h, prev_l = h.shift(1), l.shift(1)
    return (h <= prev_h) & (l >= prev_l)

def _engulfing(o, c):
    # bullish: current up, prev down, body engulfs prev body
    prev_o, prev_c = o.shift(1), c.shift(1)
    cur_up = c > o
    prev_down = prev_c < prev_o
    bull = cur_up & prev_down & (o <= prev_c) & (c >= prev_o)

    # bearish
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

    # Score last bar
    s = 0.0
    label = []
    if bull_eng.iloc[-1]:
        s += 3.0; label.append("Bullish Engulfing")
    if bear_eng.iloc[-1]:
        s -= 3.0; label.append("Bearish Engulfing")
    if hammer.iloc[-1]:
        s += 2.0; label.append("Hammer")
    if star.iloc[-1]:
        s -= 2.0; label.append("Shooting Star")
    if doji.iloc[-1]:
        label.append("Doji")
    if maru.iloc[-1]:
        label.append("Marubozu")
    if inside.iloc[-1]:
        label.append("Inside Bar")

    text = ", ".join(label) if label else "None"
    return text, float(s)

# ---------- Main compute ----------
def compute_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> IndicatorPacket:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)

    # Momentum / trend / vol
    df["rsi"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"] = macd.iloc[:, 2]

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]

    adx = ta.adx(df["high"], df["low"], df["close"])
    if adx is not None and not adx.empty:
        df["adx"] = adx.iloc[:, 0]

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_pct"] = df["atr"] / df["close"]

    df["bb_width"] = _bb_width_safe(df["close"], length=20, stdev=2.0)

    # MAs & slopes
    df["sma50"] = ta.sma(df["close"], 50)
    df["sma200"] = ta.sma(df["close"], 200)
    df["sma50_slope"] = df["sma50"].pct_change(5)
    df["sma200_slope"] = df["sma200"].pct_change(5)
    df["above_sma200"] = df["close"] > df["sma200"]

    # Volume analytics
    vol_mean20 = df["volume"].rolling(20, min_periods=20).mean()
    vol_std20  = df["volume"].rolling(20, min_periods=20).std()
    df["volume_trend"] = (df["volume"] - vol_mean20) / (vol_std20.replace(0, pd.NA))
    df["rvol_20"] = df["volume"] / vol_mean20
    df["adv_usd_20"] = (df["close"] * df["volume"]).rolling(20, min_periods=20).mean()

    # Require last row with all needed fields
    req = [
        "close","rsi","macd","macd_signal","macd_hist","stoch_k","stoch_d",
        "adx","atr","atr_pct","bb_width","sma50_slope","sma200_slope","above_sma200",
        "volume_trend","rvol_20","adv_usd_20","open","high","low"
    ]
    df_valid = df.dropna(subset=req)
    if df_valid.empty:
        raise ValueError("indicators not ready (insufficient data)")

    # Candle signal on the last few bars (use all but only return last)
    candle_signal, candle_score = _candlestick_signal(df_valid.tail(5))

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
        candle_signal=candle_signal, candle_score=candle_score
    )