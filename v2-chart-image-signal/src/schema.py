from pydantic import BaseModel, Field
from typing import List, Optional

class IndicatorPacket(BaseModel):
    symbol: str
    timeframe: str
    latest_close: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    stoch_k: float
    stoch_d: float
    adx: float
    atr: float
    atr_pct: float                 # NEW
    bb_width: float
    sma50_slope: float
    sma200_slope: float
    above_sma200: bool
    volume_trend: float            # z-score(20)
    rvol_20: float                 # NEW
    adv_usd_20: float              # NEW
    live_price: Optional[float] = None        # NEW (real-time if Binance)
    price_time_iso: Optional[str] = None      # NEW (timestamp for latest candle)
    candle_signal: Optional[str] = None       # NEW (e.g., "Bullish Engulfing")
    candle_score: Optional[float] = None      # NEW (strength heuristic)
    notes: Optional[str] = None

class LLMDecision(BaseModel):
    symbol: str
    bias: str                 # bull | bear | neutral
    conviction: float         # 0..1
    action: str               # long | short | wait
    entry_zone: Optional[str] = None
    invalidation: Optional[str] = None
    targets: Optional[List[str]] = None
    timeframe: str
    reasoning: str
    risks: List[str] = Field(default_factory=list)

class RankedIdea(BaseModel):
    symbol: str
    score: float
    decision: LLMDecision