from __future__ import annotations
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field, model_validator, field_validator

# ----- Packets passed around the pipeline -----

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
    atr_pct: float
    bb_width: float
    sma50_slope: float
    sma200_slope: float
    above_sma200: bool
    volume_trend: float
    rvol_20: float
    adv_usd_20: float
    live_price: Optional[float] = None
    price_time_iso: Optional[str] = None
    candle_signal: Optional[str] = None
    candle_score: Optional[float] = None
    notes: Optional[str] = None

# ----- LLM decision (robust to sloppy outputs) -----

TradeStyle = Literal["scalp", "swing", "position", "hodl"]
Action     = Literal["long", "short", "flat"]
Bias       = Literal["bull", "bear", "neutral"]

class LLMDecision(BaseModel):
    symbol: str
    timeframe: str
    action: Action = "flat"
    bias:   Bias   = "neutral"
    conviction: float = 0.5
    entry_zone: Optional[str] = None
    invalidation: Optional[str] = None
    # Accept numbers/strings/nested lists from the LLM → coerce to floats
    targets: List[float] = Field(default_factory=list)

    # Multi-timeframe / style-aware hints
    style: Optional[TradeStyle] = None
    max_hold_bars: Optional[int] = None
    risk_pct_override: Optional[float] = None
    trail_atr_k_override: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any):
        if not isinstance(data, dict):
            return data

        # action
        raw_action = (str(data.get("action", "")) or "").strip().lower()
        data["action"] = {
            "buy": "long", "go long": "long", "long": "long",
            "sell": "short", "go short": "short", "short": "short",
            "flat": "flat", "no trade": "flat", "none": "flat", "hold": "flat",
        }.get(raw_action, "flat")

        # bias
        raw_bias = (str(data.get("bias", "")) or "").strip().lower()
        data["bias"] = {
            "bullish": "bull", "bull": "bull",
            "bearish": "bear", "bear": "bear",
            "neutral": "neutral", "sideways": "neutral",
        }.get(raw_bias, "neutral")

        # conviction
        try:
            conv = float(data.get("conviction", 0.5))
        except Exception:
            conv = 0.5
        data["conviction"] = max(0.0, min(1.0, conv))

        # targets → List[float]
        raw_targets = data.get("targets", [])
        if raw_targets is None:
            raw_targets = []
        if not isinstance(raw_targets, list):
            raw_targets = [raw_targets]

        def _flatten(xs):
            for x in xs:
                if isinstance(x, (list, tuple)):
                    for y in _flatten(x):
                        yield y
                else:
                    yield x

        parsed: List[float] = []
        import re
        for item in _flatten(raw_targets):
            if isinstance(item, (int, float)):
                parsed.append(float(item))
            elif isinstance(item, str):
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", item)
                if m:
                    try:
                        parsed.append(float(m.group(0)))
                    except Exception:
                        pass
        data["targets"] = sorted(set(parsed))

        # entry_zone/invalidation as strings if present
        for k in ("entry_zone", "invalidation"):
            v = data.get(k, None)
            if v is not None and not isinstance(v, str):
                data[k] = str(v)

        # style normalization
        s = (str(data.get("style", "")) or "").strip().lower()
        data["style"] = s if s in ("scalp", "swing", "position", "hodl") else None

        return data

    @field_validator("timeframe")
    @classmethod
    def _tf_default(cls, v: str):
        return v or "1h"

    class Config:
        extra = "ignore"

# ----- Ranked output for printing -----

class RankedIdea(BaseModel):
    symbol: str
    score: float
    decision: LLMDecision