from __future__ import annotations

import json
import re
import requests
from typing import Dict
from .schema import IndicatorPacket

OLLAMA_URL = "http://localhost:11434/api/generate"  # default ollama endpoint


def _ollama_generate_json(model: str, prompt: str, max_repair: int = 1) -> str:
    """
    Call Ollama's /api/generate, expect JSON. If response has extra text,
    try to extract the outermost JSON object. Optionally one repair pass.
    """
    resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    txt = resp.json().get("response", "")

    # Extract strict JSON object if wrapper text exists
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        txt = m.group(0)

    # Optional one-shot repair by asking model to fix to strict JSON
    if max_repair:
        try:
            json.loads(txt)
        except Exception:
            reprompt = (
                "The following text must be valid JSON. Return only valid JSON, nothing else.\n\n" + txt
            )
            r2 = requests.post(OLLAMA_URL, json={"model": model, "prompt": reprompt, "stream": False}, timeout=120)
            r2.raise_for_status()
            cand = r2.json().get("response", "")
            m2 = re.search(r"\{.*\}", cand, flags=re.S)
            if m2:
                txt = m2.group(0)
    return txt


def ask_text(model: str, ind: IndicatorPacket) -> str:
    """
    Your original single-timeframe text call (kept for compatibility).
    """
    payload = {
        "symbol": ind.symbol,
        "timeframe": ind.timeframe,
        "price": ind.latest_close,
        "rsi": ind.rsi, "macd": ind.macd, "macd_hist": ind.macd_hist,
        "stoch_k": ind.stoch_k, "adx": ind.adx, "atr": ind.atr, "atr_pct": ind.atr_pct,
        "bb_width": ind.bb_width,
        "sma50_slope": ind.sma50_slope, "sma200_slope": ind.sma200_slope,
        "above_sma200": ind.above_sma200,
        "rvol_20": ind.rvol_20, "adv_usd_20": ind.adv_usd_20,
        "candle_signal": ind.candle_signal, "candle_score": ind.candle_score,
    }
    system = (
        "You are a crypto trading assistant. Respond ONLY with valid JSON matching keys: "
        "symbol,timeframe,action,bias,conviction,entry_zone,invalidation,targets,style,max_hold_bars,"
        "risk_pct_override,trail_atr_k_override"
    )
    user = {
        "task": "Decide long/short/flat with entry_zone, invalidation, 1-3 targets and trading style.",
        "indicators": payload
    }
    prompt = json.dumps({"system": system, "user": user})
    return _ollama_generate_json(model, prompt)


def ask_text_mtf(model: str, symbol: str, mtf: Dict[str, IndicatorPacket]) -> str:
    """
    Multi-timeframe request. Ask model to choose a trade style & decision.
    """
    def pack(pkt: IndicatorPacket):
        return {
            "timeframe": pkt.timeframe,
            "price": pkt.latest_close,
            "rsi": pkt.rsi, "macd": pkt.macd, "macd_hist": pkt.macd_hist,
            "stoch_k": pkt.stoch_k, "adx": pkt.adx, "atr": pkt.atr, "atr_pct": pkt.atr_pct,
            "bb_width": pkt.bb_width,
            "sma50_slope": pkt.sma50_slope, "sma200_slope": pkt.sma200_slope,
            "above_sma200": pkt.above_sma200,
            "rvol_20": pkt.rvol_20, "adv_usd_20": pkt.adv_usd_20,
            "candle_signal": pkt.candle_signal, "candle_score": pkt.candle_score,
        }
    mtf_payload = {tf: pack(pkt) for tf, pkt in mtf.items()}

    system = (
        "You are a crypto trading assistant. Respond ONLY with valid JSON.\n"
        "Required keys: symbol,timeframe,action,bias,conviction,targets.\n"
        "Optional: entry_zone,invalidation,style,max_hold_bars,risk_pct_override,trail_atr_k_override.\n"
        "Constraints:\n"
        "- action ∈ ['long','short','flat']\n"
        "- bias   ∈ ['bull','bear','neutral'] (use 'bull' or 'bear', not 'bullish'/'bearish')\n"
        "- targets must be an array of numbers (no strings, no nested arrays)\n"
        "- If unsure, set action='flat' and conviction=0.4–0.6\n"
    )
    user = {
        "symbol": symbol,
        "multi_timeframe": mtf_payload,
        "instructions": (
            "Choose style among ['scalp','swing','position','hodl'] based on alignment:\n"
            "- If 1h bullish but 1d/1w bearish -> scalp.\n"
            "- If 1h/4h/1d aligned -> swing.\n"
            "- If 1d/1w aligned with strong trend -> position or hodl.\n"
            "Provide entry_zone, invalidation (numeric or 'close below X'), and 1-3 targets. "
            "If uncertain, action='flat'."
        )
    }
    prompt = json.dumps({"system": system, "user": user})
    return _ollama_generate_json(model, prompt)