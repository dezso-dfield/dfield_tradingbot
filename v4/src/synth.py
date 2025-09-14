from __future__ import annotations
import json
from typing import Any, Dict
from .schema import LLMDecision

def parse_llm_json(symbol: str, timeframe: str, raw: str) -> LLMDecision:
    """
    Robustly parse LLM JSON:
    - Extract outermost JSON object if wrapper text present.
    - Inject symbol/timeframe if missing.
    - Let LLMDecision handle coercions (action/bias/targets).
    """
    if raw is None:
        return LLMDecision(symbol=symbol, timeframe=timeframe, action="flat", bias="neutral", conviction=0.5)

    # Extract JSON object if extra text exists
    if isinstance(raw, str):
        import re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            raw_json = m.group(0)
        else:
            raw_json = raw
        try:
            data = json.loads(raw_json)
        except Exception:
            # try another quick repair: strip backticks / code fences
            raw_json2 = raw_json.strip().strip("`").strip()
            try:
                data = json.loads(raw_json2)
            except Exception:
                # final fallback: empty data
                data = {}
    elif isinstance(raw, dict):
        data = raw
    else:
        data = {}

    # Inject required fields if missing
    data.setdefault("symbol", symbol)
    data.setdefault("timeframe", timeframe)

    try:
        return LLMDecision(**data)
    except Exception:
        # Final defensive fallback to keep the scan running
        return LLMDecision(symbol=symbol, timeframe=timeframe, action="flat", bias="neutral", conviction=0.5)