import json
from pydantic import ValidationError
from .schema import LLMDecision

def parse_llm_json(symbol: str, timeframe: str, json_str: str) -> LLMDecision:
    data = json.loads(json_str)
    data["symbol"] = symbol
    data["timeframe"] = timeframe
    try:
        return LLMDecision(**data)
    except ValidationError:
        # tolerant fallback
        return LLMDecision(
            symbol=symbol, timeframe=timeframe,
            bias=str(data.get("bias","neutral")),
            conviction=float(data.get("conviction",0.3)),
            action=str(data.get("action","wait")),
            entry_zone=str(data.get("entry_zone","")),
            invalidation=str(data.get("invalidation","")),
            targets=list(data.get("targets", [])),
            reasoning=str(data.get("reasoning","fallback")),
            risks=list(data.get("risks", ["formatting"])),
        )