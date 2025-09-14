# src/llm.py
import json
import subprocess

SYSTEM_PROMPT = """You are a crypto technical analyst.
Return STRICT JSON only with these keys exactly:
bias: "bull" | "bear" | "neutral"
conviction: number between 0 and 1
action: "long" | "short" | "wait"
entry_zone: string
invalidation: string
targets: array of strings
timeframe: string
reasoning: short string (<= 60 words)
risks: array of short strings
Do not return any text that is not valid JSON.
"""

# Default per-call timeout (seconds)
OLLAMA_TIMEOUT = 25

SAFE_FALLBACK = json.dumps({
    "bias": "neutral",
    "conviction": 0.3,
    "action": "wait",
    "entry_zone": "",
    "invalidation": "",
    "targets": [],
    "timeframe": "",
    "reasoning": "Fallback due to model/timeout.",
    "risks": ["model_timeout_or_invalid_json"]
})

def _ollama_run(model: str, prompt: str, images=None, timeout=OLLAMA_TIMEOUT) -> str:
    args = ["ollama", "run", model]
    if images:
        for img in images:
            args += ["--image", img]
    try:
        proc = subprocess.run(
            args,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
        return proc.stdout.decode("utf-8", errors="ignore")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ollama run timed out")

def build_prompt(ind, extra_note=""):
    return f"""{SYSTEM_PROMPT}

Instrument: {ind.symbol}
Timeframe: {ind.timeframe}
LatestClose: {ind.latest_close}
PriceTime: {getattr(ind,'price_time_iso',None)}
Indicators:
- RSI: {ind.rsi:.2f}
- MACD: {ind.macd:.4f} / signal {ind.macd_signal:.4f} / hist {ind.macd_hist:.4f}
- Stoch: {ind.stoch_k:.2f}/{ind.stoch_d:.2f}
- ADX: {ind.adx:.2f}
- ATR: {ind.atr:.6f} (ATR% {ind.atr_pct:.2%})
- BBWidth: {ind.bb_width:.4f}
- SMA50Slope: {ind.sma50_slope:.4f}
- SMA200Slope: {ind.sma200_slope:.4f}
- AboveSMA200: {ind.above_sma200}
- VolumeTrendZ: {ind.volume_trend:.2f}
- RVOL20: {ind.rvol_20:.2f}
- ADV20_USD: {ind.adv_usd_20:.0f}
- CandleSignal: {getattr(ind,'candle_signal','None')} (score {getattr(ind,'candle_score',0):+.1f})
{extra_note}
"""

def _extract_json_or_fallback(raw: str) -> str:
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return SAFE_FALLBACK
    return raw[s:e+1]

def ask_text(model: str, ind):
    try:
        raw = _ollama_run(model, build_prompt(ind))
        return _extract_json_or_fallback(raw)
    except Exception:
        return SAFE_FALLBACK

def ask_vision(model: str, ind, chart_path: str):
    try:
        raw = _ollama_run(model, build_prompt(ind, "\nConsider the attached chart."), images=[chart_path])
        return _extract_json_or_fallback(raw)
    except Exception:
        return SAFE_FALLBACK