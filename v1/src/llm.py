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

def _ollama_run(model: str, prompt: str, images=None) -> str:
    args = ["ollama", "run", model]
    if images:
        for img in images:
            args += ["--image", img]
    proc = subprocess.run(args, input=prompt.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout.decode("utf-8", errors="ignore")

def build_prompt(ind, extra_note=""):
    return f"""{SYSTEM_PROMPT}

Instrument: {ind.symbol}
Timeframe: {ind.timeframe}
LatestClose: {ind.latest_close}
PriceTime: {ind.price_time_iso}
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
- CandleSignal: {ind.candle_signal} (score {ind.candle_score:+.1f})
{extra_note}
"""

def ask_text(model: str, ind):
    raw = _ollama_run(model, build_prompt(ind))
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("Model did not return JSON")
    return raw[s:e+1]

def ask_vision(model: str, ind, chart_path: str):
    raw = _ollama_run(model, build_prompt(ind, "\nConsider the attached chart."), images=[chart_path])
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("Vision model did not return JSON")
    return raw[s:e+1]