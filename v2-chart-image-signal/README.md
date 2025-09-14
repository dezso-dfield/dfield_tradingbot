# Crypto LLM Scanner (Free, Local)
A fully free, local-first crypto scanner that:
- pulls **top 50** coins (CoinGecko, free),
- fetches free OHLCV (CoinGecko daily or Binance public klines),
- computes TA indicators,
- (optional) renders charts,
- asks **Ollama** LLMs (text or vision) for a structured decision,
- ranks trade ideas and writes `outputs/signals.json`.

> **Not financial advice. Research & education only.**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama models (free, local)
# https://ollama.com
ollama pull llama3:instruct
ollama pull llava   # optional vision

# Run a daily scan (CoinGecko)
python src/main.py --timeframe 1d --days 180 --universe-size 50

# Run with charts + vision
python src/main.py --timeframe 1d --days 180 --use-vision true