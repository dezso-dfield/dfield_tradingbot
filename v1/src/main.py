import argparse
import csv
from datetime import datetime, timezone 
from pathlib import Path
from rich import print

from .constants import NOT_FINANCIAL_ADVICE
from .universe import top_coins
from .fetch import (
    ohlcv_daily_coingecko,
    binance_klines,
    binance_last_price,
)
# If you added the availability helper, uncomment this and use it below:
# from .fetch import binance_spot_usdt_symbols

from .ta import compute_indicators
from .charts import save_chart
from .llm import ask_text, ask_vision
from .synth import parse_llm_json
from .rank import rank_ideas
from .report import write_signals_json
from .utils import ensure_dir

STABLES_OR_NONPAIRS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDD","USDP","USDE","BSC-USD","CBBTC","WETH","WBETH","WBTC","LEO","CRO",
    "WSTETH","STETH","WEETH","USDS","HYPE","FIGR_HELOC"
}

def sym_to_binance(symbol: str) -> str:
    return f"{symbol}USDT"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--universe-size", type=int, default=50)
    p.add_argument("--timeframe", type=str, default="1d")  # with Binance: e.g., 4h,1h
    p.add_argument("--days", type=int, default=180)        # CoinGecko lookback days
    p.add_argument("--use-vision", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--text-model", type=str, default="llama3:instruct")
    p.add_argument("--vision-model", type=str, default="llava")
    p.add_argument("--use-binance", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--limit", type=int, default=1000)      # Binance klines limit (clamped in fetch)
    return p.parse_args()

def main():
    args = parse_args()
    print(f"[bold green]LLM Crypto Scanner[/] — {NOT_FINANCIAL_ADVICE}")
    ensure_dir("outputs/charts")
    ensure_dir("outputs/run_logs")

    # Optional: prefetch tradable USDT symbols to cut noise (requires helper in fetch.py)
    avail_usdt = None
    # if args.use_binance:
    #     print("[cyan]Fetching Binance tradable USDT symbols…[/]")
    #     avail_usdt = binance_spot_usdt_symbols()

    universe = top_coins(args.universe_size)
    pairs = []

    for coin_id, symbol in universe:
        # Skip stables and known non-tradable symbols to reduce noise
        if symbol in STABLES_OR_NONPAIRS:
            print(f"[yellow]Skip {symbol}: stablecoin or non-tradable on Binance[/]")
            continue
        try:
            if args.use_binance:
                symbol_usdt = sym_to_binance(symbol)
                if avail_usdt and symbol_usdt not in avail_usdt:
                    print(f"[yellow]Skip {symbol}: no {symbol_usdt} on Binance spot[/]")
                    continue
                df = binance_klines(symbol_usdt, interval=args.timeframe, limit=args.limit)
                if df is None or len(df) < 60:
                    print(f"[yellow]Skip {symbol}: no Binance data or too short[/]")
                    continue
                tf = args.timeframe
            else:
                if args.timeframe != "1d":
                    print(f"[yellow]{symbol}: CoinGecko supports daily; forcing TF=1d for this symbol[/]")
                df = ohlcv_daily_coingecko(coin_id, days=args.days)
                if len(df) < 60:
                    print(f"[yellow]Skip {symbol}: not enough daily data[/]")
                    continue
                tf = "1d"

            # Compute indicators (includes candle analysis if you applied that patch)
            ind = compute_indicators(df, symbol=symbol, timeframe=tf)

            # Live price (Binance) for logging / display
            if args.use_binance:
                live = binance_last_price(sym_to_binance(symbol))
                if live is not None:
                    ind.live_price = float(live)

            # Ask LLM (text or vision)
            if args.use_vision:
                chart_path = save_chart(df.tail(180), symbol)
                raw = ask_vision(args.vision_model, ind, chart_path)
            else:
                raw = ask_text(args.text_model, ind)

            dec = parse_llm_json(symbol, ind.timeframe, raw)
            pairs.append((ind, dec))

        except Exception as e:
            print(f"[red][{symbol}] error:[/] {e}")

    ranked = rank_ideas(pairs)
    write_signals_json("outputs/signals.json", ranked)

    # CSV run log with current price, candle signals, etc. (append)
    log_path = "outputs/run_logs/scan_log.csv"
    first_time = not Path(log_path).exists()
    ind_map = {ind.symbol: ind for (ind, _d) in pairs}
    ts_run = datetime.now(timezone.utc).isoformat()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first_time:
            writer.writerow([
                "ts_run_iso","symbol","timeframe","price_time_iso",
                "latest_close","live_price","action","bias","conviction",
                "entry_zone","invalidation","targets","candle_signal","candle_score"
            ])
        for r in ranked:
            d = r.decision
            ind = ind_map.get(r.symbol)
            writer.writerow([
                ts_run,
                r.symbol,
                d.timeframe,
                getattr(ind, "price_time_iso", None),
                getattr(ind, "latest_close", None),
                getattr(ind, "live_price", None),
                d.action,
                d.bias,
                f"{d.conviction:.2f}",
                d.entry_zone,
                d.invalidation,
                "|".join(d.targets or []),
                getattr(ind, "candle_signal", None),
                getattr(ind, "candle_score", None),
            ])

    print("\n[bold]Top 10 ideas[/]:")
    for x in ranked[:10]:
        d = x.decision
        print(f"{x.symbol:<8} score={x.score:6.1f}  {d.action.upper():5} {d.bias:<7} conf={d.conviction:.2f}  inv={d.invalidation}  tgts={d.targets}")

if __name__ == "__main__":
    main()