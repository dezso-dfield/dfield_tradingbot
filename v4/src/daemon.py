# src/daemon.py
from __future__ import annotations
import argparse
from .runner import run_loop
from .broker import Broker

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interval-sec", type=int, default=300)
    p.add_argument("--universe-size", type=int, default=80)
    p.add_argument("--timeframe", type=str, default="1h")
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--text-model", type=str, default="llama3:instruct")
    p.add_argument("--vision-model", type=str, default="llava")
    p.add_argument("--use-vision", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--use-binance", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--limit", type=int, default=1000)

    # trading knobs
    p.add_argument("--risk-per-trade-pct", type=float, default=0.01)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--cooldown-minutes", type=int, default=60)
    p.add_argument("--fallback-stop-pct", type=float, default=0.03)
    p.add_argument("--min-adv-usd", type=float, default=2_000_000)
    p.add_argument("--max-atr-pct", type=float, default=0.25)

    # execution
    p.add_argument("--exchange", type=str, default="binance")
    p.add_argument("--testnet", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--live", type=lambda x: str(x).lower()=="true", default=False)
    return p.parse_args()

def main():
    args = parse_args()
    broker = Broker(exchange=args.exchange, testnet=args.testnet, live=args.live)
    run_loop(
        interval_sec=args.interval_sec,
        universe_size=args.universe_size,
        timeframe=args.timeframe,
        days=args.days,
        text_model=args.text_model,
        vision_model=args.vision_model,
        use_vision=args.use_vision,
        use_binance=args.use_binance,
        limit=args.limit,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_positions=args.max_positions,
        cooldown_minutes=args.cooldown_minutes,
        fallback_stop_pct=args.fallback_stop_pct,
        min_adv_usd=args.min_adv_usd,
        max_atr_pct=args.max_atr_pct,
        broker=broker,
    )

if __name__ == "__main__":
    # never let top-level exceptions kill the process
    try:
        main()
    except Exception as e:
        # last-resort log to stderr; launchd/tmux will keep it alive anyway
        import sys, traceback
        traceback.print_exc(file=sys.stderr)