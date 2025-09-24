# src/runner.py
from __future__ import annotations

import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Callable, Any

from rich import print

from .universe import top_coins
from .fetch import ohlcv_daily_coingecko, binance_klines, binance_last_price
from .ta import compute_indicators
from .llm import ask_text, ask_vision
from .synth import parse_llm_json
from .rank import rank_ideas
from .report import write_signals_json
from .store import load_state, save_state
from .broker import Broker
from .strategy import (
    decide_size, should_open, set_cooldown, map_targets, pick_stop,
    in_entry_band, sanitize_levels, evaluate_exit_rules
)
from .utils import ensure_dir

TRADES_CSV = Path("outputs/run_logs/trades.csv")
DAEMON_LOG = Path("outputs/run_logs/daemon.log")
HEARTBEAT = Path("outputs/run_logs/heartbeat.txt")
POSITIONS_JSON = Path("outputs/positions/current_positions.json")
POSITIONS_CSV = Path("outputs/positions/current_positions.csv")
EQUITY_TS_CSV = Path("outputs/run_logs/equity_timeseries.csv")
SCAN_DEBUG_CSV = Path("outputs/run_logs/scan_debug.csv")

# skip these from trading
STABLES_OR_NONPAIRS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDD","USDP","USDE","BSC-USD","CBBTC",
    "WETH","WBETH","WBTC","LEO","CRO","WSTETH","STETH","WEETH","USDS","HYPE","FIGR_HELOC"
}

# ------------------ small retry helper ------------------
def retry_call(fn: Callable, *, tries: int = 3, delay: float = 1.0, backoff: float = 2.0, on_error_msg: str = "") -> Any:
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if on_error_msg:
                print(f"[yellow]{on_error_msg} (attempt {attempt}/{tries}): {e}")
            time.sleep(delay)
            delay *= backoff
    return None

def safe(val, default=0.0):
    return default if val is None else val

def _append_trade_row(ts: str, action: str, symbol: str, side: str, qty: float, price: float,
                      equity_usd: float, stop_price=None, targets=None, note: str = ""):
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    first = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow([
                "ts_iso", "action", "symbol", "direction", "qty", "price",
                "notional_usd", "equity_usd", "stop_price", "targets", "note"
            ])
        notional = (qty or 0.0) * (price or 0.0)
        w.writerow([
            ts, action, symbol, side, f"{qty:.10g}", f"{price:.10g}",
            f"{notional:.2f}", f"{equity_usd:.2f}",
            "" if stop_price is None else f"{float(stop_price):.10g}",
            "" if not targets else "|".join(f"{float(t):.10g}" for t in targets),
            note
        ])

def _append_scan_debug(rows: List[List[str]]):
    SCAN_DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    first = not SCAN_DEBUG_CSV.exists()
    with open(SCAN_DEBUG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["ts_iso","symbol","stage","detail"])
        for r in rows:
            w.writerow(r)

def _write_positions_snapshot(state, price_map: dict[str, float]) -> None:
    POSITIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = []
    total_mkt_value = 0.0
    total_unreal = 0.0

    for p in state.open_positions:
        cur_px = float(price_map.get(p.symbol, p.entry_price))
        mkt_val = float(cur_px * p.qty)
        unreal = float((cur_px - p.entry_price) * p.qty) if p.side == "long" else float((p.entry_price - cur_px) * p.qty)
        total_mkt_value += mkt_val if p.side == "long" else -mkt_val  # show signed for shorts
        total_unreal += unreal
        data.append({
            "symbol": p.symbol,
            "side": p.side,
            "qty": p.qty,
            "entry_price": p.entry_price,
            "current_price": cur_px,
            "market_value_usd": mkt_val if p.side == "long" else -mkt_val,
            "unrealized_pnl_usd": unreal,
            "stop_price": p.stop_price,
            "targets": p.targets,
            "filled_targets": p.filled_targets,
            "opened_iso": p.opened_iso,
            "last_update_iso": p.last_update_iso,
        })

    if not hasattr(state, "starting_equity_usd") or state.starting_equity_usd is None:
        state.starting_equity_usd = float(getattr(state, "equity_usd", 10_000.0))

    total_equity = float(state.equity_usd) + total_mkt_value
    total_pnl = total_equity - float(state.starting_equity_usd)
    pnl_pct = (total_pnl / float(state.starting_equity_usd)) if state.starting_equity_usd else 0.0

    with open(POSITIONS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "cash_usd": state.equity_usd,
            "positions_market_value_usd": total_mkt_value,
            "total_equity_usd": total_equity,
            "unrealized_pnl_usd": total_unreal,
            "starting_equity_usd": state.starting_equity_usd,
            "total_pnl_usd": total_pnl,
            "pnl_pct": pnl_pct,
            "open_positions": data
        }, f, indent=2)

    with open(POSITIONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol","side","qty","entry_price","current_price","market_value_usd",
            "unrealized_pnl_usd","stop_price","targets","filled_targets","opened_iso","last_update_iso"
        ])
        for p in state.open_positions:
            cur_px = float(price_map.get(p.symbol, p.entry_price))
            mkt_val = float(cur_px * p.qty)
            unreal = float((cur_px - p.entry_price) * p.qty) if p.side == "long" else float((p.entry_price - cur_px) * p.qty)
            w.writerow([
                p.symbol, p.side, f"{p.qty:.10g}", f"{p.entry_price:.10g}",
                f"{cur_px:.10g}", f"{(mkt_val if p.side=='long' else -mkt_val):.2f}", f"{unreal:.2f}",
                "" if p.stop_price is None else f"{float(p.stop_price):.10g}",
                "|".join(f"{float(t):.10g}" for t in (p.targets or [])),
                "|".join(f"{float(t):.10g}" for t in (p.filled_targets or [])),
                p.opened_iso, p.last_update_iso
            ])

def _touch_heartbeat():
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    with open(HEARTBEAT, "w", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat())

def run_once(
    *,
    universe_size: int,
    timeframe: str,
    days: int,
    text_model: str,
    vision_model: str,
    use_vision: bool,
    use_binance: bool,
    limit: int,
    risk_per_trade_pct: float,
    max_positions: int,
    cooldown_minutes: int,
    fallback_stop_pct: float,
    min_adv_usd: float,
    max_atr_pct: float,
    broker: Broker,
) -> None:
    """
    Full scan + trade management cycle (supports LONG and SHORT),
    with defensive retries, per-symbol time guards and detailed debug logging.
    """
    state = load_state()
    ensure_dir("outputs/top10_charts"); ensure_dir("outputs/run_logs"); ensure_dir("outputs/positions")
    now = datetime.now(timezone.utc); now_iso = now.isoformat()

    if not hasattr(state, "starting_equity_usd") or state.starting_equity_usd is None:
        state.starting_equity_usd = float(getattr(state, "equity_usd", 10_000.0))

    uni = retry_call(lambda: top_coins(universe_size), tries=3, delay=1.0, backoff=2.0, on_error_msg="universe fetch failed")
    if not uni:
        print("[yellow]Universe empty after retries; skipping trading cycle[/]")
        return

    pairs: List[Tuple] = []
    latest_price_map: dict[str, float] = {}
    debug_rows: List[List[str]] = []

    PER_SYMBOL_MAX_SEC = 15.0

    for coin_id, sym in uni:
        t0 = time.time()
        try:
            if sym in STABLES_OR_NONPAIRS:
                debug_rows.append([now_iso, sym, "SKIP", "stable_or_nonpair"])
                continue

            if use_binance:
                sym_usdt = f"{sym}USDT"
                df = retry_call(lambda: binance_klines(sym_usdt, interval=timeframe, limit=limit),
                                tries=3, delay=1.0, backoff=2.0, on_error_msg=f"{sym}: klines failed")
                if df is None or len(df) < 60:
                    debug_rows.append([now_iso, sym, "SKIP", "no_or_short_data"])
                    continue
                tf = timeframe
            else:
                df = retry_call(
                    lambda: ohlcv_daily_coingecko(coin_id, days=days),
                    tries=3,
                    delay=1.0,
                    backoff=2.0,
                    on_error_msg=f"{sym}: coingecko failed"
                )
                if df is None or len(df) < 60:
                    debug_rows.append([now_iso, sym, "SKIP", "no_or_short_data"])
                    continue
                tf = "1d"

            ind = compute_indicators(df, sym, tf)

            # liquidity/volatility filters
            if safe(getattr(ind, "adv_usd_20", 0.0)) < min_adv_usd:
                debug_rows.append([now_iso, sym, "SKIP", f"min_adv_usd<{min_adv_usd}"])
                continue
            if safe(getattr(ind, "atr_pct", 0.0)) > max_atr_pct:
                debug_rows.append([now_iso, sym, "SKIP", f"atr_pct>{max_atr_pct}"])
                continue

            # live price
            if use_binance:
                lp = retry_call(lambda: binance_last_price(f"{sym}USDT"),
                                tries=3, delay=0.5, backoff=1.7, on_error_msg=f"{sym}: last price failed")
                ind.live_price = float(lp) if lp is not None else float(ind.latest_close)
            else:
                ind.live_price = float(ind.latest_close)
            latest_price_map[sym] = float(ind.live_price)

            # LLM decision (always parse, even if raw is None -> "{}")
            raw = retry_call(lambda: (ask_vision(vision_model, ind, "") if use_vision else ask_text(text_model, ind)),
                             tries=2, delay=1.0, backoff=2.0, on_error_msg=f"{sym}: LLM call failed")
            dec = parse_llm_json(sym, ind.timeframe, raw or "{}")
            pairs.append((ind, dec))
            debug_rows.append([now_iso, sym, "OK", f"action={dec.action}, bias={dec.bias}, conv={dec.conviction:.2f}"])

        except Exception as e:
            print(f"[yellow]{sym} skipped: {e}")
            traceback.print_exc()
            debug_rows.append([now_iso, sym, "ERROR", str(e)])
        finally:
            if time.time() - t0 > PER_SYMBOL_MAX_SEC:
                print(f"[yellow]{sym} exceeded {PER_SYMBOL_MAX_SEC}s; moving on[/]")

    _append_scan_debug(debug_rows)

    ranked = rank_ideas(pairs)
    write_signals_json("outputs/signals.json", ranked)

    score_map = {r.symbol: r.score for r in ranked}
    pair_map = {i.symbol: (i, d) for (i, d) in pairs}

    daemon_lines = []
    def dlog(msg: str):
        print(msg)
        daemon_lines.append(f"{now_iso} {msg}")

    def get_price(sym: str) -> float:
        if sym in latest_price_map:
            return latest_price_map[sym]
        if use_binance:
            lp = retry_call(lambda: binance_last_price(f"{sym}USDT"),
                            tries=3, delay=0.5, backoff=1.7, on_error_msg=f"{sym}: last price failed")
            if lp is not None:
                latest_price_map[sym] = float(lp)
                return float(lp)
        for p in state.open_positions:
            if p.symbol == sym:
                return float(p.entry_price)
        return 0.0

    # ---- manage existing positions (LONG & SHORT) ----
    for pos in list(state.open_positions):
        sym = pos.symbol
        tup = pair_map.get(sym)
        price = get_price(sym)

        try:
            if tup is not None:
                ind, dec = tup
                ev = evaluate_exit_rules(decision=dec, ind=ind, pos=pos, k_atr_trail=1.8)

                if ev["new_trailing_stop"] is not None:
                    pos.stop_price = float(ev["new_trailing_stop"])
                    pos.last_update_iso = datetime.now(timezone.utc).isoformat()
                    dlog(f"[blue]TRAIL {sym} -> SL {pos.stop_price:.6g}[/]")

                if ev["close_exit"]:
                    # close by side
                    if pos.side == "long":
                        broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, ind.latest_close, pos.qty)
                    else:
                        broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, ind.latest_close, pos.qty)
                    _append_trade_row(now_iso, "CLOSE_INV", sym, pos.side, pos.qty, ind.latest_close,
                                      state.equity_usd, stop_price=pos.stop_price, targets=pos.targets,
                                      note=ev.get("close_exit_reason", "close_based"))
                    state.open_positions = [p for p in state.open_positions if p.symbol != sym]
                    set_cooldown(state, sym, cooldown_minutes)
                    dlog(f"[red]CLOSE_INV {sym} @ {ind.latest_close:.6g} ({ev.get('close_exit_reason','')})[/]")
                    continue

            # hard stop (tick-based): diff for short vs long
            if pos.stop_price and price > 0:
                if (pos.side == "long" and price <= pos.stop_price) or (pos.side == "short" and price >= pos.stop_price):
                    if pos.side == "long":
                        broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, price, pos.qty)
                    else:
                        broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, price, pos.qty)
                    _append_trade_row(now_iso, "STOP", sym, pos.side, pos.qty, price, state.equity_usd,
                                      stop_price=pos.stop_price, targets=pos.targets, note="stop_hit")
                    state.open_positions = [p for p in state.open_positions if p.symbol != sym]
                    set_cooldown(state, sym, cooldown_minutes)
                    dlog(f"[red]STOP {sym} @ {price:.6g}[/]")
                    continue

            # profit targets
            if tup is not None:
                ind, _ = tup
                price = float(safe(getattr(ind, "live_price", ind.latest_close)))
                remaining = [t for t in pos.targets if t not in pos.filled_targets]
                for t in sorted(remaining, reverse=(pos.side=="short")):
                    hit = (price >= t) if pos.side == "long" else (price <= t)
                    if hit:
                        tp_qty = pos.qty * (0.5 if not pos.filled_targets else 1.0)
                        if pos.side == "long":
                            broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, price, tp_qty)
                        else:
                            broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, price, tp_qty)
                        _append_trade_row(now_iso, "TP", sym, pos.side, tp_qty, price, state.equity_usd,
                                          stop_price=pos.stop_price, targets=pos.targets, note=f"tp@{t}")
                        pos.qty -= tp_qty
                        pos.filled_targets.append(t)
                        pos.last_update_iso = datetime.now(timezone.utc).isoformat()
                        dlog(f"[green]TP {sym} at {t:.6g} qty={tp_qty:.6g}[/]")
                        # move SL to BE after first TP
                        if len(pos.filled_targets) == 1:
                            if pos.side == "long":
                                pos.stop_price = max(safe(pos.stop_price, 0.0), pos.entry_price)
                            else:
                                pos.stop_price = min(safe(pos.stop_price, float('inf')), pos.entry_price)
                            dlog(f"[blue]MOVE SL {sym} -> BE {pos.stop_price:.6g}[/]")
                        if pos.qty <= 0:
                            state.open_positions = [p for p in state.open_positions if p.symbol != sym]
                            set_cooldown(state, sym, cooldown_minutes)
                        break
        except Exception as e:
            dlog(f"[yellow]manage loop error {sym}: {e}")
            traceback.print_exc()

    # ---- open new / replace weakest (support short & long) ----
    MAX_REPLACE_MARGIN = 8.0
    open_syms = {p.symbol for p in state.open_positions}

    def worst_open():
        if not state.open_positions:
            return None, None
        w = min(state.open_positions, key=lambda p: score_map.get(p.symbol, -1e9))
        return w, score_map.get(w.symbol, -1e9)

    for r in ranked:
        sym = r.symbol
        if sym in open_syms:
            continue
        ind_dec = pair_map.get(sym)
        if ind_dec is None:
            continue
        ind, dec = ind_dec

        try:
            if not should_open(dec, ind):
                continue
            price = float(safe(getattr(ind, "live_price", ind.latest_close)))
            if not in_entry_band(dec, price):
                continue

            if len(state.open_positions) >= max_positions:
                wpos, wscore = worst_open()
                if wpos is None or r.score <= (wscore if wscore is not None else -1e9) + MAX_REPLACE_MARGIN:
                    continue
                exit_px = get_price(wpos.symbol)
                # close worst by side
                if wpos.side == "long":
                    broker.market_sell(state, f"{wpos.symbol}/USDT" if broker.live else wpos.symbol, exit_px, wpos.qty)
                else:
                    broker.market_buy(state, f"{wpos.symbol}/USDT" if broker.live else wpos.symbol, exit_px, wpos.qty)
                _append_trade_row(now_iso, "REPLACE_CLOSE", wpos.symbol, wpos.side, wpos.qty, exit_px, state.equity_usd,
                                  stop_price=wpos.stop_price, targets=wpos.targets, note=f"replaced_by={sym}")
                state.open_positions = [p for p in state.open_positions if p.symbol != wpos.symbol]
                dlog(f"[yellow]REPLACE {wpos.symbol} (score {wscore:.1f}) -> {sym} (score {r.score:.1f})[/]")

            raw_stop = pick_stop(dec, fallback_stop_pct, price)
            raw_targets = map_targets(dec)
            targets, stop_price = sanitize_levels(
                price=price,
                atr=float(safe(getattr(ind, "atr", 0.0))),
                action=getattr(dec, "action", "long"),
                raw_targets=raw_targets,
                stop_price=raw_stop,
                fallback_stop_pct=fallback_stop_pct,
            )

            side = getattr(dec, "action", "long")
            qty = decide_size(state.equity_usd, price, risk_per_trade_pct, stop_price, side)
            if qty * price > state.equity_usd and side == "long":
                qty = max(0.0, state.equity_usd / price)
            if qty <= 0:
                continue

            # open by side
            if side == "long":
                try:
                    broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, price, qty)
                except Exception:
                    affordable_qty = max(0.0, state.equity_usd / price)
                    if affordable_qty > 0:
                        broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, price, affordable_qty)
                        qty = affordable_qty
                    else:
                        dlog(f"[yellow]SKIP OPEN {sym}: insufficient equity")
                        continue
            else:
                # opening a short = sell first
                broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, price, qty)

            from .trade_models import Position
            state.open_positions.append(
                Position(symbol=sym, side=side, qty=qty, entry_price=price,
                         stop_price=stop_price, targets=targets)
            )
            set_cooldown(state, sym, cooldown_minutes)
            notional = qty * price
            _append_trade_row(now_iso, "OPEN", sym, side, qty, price, state.equity_usd,
                              stop_price=stop_price, targets=targets, note=f"open_{side}")
            dlog(f"[cyan]OPEN {sym} dir={side} qty={qty:.6g} @ {price:.6g} "
                 f"notional=${notional:,.2f} SL={stop_price:.6g} TGTS={targets}[/]")
        except Exception as e:
            dlog(f"[yellow]open loop error {sym}: {e}")
            traceback.print_exc()

    # ---- persist & logs ----
    state.last_run_iso = now_iso
    save_state(state)

    pos_price_map = {p.symbol: get_price(p.symbol) for p in state.open_positions}
    _write_positions_snapshot(state, pos_price_map)

    DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DAEMON_LOG, "a", encoding="utf-8") as f:
        for line in daemon_lines:
            f.write(line + "\n")

    _touch_heartbeat()

def run_loop(interval_sec: int = 300, **kwargs):
    print(f"[bold]Continuous monitor[/] every {interval_sec}s (Ctrl+C to stop)")
    backoff = 1.0
    while True:
        try:
            run_once(**kwargs)
            backoff = 1.0
        except KeyboardInterrupt:
            print("[yellow]Interrupted by user[/]")
            break
        except Exception as e:
            print(f"[red]Loop error (top-level):[/] {e}")
            traceback.print_exc()
            with open(DAEMON_LOG, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} TOP-LEVEL ERROR: {e}\n")
        time.sleep(max(interval_sec, min(interval_sec * backoff, 3600)))
        backoff = min(backoff * 2.0, 12.0)