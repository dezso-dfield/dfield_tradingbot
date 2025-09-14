from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List

from rich import print

from .universe import top_coins
from .fetch import ohlcv_daily_coingecko, binance_klines, binance_last_price
from .ta import compute_indicators, compute_indicators_mtf
from .llm import ask_text_mtf
from .synth import parse_llm_json
from .rank import rank_ideas
from .report import write_signals_json
from .store import load_state, save_state
from .broker import Broker
from .strategy import (
    decide_size, should_open, set_cooldown, map_targets, pick_stop,
    in_entry_band, sanitize_levels, evaluate_exit_rules, style_params
)
from .utils import ensure_dir

TRADES_CSV = Path("outputs/run_logs/trades.csv")
DAEMON_LOG = Path("outputs/run_logs/daemon.log")
POSITIONS_JSON = Path("outputs/positions/current_positions.json")
POSITIONS_CSV = Path("outputs/positions/current_positions.csv")
EQUITY_TS_CSV = Path("outputs/run_logs/equity_timeseries.csv")


def safe(val, default=0.0): return default if val is None else val


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


def _write_positions_snapshot(state, price_map: dict[str, float]) -> None:
    POSITIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = []; total_mkt_value = 0.0; total_unreal = 0.0

    for p in state.open_positions:
        cur_px = float(price_map.get(p.symbol, p.entry_price))
        mkt_val = float(cur_px * p.qty); unreal = float((cur_px - p.entry_price) * p.qty)
        total_mkt_value += mkt_val; total_unreal += unreal
        data.append({
            "symbol": p.symbol, "side": p.side, "qty": p.qty, "entry_price": p.entry_price,
            "current_price": cur_px, "market_value_usd": mkt_val, "unrealized_pnl_usd": unreal,
            "stop_price": p.stop_price, "targets": p.targets, "filled_targets": p.filled_targets,
            "opened_iso": p.opened_iso, "last_update_iso": p.last_update_iso,
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
            mkt_val = float(cur_px * p.qty); unreal = float((cur_px - p.entry_price) * p.qty)
            w.writerow([
                p.symbol, p.side, f"{p.qty:.10g}", f"{p.entry_price:.10g}",
                f"{cur_px:.10g}", f"{mkt_val:.2f}", f"{unreal:.2f}",
                "" if p.stop_price is None else f"{float(p.stop_price):.10g}",
                "|".join(f"{float(t):.10g}" for t in (p.targets or [])),
                "|".join(f"{float(t):.10g}" for t in (p.filled_targets or [])),
                p.opened_iso, p.last_update_iso
            ])

    EQUITY_TS_CSV.parent.mkdir(parents=True, exist_ok=True)
    first = not EQUITY_TS_CSV.exists()
    with open(EQUITY_TS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow([
                "ts_iso","cash_usd","positions_market_value_usd","total_equity_usd",
                "unrealized_pnl_usd","starting_equity_usd","total_pnl_usd","pnl_pct","open_positions_count"
            ])
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            f"{float(state.equity_usd):.2f}",
            f"{total_mkt_value:.2f}",
            f"{total_equity:.2f}",
            f"{total_unreal:.2f}",
            f"{float(state.starting_equity_usd):.2f}",
            f"{total_pnl:.2f}",
            f"{pnl_pct:.6f}",
            len(state.open_positions)
        ])


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
    risk_per_trade_pct: float,   # kept for fallback; style overrides it
    max_positions: int,
    cooldown_minutes: int,       # kept for fallback; style overrides it
    fallback_stop_pct: float,
    min_adv_usd: float,
    max_atr_pct: float,
    broker: Broker,
) -> None:
    state = load_state()
    ensure_dir("outputs/top10_charts"); ensure_dir("outputs/run_logs"); ensure_dir("outputs/positions")
    now = datetime.now(timezone.utc); now_iso = now.isoformat()

    if not hasattr(state, "starting_equity_usd") or state.starting_equity_usd is None:
        state.starting_equity_usd = float(getattr(state, "equity_usd", 10_000.0))

    # ---- scan with MTF ----
    uni = top_coins(universe_size)
    pairs: List[Tuple] = []
    latest_price_map: dict[str, float] = {}
    MTF_LIST = ["1h", "4h", "1d", "1w"]

    for coin_id, sym in uni:
        try:
            mtf = compute_indicators_mtf(
                symbol=sym, coin_id=coin_id, timeframes=MTF_LIST,
                use_binance=use_binance, binance_limit=limit, coingecko_days=days
            )
            if not mtf: 
                continue

            # choose primary packet for price/logs
            primary = mtf.get(timeframe) or mtf.get("1h") or mtf.get("1d")
            if use_binance:
                live = binance_last_price(f"{sym}USDT")
                primary.live_price = float(live) if live is not None else float(primary.latest_close)
            else:
                primary.live_price = float(primary.latest_close)
            latest_price_map[sym] = float(primary.live_price)

            raw = ask_text_mtf(text_model, sym, mtf)
            dec = parse_llm_json(sym, primary.timeframe, raw)
            pairs.append((primary, dec))

        except Exception as e:
            print(f"[yellow]{sym} skipped: {e}")

    ranked = rank_ideas(pairs)
    write_signals_json("outputs/signals.json", ranked)

    score_map = {r.symbol: r.score for r in ranked}
    pair_map = {i.symbol: (i, d) for (i, d) in pairs}

    daemon_lines = []
    def dlog(msg: str):
        print(msg); daemon_lines.append(f"{now_iso} {msg}")

    def get_price(sym: str) -> float:
        if sym in latest_price_map: return latest_price_map[sym]
        if use_binance:
            lp = binance_last_price(f"{sym}USDT")
            if lp is not None:
                latest_price_map[sym] = float(lp)
                return float(lp)
        for p in state.open_positions:
            if p.symbol == sym: return float(p.entry_price)
        return 0.0

    # ---- manage existing positions ----
    for pos in list(state.open_positions):
        sym = pos.symbol
        tup = pair_map.get(sym)
        price = get_price(sym)

        if tup is not None:
            ind, dec = tup
            params = style_params(dec)
            ev = evaluate_exit_rules(decision=dec, ind=ind, pos=pos, k_atr_trail=params["trail_k"])
            if ev["new_trailing_stop"] is not None:
                pos.stop_price = float(ev["new_trailing_stop"])
                pos.last_update_iso = datetime.now(timezone.utc).isoformat()
                dlog(f"[blue]TRAIL {sym} -> SL {pos.stop_price:.6g}[/]")
            if ev["close_exit"]:
                broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, ind.latest_close, pos.qty)
                _append_trade_row(now_iso, "CLOSE_INV", sym, pos.side, pos.qty, ind.latest_close,
                                  state.equity_usd, stop_price=pos.stop_price, targets=pos.targets,
                                  note=ev.get("close_exit_reason", "close_based"))
                state.open_positions = [p for p in state.open_positions if p.symbol != sym]
                set_cooldown(state, sym, params["cooldown_min"])
                dlog(f"[red]CLOSE_INV {sym} @ {ind.latest_close:.6g} ({ev.get('close_exit_reason','')})[/]")
                continue

        if pos.stop_price and price > 0 and price <= pos.stop_price:
            broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, price, pos.qty)
            _append_trade_row(now_iso, "STOP", sym, pos.side, pos.qty, price, state.equity_usd,
                              stop_price=pos.stop_price, targets=pos.targets, note="stop_hit")
            state.open_positions = [p for p in state.open_positions if p.symbol != sym]
            # fallback cooldown if we don't have fresh decision
            set_cooldown(state, sym, cooldown_minutes)
            dlog(f"[red]STOP {sym} @ {price:.6g}[/]")
            continue

        if tup is not None:
            ind, _ = tup
            price = float(safe(getattr(ind, "live_price", ind.latest_close)))
            remaining = [t for t in pos.targets if t not in pos.filled_targets]
            for t in sorted(remaining):
                if price >= t:
                    tp_qty = pos.qty * (0.5 if not pos.filled_targets else 1.0)
                    broker.market_sell(state, f"{sym}/USDT" if broker.live else sym, price, tp_qty)
                    _append_trade_row(now_iso, "TP", sym, pos.side, tp_qty, price, state.equity_usd,
                                      stop_price=pos.stop_price, targets=pos.targets, note=f"tp@{t}")
                    pos.qty -= tp_qty; pos.filled_targets.append(t)
                    pos.last_update_iso = datetime.now(timezone.utc).isoformat()
                    dlog(f"[green]TP {sym} at {t:.6g} qty={tp_qty:.6g}[/]")
                    if len(pos.filled_targets) == 1:
                        pos.stop_price = max(safe(pos.stop_price, 0.0), pos.entry_price)
                        dlog(f"[blue]MOVE SL {sym} -> BE {pos.stop_price:.6g}[/]")
                    if pos.qty <= 0:
                        state.open_positions = [p for p in state.open_positions if p.symbol != sym]
                        set_cooldown(state, sym, cooldown_minutes)
                    break

    # ---- open new / replace weakest if at capacity ----
    MAX_REPLACE_MARGIN = 8.0
    open_syms = {p.symbol for p in state.open_positions}

    def worst_open():
        if not state.open_positions: return None, None
        w = min(state.open_positions, key=lambda p: score_map.get(p.symbol, -1e9))
        return w, score_map.get(w.symbol, -1e9)

    for r in ranked:
        sym = r.symbol
        if sym in open_syms: continue
        ind_dec = pair_map.get(sym)
        if ind_dec is None: continue
        ind, dec = ind_dec

        if not should_open(dec, ind): continue
        price = float(safe(getattr(ind, "live_price", ind.latest_close)))
        if not in_entry_band(dec, price): continue

        params = style_params(dec)

        if len(state.open_positions) >= max_positions:
            wpos, wscore = worst_open()
            if wpos is None or r.score <= (wscore if wscore is not None else -1e9) + MAX_REPLACE_MARGIN:
                continue
            exit_px = get_price(wpos.symbol)
            broker.market_sell(state, f"{wpos.symbol}/USDT" if broker.live else wpos.symbol, exit_px, wpos.qty)
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

        # style-aware risk
        qty = decide_size(state.equity_usd, price, params["risk_pct"], stop_price, getattr(dec, "action", "long"))
        if qty * price > state.equity_usd:
            qty = max(0.0, state.equity_usd / price)
        if qty <= 0: continue

        try:
            broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, price, qty)
        except Exception as e:
            affordable_qty = max(0.0, state.equity_usd / price)
            if affordable_qty > 0:
                broker.market_buy(state, f"{sym}/USDT" if broker.live else sym, price, affordable_qty)
                qty = affordable_qty
            else:
                dlog(f"[yellow]SKIP OPEN {sym}: {e}"); continue

        from .trade_models import Position  # local import to avoid circulars
        state.open_positions.append(
            Position(symbol=sym, side=getattr(dec, "action", "long"), qty=qty, entry_price=price,
                     stop_price=stop_price, targets=targets)
        )
        set_cooldown(state, sym, params["cooldown_min"])
        notional = qty * price
        direction = getattr(dec, "action", "long")
        _append_trade_row(now_iso, "OPEN", sym, direction, qty, price, state.equity_usd,
                          stop_price=stop_price, targets=targets, note=f"style={getattr(dec,'style',None)}")
        dlog(f"[cyan]OPEN {sym} style={getattr(dec,'style',None)} dir={direction} "
             f"qty={qty:.6g} @ {price:.6g} notional=${notional:,.2f} SL={stop_price:.6g} TGTS={targets}[/]")

    # ---- persist & logs ----
    state.last_run_iso = now_iso; save_state(state)
    pos_price_map = {p.symbol: get_price(p.symbol) for p in state.open_positions}
    _write_positions_snapshot(state, pos_price_map)

    DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DAEMON_LOG, "a", encoding="utf-8") as f:
        for line in daemon_lines: f.write(line + "\n")


def run_loop(interval_sec: int = 300, **kwargs):
    print(f"[bold]Continuous monitor[/] every {interval_sec}s (Ctrl+C to stop)")
    while True:
        try:
            run_once(**kwargs)
        except KeyboardInterrupt:
            print("[yellow]Interrupted by user[/]"); break
        except Exception as e:
            print(f"[red]Loop error:[/] {e}")
        time.sleep(interval_sec)