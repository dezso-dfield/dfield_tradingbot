from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

from .trade_models import AccountState
from .utils import ensure_dir

PORTFOLIO_PNL_CSV = Path("outputs/pnl/portfolio_pnl.csv")
POSITIONS_PNL_CSV = Path("outputs/pnl/positions_pnl.csv")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_pnl_files_if_needed() -> None:
    """Create outputs/pnl and write CSV headers if files don't exist yet."""
    ensure_dir("outputs/pnl")

    if not PORTFOLIO_PNL_CSV.exists():
        with open(PORTFOLIO_PNL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts_iso",
                "cash_usd",
                "positions_market_value_usd",
                "total_equity_usd",
                "unrealized_pnl_usd",
                "realized_pnl_usd",
                "total_pnl_usd",
                "pnl_pct",
                "open_positions_count",
            ])

    if not POSITIONS_PNL_CSV.exists():
        with open(POSITIONS_PNL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts_iso",
                "symbol",
                "side",
                "qty",
                "entry_price",
                "current_price",
                "market_value_usd",
                "unrealized_pnl_usd",
            ])


def log_portfolio_pnl(state: AccountState, price_map: Dict[str, float]) -> None:
    """
    Append a single portfolio PnL snapshot.
    Handles LONG and SHORT consistently with runner's snapshot.
    """
    _init_pnl_files_if_needed()

    # MTM aggregates (signed for shorts)
    total_mkt_value = 0.0
    total_unreal = 0.0
    for p in state.open_positions:
        cur_px = float(price_map.get(p.symbol, p.entry_price))
        mkt_val = float(cur_px * p.qty)
        # longs: unreal = (cur - entry) * qty; shorts: (entry - cur) * qty
        unreal = (cur_px - p.entry_price) * p.qty if p.side == "long" else (p.entry_price - cur_px) * p.qty
        signed_mkt_val = mkt_val if p.side == "long" else -mkt_val

        total_mkt_value += signed_mkt_val
        total_unreal += unreal

    starting_eq = float(getattr(state, "starting_equity_usd", 10_000.0) or 10_000.0)
    total_equity = float(state.equity_usd) + total_mkt_value
    total_pnl = total_equity - starting_eq
    realized_pnl = total_pnl - total_unreal  # naive realized estimate
    pnl_pct = (total_pnl / starting_eq) if starting_eq else 0.0

    with open(PORTFOLIO_PNL_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            _now_iso(),
            f"{float(state.equity_usd):.2f}",
            f"{total_mkt_value:.2f}",
            f"{total_equity:.2f}",
            f"{total_unreal:.2f}",
            f"{realized_pnl:.2f}",
            f"{total_pnl:.2f}",
            f"{pnl_pct:.6f}",
            len(state.open_positions),
        ])


def log_positions_pnl(state: AccountState, price_map: Dict[str, float]) -> None:
    """
    Append a per-position MTM row for each open position (signed market value for shorts).
    If there are no positions, this is a no-op but file/headers will still exist.
    """
    _init_pnl_files_if_needed()

    ts = _now_iso()
    if not state.open_positions:
        # No positions—still ensure headers exist; nothing else to do.
        return

    with open(POSITIONS_PNL_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for p in state.open_positions:
            cur_px = float(price_map.get(p.symbol, p.entry_price))
            mkt_val = float(cur_px * p.qty)
            unreal = (cur_px - p.entry_price) * p.qty if p.side == "long" else (p.entry_price - cur_px) * p.qty
            signed_mkt_val = mkt_val if p.side == "long" else -mkt_val
            w.writerow([
                ts,
                p.symbol,
                p.side,
                f"{p.qty:.10g}",
                f"{p.entry_price:.10g}",
                f"{cur_px:.10g}",
                f"{signed_mkt_val:.2f}",
                f"{unreal:.2f}",
            ])