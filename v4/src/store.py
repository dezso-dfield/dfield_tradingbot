# src/store.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from .trade_models import AccountState, Position

STATE_PATH = Path("outputs/state.json")

def _state_from_dict(d: Dict[str, Any]) -> AccountState:
    # Backward-compatible loader: handle missing fields gracefully
    equity = float(d.get("equity_usd", 10_000.0))
    starting = d.get("starting_equity_usd", None)
    if starting is not None:
        starting = float(starting)

    # positions
    pos_list = []
    for p in d.get("open_positions", []):
        pos_list.append(Position(
            symbol=p["symbol"],
            side=p.get("side", "long"),
            qty=float(p["qty"]),
            entry_price=float(p["entry_price"]),
            stop_price=(None if p.get("stop_price") in (None, "", "null") else float(p["stop_price"])),
            targets=[float(x) for x in p.get("targets", []) or []],
            filled_targets=[float(x) for x in p.get("filled_targets", []) or []],
            opened_iso=p.get("opened_iso"),
            last_update_iso=p.get("last_update_iso"),
        ))

    state = AccountState(
        equity_usd=equity,
        starting_equity_usd=starting,
        open_positions=pos_list,
        cooldown_until=d.get("cooldown_until", {}),
        last_run_iso=d.get("last_run_iso"),
    )

    # If starting equity missing, set it now to the current equity
    if state.starting_equity_usd is None:
        state.starting_equity_usd = float(state.equity_usd)

    return state

def load_state() -> AccountState:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        # fresh state
        st = AccountState()
        st.starting_equity_usd = float(st.equity_usd)
        save_state(st)
        return st
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _state_from_dict(data)

def save_state(state: AccountState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "equity_usd": float(state.equity_usd),
        "starting_equity_usd": float(state.starting_equity_usd) if state.starting_equity_usd is not None else None,
        "open_positions": [{
            "symbol": p.symbol,
            "side": p.side,
            "qty": float(p.qty),
            "entry_price": float(p.entry_price),
            "stop_price": (None if p.stop_price is None else float(p.stop_price)),
            "targets": [float(x) for x in (p.targets or [])],
            "filled_targets": [float(x) for x in (p.filled_targets or [])],
            "opened_iso": p.opened_iso,
            "last_update_iso": p.last_update_iso,
        } for p in state.open_positions],
        "cooldown_until": state.cooldown_until,
        "last_run_iso": state.last_run_iso,
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)