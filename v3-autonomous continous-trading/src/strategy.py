from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta, timezone

from .trade_models import AccountState, Position
from .schema import LLMDecision, IndicatorPacket


# ---------------- LLM helpers & parsing ----------------

def _first_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            import re
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
            return float(m.group(0)) if m else default
        if isinstance(x, list) and x:
            return _first_float(x[0], default)
    except Exception:
        return default
    return default


def map_targets(decision: LLMDecision) -> List[float]:
    """Parse any LLM targets to a clean, sorted float list (may be empty)."""
    vals: List[float] = []
    if getattr(decision, "targets", None):
        import re
        for t in decision.targets:
            if isinstance(t, (int, float)):
                vals.append(float(t))
            elif isinstance(t, str):
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
                if m:
                    vals.append(float(m.group(0)))
    return sorted({float(v) for v in vals if v is not None})


def parse_close_invalidation(inv_text: Optional[str]) -> Optional[float]:
    """
    Parse phrases like:
      - "close below 28.5"
      - "daily close under 2.30"
      - "weekly close < 1.05"
    Returns the numeric level if recognized, else None.
    """
    if not inv_text or not isinstance(inv_text, str):
        return None
    import re
    s = inv_text.lower()
    if any(k in s for k in ("close below", "close under", "close <", "close<= ", "close <= ", "close<")):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else None
    return None


def pick_stop(decision: LLMDecision, fallback_pct: float, price: float) -> float:
    """
    Prefer explicit numeric invalidation; else a percent stop (long).
    For shorts (not used yet), runner/ sanitize handles inversion.
    """
    inv = _first_float(getattr(decision, "invalidation", None))
    if inv is not None and inv > 0:
        return float(inv)
    return price * (1.0 - abs(fallback_pct))


def in_entry_band(decision: LLMDecision, price: float) -> bool:
    """If LLM provided an entry range, enforce it."""
    import re
    zone = getattr(decision, "entry_zone", None)
    if not zone:
        return True
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", zone)
    if len(nums) >= 2:
        lo, hi = sorted([float(nums[0]), float(nums[1])])
        return lo <= price <= hi
    return True


def sanitize_levels(
    *,
    price: float,
    atr: float,
    action: str,
    raw_targets: List[float],
    stop_price: float | None,
    fallback_stop_pct: float,
) -> tuple[List[float], float]:
    """
    Clean absurd LLM levels and inject TA-based fallbacks:
    - Drop targets far outside plausible band
      long: within 0.5x..3x and > price
      short: 0.33x..1.2x and < price
    - If no valid targets, ATR ladders: long -> [P+1*ATR, P+2*ATR], short -> [P-1*ATR, P-2*ATR]
    - If stop invalid/missing: P -/+ max(1.2*ATR, fallback% * P) (long/short)
    """
    price = float(price)
    atr = float(max(atr, 1e-12))
    action = (action or "long").lower()

    if action == "short":
        lo_bound, hi_bound = price * 0.33, price * 1.2
    else:
        lo_bound, hi_bound = price * 0.5, price * 3.0

    targets = [t for t in (raw_targets or []) if isinstance(t, (int, float)) and t > 0]
    if action == "short":
        targets = [t for t in targets if lo_bound <= t <= hi_bound and t < price]
    else:
        targets = [t for t in targets if lo_bound <= t <= hi_bound and t > price]

    if not targets:
        if action == "short":
            targets = [price - 1.0 * atr, price - 2.0 * atr]
            targets = [t for t in targets if t > 0]
        else:
            targets = [price + 1.0 * atr, price + 2.0 * atr]

    targets = sorted({round(float(t), 10) for t in targets})

    sp = float(stop_price) if stop_price is not None else 0.0
    if action == "short":
        bad = (sp <= price) or (sp <= 0) or (sp > price * 5.0)
        if bad:
            sp = price + max(1.2 * atr, price * abs(fallback_stop_pct))
    else:
        bad = (sp >= price) or (sp <= 0) or (sp < price * 0.05)
        if bad:
            sp = price - max(1.2 * atr, price * abs(fallback_stop_pct))

    return targets, sp


# ---------------- Policy & risk rails (LLM-led with guards) ----------------

def decide_size(equity_usd: float, price: float, risk_pct: float, stop_price: float, side: str) -> float:
    """
    LLM decides direction/bias; we compute safe size.
    qty = min( risk_based_qty, equity_based_qty )
    """
    risk_usd = max(0.0, risk_pct) * max(0.0, float(equity_usd))
    per_unit_risk = abs(float(price) - float(stop_price))
    risk_qty = (risk_usd / per_unit_risk) if per_unit_risk > 0 else 0.0

    equity_qty = (float(equity_usd) / float(price)) if price > 0 else 0.0
    qty = max(0.0, min(risk_qty, equity_qty))
    return qty


def should_open(decision: LLMDecision, ind: IndicatorPacket) -> bool:
    """
    LLM controls action; we apply minimal regime guards to avoid obviously poor contexts.
    """
    if getattr(decision, "action", None) not in ("long", "short"):
        return False
    if float(getattr(decision, "conviction", 0.0)) < 0.6:
        return False
    # Example regime guard: for longs, require price above SMA200 (remove if you want fully LLM-led)
    if decision.action == "long" and not bool(getattr(ind, "above_sma200", False)):
        return False
    return True


# ---------------- Exit evaluation (hard SL, close-based invalidation, ATR trail) ----------------

def evaluate_exit_rules(
    *,
    decision: LLMDecision,
    ind: IndicatorPacket,
    pos: Position,
    k_atr_trail: float = 1.8,
) -> Dict[str, object]:
    """
    Returns dict with:
      - 'hard_stop_hit': bool  (price <= stop)
      - 'new_trailing_stop': Optional[float] (when ratchet up)
      - 'close_exit': bool (trigger close-at-close invalidation)
      - 'close_exit_reason': str
    """
    out = {
        "hard_stop_hit": False,
        "new_trailing_stop": None,
        "close_exit": False,
        "close_exit_reason": "",
    }

    price = float(getattr(ind, "live_price", ind.latest_close))
    atr = float(getattr(ind, "atr", 0.0))
    action = getattr(decision, "action", "long")

    # 1) Hard tick stop (immediate)
    if pos.stop_price and price <= float(pos.stop_price) and action == "long":
        out["hard_stop_hit"] = True
        return out

    # 2) ATR trailing (only after we’re in profit)
    if action == "long" and atr > 0:
        trail = price - k_atr_trail * atr
        # ratchet only upwards and never above price
        if pos.stop_price is None:
            pass
        else:
            new_stop = max(float(pos.stop_price), trail)
            if new_stop > float(pos.stop_price) and new_stop < price:
                out["new_trailing_stop"] = new_stop

    # 3) Close-based invalidation:
    #    - Explicit: "close below X" from LLM invalidation
    #    - TA guard: close < SMA50 with bearish candle OR RSI breakdown
    close_level = parse_close_invalidation(getattr(decision, "invalidation", None))

    if action == "long":
        conds = []
        reasons = []

        if close_level is not None:
            conds.append(ind.latest_close <= close_level)
            reasons.append(f"close <= {close_level:g}")

        # TA guard: trend & candle deterioration
        if getattr(ind, "sma50_slope", 0.0) < 0 and ind.latest_close < getattr(ind, "sma50", ind.latest_close * 10):
            conds.append(True)
            reasons.append("close < SMA50 & slope<0")
        if getattr(ind, "candle_score", 0.0) <= -2.0:
            conds.append(True)
            reasons.append("bearish candle")

        # optional RSI guard
        if getattr(ind, "rsi", 50.0) < 40.0:
            conds.append(True)
            reasons.append("RSI<40")

        if any(conds):
            out["close_exit"] = True
            out["close_exit_reason"] = "; ".join(reasons)

    return out


# ---------------- Cooldown helpers ----------------

def cooldown_active(state: AccountState, symbol: str, now: datetime) -> bool:
    iso = state.cooldown_until.get(symbol)
    if not iso:
        return False
    try:
        until = datetime.fromisoformat(iso)
        return now < until
    except Exception:
        return False


def set_cooldown(state: AccountState, symbol: str, minutes: int):
    until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    state.cooldown_until[symbol] = until.isoformat()