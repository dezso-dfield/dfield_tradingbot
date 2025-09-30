from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Tuple, Any

LEARN_DIR = Path("outputs/learn")
WEIGHTS_JSON = LEARN_DIR / "weights.json"

# --------- storage ----------
def load_weights() -> Dict[str, float]:
    try:
        if WEIGHTS_JSON.exists():
            return json.loads(WEIGHTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_weights(w: Dict[str, float]) -> None:
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_JSON.write_text(json.dumps(w, indent=2), encoding="utf-8")

# --------- features ----------
def _bucket(x: float | None, edges: Tuple[float, ...]) -> str:
    if x is None:
        return "na"
    for i, e in enumerate(edges):
        if x < e:
            return f"<{e}"
    return f">={edges[-1]}"

def features_for(ind, dec) -> Dict[str, str]:
    """
    Produce a coarse feature set — small, stable, explainable.
    """
    side = (getattr(dec, "action", "wait") or "wait").lower()
    tf = getattr(ind, "timeframe", "?")
    rsi = getattr(ind, "rsi", None)
    atr_pct = getattr(ind, "atr_pct", None)
    cndl = getattr(ind, "candle_score", None)
    conv = getattr(dec, "conviction", 0.0)
    above200 = bool(getattr(ind, "above_sma200", False))

    return {
        "side": side,                         # long / short / wait
        "tf": str(tf),                        # 1h / 4h / 1d
        "rsi": _bucket(rsi, (35, 45, 55, 65, 75)),
        "atrp": _bucket(atr_pct, (0.02, 0.05, 0.10, 0.20, 0.35)),
        "cndl": _bucket(cndl, (-3, -1, 0, 1, 3)),
        "conv": _bucket(conv, (0.5, 0.6, 0.7, 0.8, 0.9)),
        "trend": "above200" if above200 else "below200",
    }

def _keys_from_feats(F: Dict[str,str]) -> list[str]:
    # include individual & small combos (keeps it sparse and robust)
    k = []
    k += [f"{a}={b}" for a,b in F.items()]
    # pairwise interactions we care about
    combos = [
        ("side","trend"), ("side","rsi"), ("side","atrp"),
        ("side","cndl"), ("side","conv"), ("trend","rsi"),
    ]
    for a,b in combos:
        k.append(f"{a}={F[a]}|{b}={F[b]}")
    # one triple that often matters
    k.append(f"side={F['side']}|trend={F['trend']}|atrp={F['atrp']}")
    return k

# --------- online update ----------
def update_reward(weights: Dict[str,float], ind, dec, reward: float, lr: float=0.15, clip: float=2.0) -> None:
    """
    reward: +1 good / -1 bad (you can pass fractional rewards too)
    We apply EMA updates with a small learning rate and clipping.
    """
    F = features_for(ind, dec)
    keys = _keys_from_feats(F)
    r = max(-clip, min(clip, float(reward)))
    for k in keys:
        old = float(weights.get(k, 0.0))
        new = (1.0 - lr) * old + lr * r
        weights[k] = new

def score_boost(weights: Dict[str,float], ind, dec, max_boost: float=10.0) -> float:
    """
    Sum relevant weights with a small shrinkage → clamp to [-max_boost, +max_boost].
    """
    F = features_for(ind, dec)
    keys = _keys_from_feats(F)
    s = sum(float(weights.get(k, 0.0)) for k in keys)
    # shrink to keep it sane on sparse data
    s *= 0.6
    return max(-max_boost, min(max_boost, s))