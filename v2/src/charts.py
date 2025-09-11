# src/charts.py
import re
from typing import Iterable, List, Optional, Tuple

import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir

def _sanitize_df(dfc: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime index, sorted, unique, and drop obvious junk
    dfc = dfc.copy()
    dfc["ts"] = pd.to_datetime(dfc["ts"], errors="coerce", utc=True)
    dfc = dfc.dropna(subset=["ts"])
    dfc = dfc.sort_values("ts").drop_duplicates(subset=["ts"])
    # Keep only columns we plot
    cols = [c for c in ["ts","open","high","low","close","volume"] if c in dfc.columns]
    return dfc[cols]

def save_chart(df: pd.DataFrame, symbol: str, out_dir: str="outputs/charts", mas=(20,50,200)) -> str:
    ensure_dir(out_dir)
    dfc = _sanitize_df(df)
    dfc.index = pd.DatetimeIndex(dfc["ts"])
    ap = []
    for w in mas:
        dfc[f"SMA{w}"] = dfc["close"].rolling(w).mean()
        ap.append(mpf.make_addplot(dfc[f"SMA{w}"]))
    path = f"{out_dir}/{symbol}.png"
    mpf.plot(
        dfc.set_index("ts")[["open","high","low","close","volume"]],
        type="candle",
        volume=True,
        addplot=ap,
        style="yahoo",
        figratio=(16, 9),
        figscale=1.6,
        # IMPORTANT: avoid bbox_inches="tight" to prevent giant canvas issues
        savefig=dict(fname=path, dpi=180, pad_inches=0.2),
        update_width_config=dict(candle_linewidth=0.8, candle_width=0.6, volume_width=0.6),
    )
    return path

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _to_floats(x) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        vals = []
        for item in x:
            if isinstance(item, (int, float)):
                vals.append(float(item))
            elif isinstance(item, str):
                vals += [float(m.group(0)) for m in _NUM.finditer(item)]
        return vals
    if isinstance(x, str):
        return [float(m.group(0)) for m in _NUM.finditer(x)]
    return []

def _entry_band_from_text(entry_text: Optional[str]) -> Optional[Tuple[float, float]]:
    fs = sorted(_to_floats(entry_text))
    if len(fs) >= 2:
        return (fs[0], fs[1])
    return None

def save_annotated_top_chart(
    df: pd.DataFrame,
    symbol: str,
    decision,
    ind,
    out_dir: str = "outputs/top10_charts",
    mas: Iterable[int] = (20, 50, 200),
    lookback: int = 200
) -> str:
    ensure_dir(out_dir)
    dfc = _sanitize_df(df.tail(max(lookback, 60)))  # cap lookback, keep >=60 rows
    dfc.index = pd.DatetimeIndex(dfc["ts"])

    # MAs
    ap = []
    for w in mas:
        dfc[f"SMA{w}"] = dfc["close"].rolling(w).mean()
        ap.append(mpf.make_addplot(dfc[f"SMA{w}"]))

    fig, axes = mpf.plot(
        dfc.set_index("ts")[["open","high","low","close","volume"]],
        type="candle",
        volume=True,
        addplot=ap,
        style="yahoo",
        figratio=(16, 9),
        figscale=1.7,
        returnfig=True,
        # no tight bbox to avoid giant image bug
        update_width_config=dict(candle_linewidth=0.8, candle_width=0.6, volume_width=0.6),
    )
    ax_price = axes[0]

    current_price = getattr(ind, "live_price", None) or getattr(ind, "latest_close", None)
    targets = _to_floats(getattr(decision, "targets", None))
    invalidation_vals = _to_floats(getattr(decision, "invalidation", None))
    entry_band = _entry_band_from_text(getattr(decision, "entry_zone", None))

    if current_price is not None:
        ax_price.axhline(current_price, linestyle="-", linewidth=1.3, color="black", alpha=0.7)
        ax_price.text(dfc.index[-1], current_price, f"  Price {current_price:.4g}",
                      va="center", ha="left", fontsize=9, color="black")

    if invalidation_vals:
        inv = invalidation_vals[0]
        ax_price.axhline(inv, linestyle="--", linewidth=1.1, color="red", alpha=0.85)
        ax_price.text(dfc.index[int(len(dfc)*0.02)], inv, f"Invalidation {inv:.4g}",
                      va="bottom", ha="left", color="red", fontsize=9)

    for i, t in enumerate(sorted(targets)):
        ax_price.axhline(t, linestyle="--", linewidth=1.0, color="green", alpha=0.8)
        ax_price.text(dfc.index[int(len(dfc)*0.98)], t, f"T{i+1} {t:.4g}  ",
                      va="bottom", ha="right", color="green", fontsize=9)

    if entry_band:
        lo, hi = entry_band
        lo, hi = min(lo, hi), max(lo, hi)
        ax_price.axhspan(lo, hi, color="blue", alpha=0.08)
        ax_price.text(dfc.index[int(len(dfc)*0.02)], hi,
                      f"Entry {lo:.4g}–{hi:.4g}", va="top", ha="left", color="blue", fontsize=9)

    info_lines = [
        f"{symbol}  |  {decision.action.upper()}  {decision.bias}  (conf {decision.conviction:.2f})",
        f"Candle: {getattr(ind,'candle_signal','None')}  |  ATR%: {getattr(ind,'atr_pct',0.0):.1%}  RVOL20: {getattr(ind,'rvol_20',0.0):.2f}",
        f"Reason: {decision.reasoning[:140]}",
    ]
    risks = getattr(decision, "risks", []) or []
    if risks:
        info_lines.append("Risks: " + "; ".join(risks[:3]))
    text = "\n".join(info_lines)

    bbox_props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#888")
    ax_price.text(0.01, 0.98, text, transform=ax_price.transAxes,
                  va="top", ha="left", fontsize=9, bbox=bbox_props)

    path = f"{out_dir}/{symbol}.png"
    fig.savefig(path, dpi=180)  # no bbox_inches="tight"
    plt.close(fig)
    return path