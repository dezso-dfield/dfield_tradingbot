import mplfinance as mpf
import pandas as pd
from .utils import ensure_dir

def save_chart(df: pd.DataFrame, symbol: str, out_dir: str="outputs/charts", mas=(20,50,200)) -> str:
    ensure_dir(out_dir)
    dfc = df.copy()
    dfc.index = pd.DatetimeIndex(dfc["ts"])
    ap = []
    for w in mas:
        dfc[f"SMA{w}"] = dfc["close"].rolling(w).mean()
        ap.append(mpf.make_addplot(dfc[f"SMA{w}"]))
    path = f"{out_dir}/{symbol}.png"
    mpf.plot(dfc.set_index("ts")[["open","high","low","close","volume"]],
             type="candle", volume=True, addplot=ap, savefig=path)
    return path