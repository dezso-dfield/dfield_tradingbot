import requests
from typing import List, Tuple
from .constants import COINGECKO
from .utils import sleep_polite

def top_coins(n: int) -> List[Tuple[str, str]]:
    """Return [(coingecko_id, SYMBOL_UPPER), ...] for top n by market cap."""
    out = []
    page = 1
    while len(out) < n:
        r = requests.get(
            f"{COINGECKO}/coins/markets",
            params={"vs_currency":"usd","order":"market_cap_desc","per_page":250,"page":page}
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for c in data:
            out.append((c["id"], c["symbol"].upper()))
            if len(out) == n:
                return out
        page += 1
        sleep_polite(0.9)
    return out